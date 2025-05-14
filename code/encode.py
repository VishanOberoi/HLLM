import os
import numpy as np
import argparse
from tqdm import tqdm 
import torch
import pickle
import sys

from cProfile import run
from logging import getLogger
import torch
import json

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.utils.ClueWeb22Api import ClueWeb22Api, create_shards

from REC.data import *
from REC.data.dataset.collate_fn import customize_rmpad_collate
from REC.data.dataset import BatchTextDataset

from REC.trainer import Trainer


import lightning as L
from lightning.fabric.strategies import DeepSpeedStrategy, DDPStrategy




class ClueWebBatchTextDataset(Dataset):
    """
    ClueWeb dataset for item encoding. 
    """

    def __init__(self, config, args):

        self.args = args

        # ClueWeb22 location on disk 
        self.dataset_dir = "/data/datasets/clueweb22/ClueWeb22_B"

        # get the cwid - id map 
        self.id_to_cwid = {}
        # the list of internal item ids 
        self.item_list = [] 
        with open(self.args.id_map_path, "r") as f:
            for line in f:  
                parts = line.strip().split("\t")
                self.id_to_cwid[int(parts[1])] = parts[0]
                # the encode data is the internal ids 
                self.item_list.append(int(parts[1]))
   
        if self.args.dataset_number_of_shards > 1:
            self.item_list = create_shards(
                data=self.item_list, 
                num_shards=self.args.dataset_number_of_shards, 
                index=self.args.dataset_shard_index
            )
        print(f"EncodeDataset_ClueWeb22 shard {self.args.dataset_shard_index} length: {len(self.item_list)}")

        self.item_num = len(self.id_to_cwid) # total number of items 
        self.max_text_length = config['MAX_TEXT_LENGTH']
        self.device = config['device']
        self.text_keys = config['text_keys']
        self.tokenizer = AutoTokenizer.from_pretrained(config['item_pretrain_dir'], trust_remote_code=True)

        self.item_prompt = config['item_prompt']
        self.item_emb_token_n = config['item_emb_token_n']


    def __len__(self):
        # the current shard 
        return len(self.item_list)
        

    def __getitem__(self, index):

        def get_features(index): 
            # extract required webpage contents from ClueWeb22Api
            clueweb_api = ClueWeb22Api(cweb_doc_id, self.dataset_dir)
            clean_txt = eval(clueweb_api.get_clean_text())
            content = clean_txt["Clean-Text"]
            title = content.split('\n')[0].replace("\n", "").replace("\t", "").replace("\r", "").replace("\'", "").replace("\"", "").strip()
            content = content.replace("\n", "").replace("\t", "").replace("\r", "").replace("\'", "").replace("\"", "").strip()
            item_feature = {
                "title": title, 
                "description": content
            }
            if 'tag' in self.text_keys: 
                topics = ",".join(clueweb_api.get_topics()) 
                item_feature['tag'] = topics
            return item_feature

        def process_item(cweb_doc_id):
            # tokenize webpage 
            item_i = get_features(cweb_doc_id)
            text_str = ""
            if len(item_i):
                text_str = f"{self.item_prompt}"
                for key in self.text_keys: # ['title', 'tag', 'description']
                    value = item_i[key]
                    if value and str(value) != 'nan':
                        text_str += f"{key}: {value}"
                        
            ids = self.tokenizer.encode(text_str)
            ids = ids[:self.max_text_length]
            mask = [1] * len(ids)
            return ids, mask

        # document processing 
        id_ = self.item_list[index] # internal_cwid 
        # get the corresponing cwid 
        cweb_doc_id = self.id_to_cwid[id_] 

        # get the tokenized ids
        pos_input_ids, pos_cu_input_lens, pos_position_ids = [], [], []
        ids, _ = process_item(cweb_doc_id)

        # flash attention prep 
        pos_input_ids.extend(ids + [0] * self.item_emb_token_n)
        pos_cu_input_lens.append(len(ids) + self.item_emb_token_n)
        pos_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())
        outputs = {
            "pos_item_ids": torch.as_tensor(index, dtype=torch.int64),
            "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64),
            "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),
            "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64)
        }

        return outputs


    def __skip__(self, num_to_skip): 
        self.item_list = self.item_list[num_to_skip:]


def convert_str(s):
    try:
        if s.lower() == 'none':
            return None
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        float_val = float(s)
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except ValueError:
        print(f"Unable to convert the string '{s}' to None / Bool / Float / Int, retaining the original string.")
        return s


def to_device(data):
    device = "cuda"
    if isinstance(data, tuple) or isinstance(data, list):
        tdata = ()
        for d in data:
            d = d.to(device)
            tdata += (d,)
        return tdata
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = v.to(device)
        return data
    else:
        return data.to(device)


def item_encode(model, config, args, output_path, save_step=50): 

    item_data = ClueWebBatchTextDataset(config, args)
    # item_batch_size = config['MAX_ITEM_LIST_LENGTH'] * config['train_batch_size']
    item_batch_size = args.batch_size

    # ckpt file 
    batch_output_path = output_path + ".temp"

    # checkpoint resumption 
    if os.path.exists(batch_output_path):
        with open(batch_output_path, 'rb') as f:
            item_feature, start_from_idx = pickle.load(f)
            print(f"loaded checkpoint from batch {start_from_idx}")
            print(f"checkpoint contains {len(item_feature)} batches")
    else: 
        item_feature = []
        start_from_idx = -1

    finished_example = (start_from_idx + 1) * item_batch_size
    # manually skip the ones  
    item_data.__skip__(finished_example)
    print(f"number of examples left: {len(item_data)}")
    item_loader = DataLoader(item_data, batch_size=item_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=customize_rmpad_collate)
    print(f"Inference item_data with {item_batch_size = } {len(item_loader) = }")
    
    print(f"Start encoding from batch {start_from_idx+1}")
    model.eval()
    with torch.no_grad():
        for idx, items in tqdm(enumerate(item_loader), total=len(item_loader)):

            actual_idx = idx + 1 + start_from_idx

            # # skip the finished batches 
            # if idx <= start_from_idx: 
            #     continue 

            items = to_device(items)
            items = model(items, mode='compute_item')
            item_feature.append(items.cpu().numpy())

            # checkpoint resumption 
            if actual_idx % save_step == 0: 
                # first save to temp temp location 
                with open(f"{batch_output_path}_1", 'wb') as f:
                    pickle.dump((item_feature, actual_idx), f)
                print(f"saved checkpoint at: {output_path}.temp at batch {actual_idx}")
                sys.stdout.flush()
                # # avoid preemption when saving ckpt...
                # os.rename(f"{batch_output_path}_1", batch_output_path)

        if isinstance(items, tuple):
            item_feature = torch.cat([x[0] for x in self.item_feature]), torch.cat([x[1] for x in self.item_feature])
        else:
            item_feature = np.concatenate(item_feature, 0)
    
    print(f"final item features has {item_feature.shape[0]} examples")
    # output embeddings 
    with open(output_path, 'wb') as f:
        pickle.dump(item_feature, f)



def seq_encode(model, config, args, output_path, save_step=50): 

    seq_loader = None # TODO 
    seq_emb = []
    with torch.no_grad():
        for idx, seqs in tqdm(enumerate(seq_loader), total=len(seq_loader)):
            attention_mask = (item_seq > 0).int()
            pos_embedding = item_feature[seqs] # historically interacted 
            user_embedding = self.user_llm(inputs_embeds=pos_embedding, attention_mask=attention_mask).hidden_states[-1]
            seq_output = user_embedding[:, -1]
            seq_emb.append(seq_output)
    seq_emb = torch.cat(seq_emb).numpy()
    # output embeddings 
    with open(output_path, 'wb') as f:
        pickle.dump(seq_emb, f)



def run(config_file=None, extra_args=[], custom_args=None):

    # configurations initialization
    config = Config(config_file_list=config_file)

    device = torch.device("cuda")
    config['device'] = device
    if len(extra_args):
        for i in range(0, len(extra_args), 2):
            key = extra_args[i][2:]
            if key in ["item_encoding", "seq_encoding"]: 
                continue  
            value = extra_args[i + 1]
            try:
                if '[' in value or '{' in value:
                    # added for text_keys? 
                    if "\\" in value: 
                        value = value.replace("\\", "")
                    value = json.loads(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            value[k] = convert_str(v)
                    else:
                        value = [convert_str(x) for x in value]
                else:
                    value = convert_str(value)
                if '.' in key:
                    k1, k2 = key.split('.')
                    config[k1][k2] = value
                else:
                    config[key] = value
            except:
                raise ValueError(f"{key} {value} invalid")

    init_seed(config['seed'], config['reproducibility'])
    
    # get model 
    print("getting model with no dataload...")
    model = get_model(config['model'])(config, None).to(device)
    trainer = Trainer(config, model) # use for easy mixed precision setup


    # load ckpt 
    ckpt_path = os.path.join(config['checkpoint_dir'], 'pytorch_model.bin')
    if os.path.exists(ckpt_path):
        print(f"Found checkpoint at {ckpt_path}, attempting to load...")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # modify the word embedding -> get tops 
        msg = trainer.model.load_state_dict(ckpt, strict=False)
        print(f'Checkpoint loaded from {ckpt_path}')
        print(f'{msg.unexpected_keys = }')
        print(f'{msg.missing_keys = }')
    else:
        print("No checkpoint found. Starting training from scratch.")


    if config['strategy'] == 'deepspeed':
        print(f"Use deepspeed strategy")
        precision = config['precision'] if config['precision'] else '32'
        strategy = DeepSpeedStrategy(stage=config['stage'], precision=precision)
        lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=1)
        lite.launch()
        model, optimizer = lite.setup(trainer.model, trainer.optimizer)
    else:
        print(f"Use DDP strategy")
        precision = config['precision'] if config['precision'] else '32'
        strategy = DDPStrategy(find_unused_parameters=True)
        lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=1)
        lite.launch()
        model = lite.setup(traier.model)


    del trainer
    if optimizer: 
        del optimizer


    # encoding 
    if custom_args.item_encoding: 
        item_data = None
        item_encode(model, config, custom_args, custom_args.output_path, save_step=custom_args.save_step) 
    elif custom_args.seq_encoding: 
        seq_data = None
        test_seq_dataloader = DataLoader(
            seq_data,
            sampler=test_item_sampler,
            batch_size=custom_args.batch_size,
            drop_last=False,
            num_workers=0,
            collate_fn=seq_data.collect_fn
        )
        seq_encode(model, test_seq_dataloader, custom_args.output_path, save_step=custom_args.save_step)     




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs='+', type=str) # HLLM.yaml
    args, extra_args = parser.parse_known_args()

    custom_parser = argparse.ArgumentParser()
    custom_parser.add_argument("--id_map_path", type=str, help="cwid_id mapping")
    custom_parser.add_argument("--item_encoding", action="store_true", default=None) 
    custom_parser.add_argument("--seq_encoding", action="store_true", default=None) 
    custom_parser.add_argument("--best_model_path", type=str, help="Path of the best checkpoint")
    custom_parser.add_argument("--output_path", type=str, help="Path to store output embeddings")
    custom_parser.add_argument("--batch_size", type=int, default=32, help="Set batch size")
    custom_parser.add_argument("--dataset_number_of_shards", type=int, default=1, help="Number of encoding shards")
    custom_parser.add_argument("--dataset_shard_index", type=int, default=0, help="Index of current shard")
    custom_parser.add_argument("--save_step", type=int, default=50, help="Number of batches to perform checkpointing")
    custom_parser.add_argument("--item_size", type=int, default=32, help="Set item token length")
    custom_parser.add_argument("--num_workers", type=int, default=1, help="Number of CPUs available for data processing")
    custom_args, _ = custom_parser.parse_known_args()

    config_file = args.config_file


    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank) # single GPU per shard 
    dist.init_process_group(backend='nccl')

    run(config_file=config_file, extra_args=extra_args, custom_args=custom_args)
