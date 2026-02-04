from typing import List, Tuple, Literal, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

import omegaconf

class IMDBDatset(torch.utils.data.Dataset):
    def __init__(self, data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test'], name: str, seed: int):
        """
        Inputs :
            data_config : omegaconf.DictConfig{
                name : str
                batch_size : int
                val_size : float
                test_size : float
                max_len : int
                seed : int
                tokenizer : str
            }
            split : Literal['train', 'valid', 'test']
        Outputs : None
        """
        super().__init__()

        self.split = split
        self.data_config = data_config


        self.seed: int = int(getattr(data_config, "seed", 42))
        self.max_length: int = int(data_config.max_length)

        # 1) Load dataset
        raw = load_dataset(data_config.name)  # {'train':..., 'test':...}

        # 2) Merge train+test => 50k
        full = concatenate_datasets([raw["train"], raw["test"]])

        # 3) Split into train/valid/test (deterministic)
        test_size = float(data_config.test_size)
        val_size = float(data_config.val_size)

        if test_size <= 0 or val_size <= 0 or (test_size + val_size) >= 1.0:
            raise ValueError(f"Invalid split sizes: val_size={val_size}, test_size={test_size}")

        # First split: full -> (rest, test)
        tmp = full.train_test_split(test_size=test_size, seed=self.seed, shuffle=True)
        rest = tmp["train"]
        test_ds = tmp["test"]

        # Second split: rest -> (train, valid)
        # valid ratio relative to 'rest'
        valid_ratio_in_rest = val_size / (1.0 - test_size)
        tmp2 = rest.train_test_split(test_size=valid_ratio_in_rest, seed=self.seed, shuffle=True)
        train_ds = tmp2["train"]
        valid_ds = tmp2["test"]

        if split == "train":
            self.data = train_ds
        elif split == "valid":
            self.data = valid_ds
        elif split == "test":
            self.data = test_ds
        else:
            raise ValueError(f"Unknown split: {split}")

        # 4) Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)

        # 5) Tokenize (padding fixed to max_length for simple collate)
        def tokenize_fn(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        self.data = self.data.map(tokenize_fn, batched=True, desc=f"Tokenizing IMDB ({split})")

        # 6) Make torch tensors
        cols = ["input_ids", "attention_mask", "label"]
        if "token_type_ids" in self.data.column_names:
            cols.insert(2, "token_type_ids")  # input_ids, attention_mask, token_type_ids, label
        self.data.set_format(type="torch", columns=cols)

        print(f">> SPLIT : {self.split} | Total Data Length : {len(self.data)}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[dict, int]:
        """
        Inputs :
            idx : int
        Outputs :
            inputs : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor
                attention_mask : torch.Tensor
            }
            label : int
        """
        item = self.data[idx]

        inputs = {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
        }
        # token_type_ids는 모델/토크나이저에 따라 없을 수 있음
        if "token_type_ids" in item:
            inputs["token_type_ids"] = item["token_type_ids"]

        label = int(item["label"])
        return inputs, label
        

    @staticmethod
    def collate_fn(batch : List[Tuple[dict, int]]) -> dict:
        # 
        """
        Inputs :
            batch : List[Tuple[dict, int]]
        Outputs :
            data_dict : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor
                attention_mask : torch.Tensor
                label : torch.Tensor
            }
        """
        inputs_list, labels_list = zip(*batch)

        # Keys that exist in this batch (token_type_ids optional)
        keys = inputs_list[0].keys()

        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            out[k] = torch.stack([x[k] for x in inputs_list], dim=0)

        out["labels"] = torch.tensor(labels_list, dtype=torch.long)
        return out
    
def get_dataloader(data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test'], name: str, seed: int) -> torch.utils.data.DataLoader:
    """
    Output : torch.utils.data.DataLoader
    """
    dataset = IMDBDatset(data_config, split, name=name, seed=seed)
    dataloader = DataLoader(
        dataset,
        batch_size=int(data_config.batch_size),
        shuffle=(split == "train"),
        collate_fn=IMDBDatset.collate_fn,
        num_workers=int(getattr(data_config, "num_workers", 2)),
        pin_memory=bool(getattr(data_config, "pin_memory", True)),
    )
    return dataloader
