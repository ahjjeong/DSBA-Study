from typing import List, Tuple, Literal, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import DatasetDict, load_dataset, concatenate_datasets
from transformers import AutoTokenizer

import omegaconf

class IMDBDatset(torch.utils.data.Dataset):
    def __init__(self, config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test'], model_name: str):
        """
        Inputs :
            config : omegaconf.DictConfig{
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
        self.config = config
        data_config = config.dataset
        self.data_config = data_config

        self.seed = int(config.seed)
        self.max_length = int(data_config.max_length)

        # 데이터셋 불러오기
        dataset = load_dataset(data_config.name)

        # train(25k) + test(25k) => 50k
        full_dataset = concatenate_datasets([dataset["train"], dataset["test"]])

        # train/valid/test split
        test_size = float(data_config.test_size)
        val_size = float(data_config.val_size)

        # full -> (train_valid, test)
        train_val_test = full_dataset.train_test_split(test_size=test_size, seed=self.seed, shuffle=True)
        train_val = train_val_test["train"]
        test_data = train_val_test["test"]

        # train_valid -> (train, valid)
        val_ratio = val_size / (1.0 - test_size)
        train_val = train_val.train_test_split(test_size=val_ratio, seed=self.seed, shuffle=True)
        train_data = train_val["train"]
        val_data = train_val["test"]

        self.data = DatasetDict({
            'train': train_data,
            'valid': val_data,
            'test': test_data
        })[split]

        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        def tokenize_fn(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        self.data = self.data.map(tokenize_fn, batched=True, desc=f"Tokenizing IMDB ({split})")

        # torch.Tensor 형태로 반환
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
    
def get_dataloader(config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test'], model_name: str) -> torch.utils.data.DataLoader:
    """
    Output : torch.utils.data.DataLoader
    """
    data_config = config.dataset

    dataset = IMDBDatset(config, split, model_name=model_name)
    dataloader = DataLoader(
        dataset,
        batch_size=int(data_config.batch_size),
        shuffle=(split == "train"),
        collate_fn=IMDBDatset.collate_fn,
        num_workers=int(getattr(data_config, "num_workers", 2)),
        pin_memory=bool(getattr(data_config, "pin_memory", True)),
    )
    return dataloader
