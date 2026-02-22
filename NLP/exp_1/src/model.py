from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
import omegaconf

class EncoderForClassification(nn.Module):
    def __init__(self, model_config : omegaconf.DictConfig, num_labels : int):
        super().__init__()

        self.model_name = model_config.name
        self.dropout_rate = float(model_config.dropout_rate)
        self.num_labels = num_labels  # IMDB: pos/neg (2)

        # Backbone encoder
        self.encoder = AutoModel.from_pretrained(self.model_name)

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor, labels : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs : 
            input_ids : (batch_size, max_seq_len)
            attention_mask : (batch_size, max_seq_len)
            token_type_ids : (batch_size, max_seq_len) # only for BERT -> imdb 데이터셋에서는 필요 없어서 제거함
            label : (batch_size)
        Outputs :
            logits : (batch_size, num_labels)
            loss : (1)
        """
        # Encoder 인자
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        outputs = self.encoder(**encoder_kwargs)

        # last_hidden_state 반환
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        # CLS pooling
        cls = last_hidden[:, 0, :]  # (B, H)
        cls = self.dropout(cls)
        logits = self.classifier(cls)  # (B, num_labels)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return logits, loss