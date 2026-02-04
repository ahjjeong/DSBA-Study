from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
import omegaconf

class EncoderForClassification(nn.Module):
    def __init__(self, model_config : omegaconf.DictConfig):
        super().__init__()


        self.name: str = model_config.name
        self.dropout_rate: float = float(getattr(model_config, "dropout_rate", 0.1))
        self.num_labels: int = int(getattr(model_config, "num_labels", 2))  # IMDB: pos/neg

        # Backbone encoder (NO AutoModelForSequenceClassification)
        self.encoder = AutoModel.from_pretrained(self.name)

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor, token_type_ids : torch.Tensor, label : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs : 
            input_ids : (batch_size, max_seq_len)
            attention_mask : (batch_size, max_seq_len)
            token_type_ids : (batch_size, max_seq_len) # only for BERT
            label : (batch_size)
        Outputs :
            logits : (batch_size, num_labels)
            loss : (1)
        """
        # Build kwargs safely (ModernBERT 등은 token_type_ids를 안 받을 수 있음)
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            # 모델이 token_type_ids를 지원하지 않는 경우를 대비해 try/except
            try:
                outputs = self.encoder(**encoder_kwargs, token_type_ids=token_type_ids)
            except TypeError:
                outputs = self.encoder(**encoder_kwargs)
        else:
            outputs = self.encoder(**encoder_kwargs)

        # Most encoders return last_hidden_state
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        # CLS pooling
        cls = last_hidden[:, 0, :]  # (B, H)
        cls = self.dropout(cls)
        logits = self.classifier(cls)  # (B, num_labels)

        loss = None
        if label is not None:
            loss = self.loss_fn(logits, label)

        return logits, loss