import numpy as np
import torch
from torch.utils.data import Dataset

class BuildDataset(Dataset):
    def __init__(self, data, seq_len, stride, label=None, timestamp=None):
        """
        목적: 시계열 데이터를 sliding window 방식으로 구성하는 Dataset 클래스
        조건
        - data: (N, F) numpy array, 시계열 데이터
        - seq_len: window size
        - stride: sliding window stride
        - label: (N,) numpy array, anomaly label (test 시 사용)
        - timestamp: (N, ...) numpy array, timestamp (사용하지 않을 경우 None)
        """
        self.data = data
        self.seq_len = seq_len
        self.stride = stride
        self.label = label
        self.timestamp = timestamp

    def __len__(self):
        return (self.data.shape[0] - self.seq_len) // self.stride + 1

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.stride
        end = start + self.seq_len

        item = {
            'input': np.float32(self.data[start:end]),
            'target': np.float32(self.data[start:end]),  # autoencoder: target = input
        }

        if self.timestamp is not None:
            item['timestamp'] = np.float32(self.timestamp[start:end])

        if self.label is not None:
            item['label'] = np.float32(self.label[start:end])

        return item
