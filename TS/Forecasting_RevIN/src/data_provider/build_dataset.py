from torch.utils.data import Dataset
import numpy as np
import torch

class BuildDataset(Dataset):
    def __init__(self, values, ts_feat, seq_len, label_len, pred_len, scaler=None):
        super().__init__()
        self.values = np.asarray(values, dtype=np.float32)
        self.ts_feat = None if ts_feat is None else np.asarray(ts_feat, dtype=np.float32)

        self.seq_len = int(seq_len)
        self.label_len = int(label_len)  # iTransformer에서는 0
        self.pred_len = int(pred_len)

        self.scaler = scaler

        T = self.values.shape[0]
        self.var = self.values.shape[1] # 변수 개수 (N)

        # valid window
        self.valid_window = T - self.seq_len - self.pred_len + 1
        if self.valid_window <= 0:
            raise ValueError(
                f"Not enough length. T={T}, seq_len={self.seq_len}, pred_len={self.pred_len}"
            )

        # time feature 차원
        self.time_dim = 0 if self.ts_feat is None else self.ts_feat.shape[1]

    def __getitem__(self, idx):
        '''
        슬라이딩 윈도우를 이용해서 전체 시계열을 학습 가능한 작은 샘플 단위로 변환
        '''
        s = idx
        e = s + self.seq_len
        p = e + self.pred_len

        x = self.values[s:e]  # (seq_len, N)
        y = self.values[e:p]  # (pred_len, N)

        if self.ts_feat is None:
            x_mark = np.zeros((self.seq_len, 0), dtype=np.float32)
            y_mark = np.zeros((self.pred_len, 0), dtype=np.float32)
        else:
            x_mark = self.ts_feat[s:e]
            y_mark = self.ts_feat[e:p]

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_mark),
        )
    
    def __len__(self):
        return self.valid_window
    
    def inverse_transform(self, data):
        if self.scaler is None:
            return data

        return self.scaler.inverse_transform(data)