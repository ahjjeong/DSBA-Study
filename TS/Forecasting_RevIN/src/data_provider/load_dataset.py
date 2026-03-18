from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import timedelta
from utils.timefeatures import time_features
import dateutil
import pdb
from omegaconf import OmegaConf

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features_from_date

def _ett_borders(dataname: str, seq_len: int, flag: str):
    """
      ETTh*: 1 hour
      ETTm*: 15 min(=4 per hour)
    """
    assert flag in ["train", "val", "test"]
    if dataname.lower().startswith("ettm"):
        unit = 24 * 4   # 하루 = 96 (15분 단위)
    else:
        unit = 24       # 하루 = 24 (1시간 단위)

    # 12개월 / 4개월 / 4개월 (30일 가정)
    train_len = 12 * 30 * unit
    val_len = 4 * 30 * unit
    test_len  = 4 * 30 * unit

    # border1s/border2s (val/test 시작을 -seq_len만큼 당김)
    border1s = [
        0,
        train_len - seq_len,
        train_len + val_len - seq_len,
    ]
    border2s = [
        train_len,
        train_len + val_len,
        train_len + val_len + test_len,
    ]

    type_map = {"train": 0, "val": 1, "test": 2}
    i = type_map[flag]
    return border1s[i], border2s[i]


def load_dataset(
    datadir: str,
    dataname: str,
    time_embedding: list = ['timeF', 'h'],
    del_feature: list = None,
    seq_len: int = 96
):
    # 데이터 경로
    file_path = Path(datadir) / f"{dataname}.csv"

    # 데이터 불러오기
    df = pd.read_csv(file_path)

    # date 컬럼 처리
    df['date'] = pd.to_datetime(df['date'])

    # del_feature가 있을 경우 컬럼 drop
    if del_feature is not None:
        df = df.drop(columns=del_feature, errors="ignore")

    feature_cols = [c for c in df.columns if c != 'date']
    values = df[feature_cols].to_numpy(dtype=np.float32)  # (T, N)
    T = len(values)
    var = values.shape[1] # N = 변수 개수 = 7

    # border 호출
    trn_b1, trn_b2 = _ett_borders(dataname, seq_len, "train")
    val_b1, val_b2 = _ett_borders(dataname, seq_len, "val")
    tst_b1, tst_b2 = _ett_borders(dataname, seq_len, "test")

    # 길이 초과 방지
    trn_b2 = min(trn_b2, T)
    val_b2 = min(val_b2, T)
    tst_b2 = min(tst_b2, T)

    trn = values[trn_b1:trn_b2]
    val = values[val_b1:val_b2]
    tst = values[tst_b1:tst_b2]

    # timestamp feature 생성
    embed = time_embedding[0]
    freq  = time_embedding[1]
    timeenc = 0 if embed != "timeF" else 1

    ts_feat = time_features_from_date(df["date"], timeenc=timeenc, freq=freq)
    ts_feat = np.asarray(ts_feat, dtype=np.float32)

    trn_ts = ts_feat[trn_b1:trn_b2]
    val_ts = ts_feat[val_b1:val_b2]
    tst_ts = ts_feat[tst_b1:tst_b2]

    return trn, trn_ts, val, val_ts, tst, tst_ts, var