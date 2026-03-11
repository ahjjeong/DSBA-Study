import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.timefeatures import time_features_from_date


def load_dataset(
    datadir: str,
    dataname: str,
    val_split_rate: float = 0.1,
    del_feature: list = None
):
    """
    목적: 다양한 시계열 이상탐지 데이터셋을 로드하여 train/val/test 분할 및 label 반환
    조건
    - PSM: CSV 형식 (train.csv, test.csv, test_label.csv)
    - MSL, SMAP, SMD: NPY 형식 ({dataname}_train.npy, {dataname}_test.npy, {dataname}_test_label.npy)
    - NaN 값 처리
    - val_split_rate를 이용하여 train 데이터에서 validation 분리
    - Anomaly Transformer는 timestamp를 사용하지 않으므로 ts는 None 반환
    """
    data_path = os.path.join(datadir, dataname)

    if dataname == 'PSM':
        # PSM dataset: CSV format
        train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
        train_data = train_data.values[:, 1:]  # drop first column (timestamp)
        train_data = np.nan_to_num(train_data).astype(np.float32)

        test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data).astype(np.float32)

        label = pd.read_csv(os.path.join(data_path, 'test_label.csv')).values[:, 1:]
        label = np.nan_to_num(label).astype(np.float32)

    else:
        raise ValueError(f"Unsupported dataset: {dataname}. Supported: PSM")

    # delete features if specified
    if del_feature is not None:
        train_data = np.delete(train_data, del_feature, axis=1)
        test_data = np.delete(test_data, del_feature, axis=1)

    # number of features
    var = train_data.shape[1]

    # train/val split
    val_size = int(len(train_data) * val_split_rate)
    trn = train_data[:len(train_data) - val_size]
    val = train_data[len(train_data) - val_size:]

    # label reshape
    if len(label.shape) > 1:
        label = label.reshape(-1)

    test_df = test_data

    # timestamp is not used for Anomaly Transformer
    trn_ts = None
    val_ts = None
    test_ts = None

    print(f"Dataset: {dataname} | train: {trn.shape}, val: {val.shape}, test: {test_df.shape}, label: {label.shape}, var: {var}")

    return trn, trn_ts, val, val_ts, test_df, test_ts, var, label
