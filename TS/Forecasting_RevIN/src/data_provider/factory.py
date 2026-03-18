import torch
from torch.utils.data import DataLoader

from data_provider.load_dataset import load_dataset
from data_provider.build_dataset import BuildDataset
from utils.scaling import apply_scaling
import warnings


def create_dataloader(
                datadir: str,
                dataname: str,
                modelname: str,
                model_config,
                scaler: str,
                batch_size: int,
                shuffle: bool,
                num_workers: int,
                pin_memory: bool,
                drop_last: bool,
                seq_len: int,
                label_len: int,
                pred_len: int,
                del_feature: list = None
                    ):

    """
    목적: 모든 Argument를 사용하여 아래 load_dataset function, BuildDataset class, apply_scaling function, DataLoader class를 이용하여 dataloader를 생성하고 반환
    조건
    - Data provider 폴더 내의 load_dataset.py, build_dataset.py, utils폴더 내의 scaling.py를 수정하여 구현
    - load_dataset에서 trn은 train data, val은 validation data, tst는 test data, var은 feature 수,
    ts는 time stamp의미 
    - 결론적으로 next(iter(trn_dataloader)).shape: (batch_size, seq_len, var)가 되어야함.
    - 최대한 범용적으로 사용할 수 있게끔 코드 작성
    """

    # time embedding 설정
    embed = getattr(model_config, 'embed', 'timeF')
    freq  = getattr(model_config, 'freq', 'h')
    time_embedding = [embed, freq]

    # 데이터셋 로드
    trn, trn_ts, val, val_ts, tst, tst_ts, var = load_dataset(
        datadir=datadir,
        dataname=dataname,
        time_embedding=time_embedding,
        del_feature=del_feature,
        seq_len=seq_len
    )

    # scaling (minmax, minmax square, minmax m1p1, standard)
    trn, val, tst, scaler = apply_scaling(trn=trn, val=val, tst=tst, scaler=scaler)

    # 데이터셋 구성
    trn_dataset = BuildDataset(
        values=trn,
        ts_feat=trn_ts,
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
    )
    val_dataset = BuildDataset(
        values=val,
        ts_feat=val_ts,
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
    )
    tst_dataset = BuildDataset(
        values=tst,
        ts_feat=tst_ts,
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
    )

    # Dataloader
    trn_dataloader = DataLoader(
        trn_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    tst_dataloader = DataLoader(
        tst_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return trn_dataset, val_dataset, tst_dataset, trn_dataloader, val_dataloader, tst_dataloader, var