from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def apply_scaling(trn, val, tst, scaler_type='standard'):
    """
    목적: 학습 데이터를 기준으로 scaler를 fit한 후 train/val/test에 적용
    조건
    - scaler_type: 'standard' (StandardScaler) 또는 'minmax' (MinMaxScaler)
    - train 데이터로 fit, train/val/test 모두 transform
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler: {scaler_type}. Supported: standard, minmax")

    scaler.fit(trn)
    trn = scaler.transform(trn).astype(np.float32)
    val = scaler.transform(val).astype(np.float32)
    tst = scaler.transform(tst).astype(np.float32)

    return trn, val, tst