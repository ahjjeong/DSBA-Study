from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def apply_scaling(trn, val, tst, scaler):
    """
    scaler:
      - "standard"
      - "minmax"
      - "minmax_square"   : minmax 후 제곱
      - "minmax_m1p1"     : minmax 후 [-1, 1]로 변환
    """
    if scaler is None:
        return trn, val, tst, None

    scaler = str(scaler).lower()

    if scaler == "standard":
        sc = StandardScaler()
        trn_2d = trn
        sc.fit(trn_2d)
        trn_s = sc.transform(trn_2d)
        val_s = sc.transform(val)
        tst_s = sc.transform(tst)
        return trn_s, val_s, tst_s, sc

    if scaler in ["minmax", "minmax_square", "minmax_m1p1"]:
        sc = MinMaxScaler(feature_range=(0.0, 1.0))
        sc.fit(trn)
        trn_s = sc.transform(trn)
        val_s = sc.transform(val)
        tst_s = sc.transform(tst)

        if scaler == "minmax_square":
            return trn_s**2, val_s**2, tst_s**2, sc
        if scaler == "minmax_m1p1":
            # [0,1] -> [-1,1]
            return trn_s * 2.0 - 1.0, val_s * 2.0 - 1.0, tst_s * 2.0 - 1.0
        return trn_s, val_s, tst_s, sc

    raise ValueError(f"Unknown scaler: {scaler}")

