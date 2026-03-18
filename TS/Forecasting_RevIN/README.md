# DSBA Time-series pretraining: Forecasting + RevIN

본 프로젝트는 `TS/Forecasting`의 iTransformer 모델에 **RevIN(Reversible Instance Normalization)** 레이어를 적용하여, 기존 정규화 방식 대비 성능 변화를 비교 분석하는 실습이다.

---

## RevIN (Reversible Instance Normalization)

RevIN은 시계열 데이터의 분포 변화(distribution shift) 문제를 완화하기 위한 정규화 기법이다. 핵심 아이디어는 모델 입력 시 instance-wise normalization을 수행하고, 출력 시 이를 역변환(denormalization)하여 원래 분포로 복원하는 것이다.

### 기존 방식 vs RevIN

| 구분 | 기존 (Non-stationary Transformer) | RevIN |
| --- | --- | --- |
| 정규화 | 수동 mean/stdev 계산 후 차감·나눗셈 | Instance Normalization (mean/stdev) |
| 역정규화 | 저장된 mean/stdev로 수동 복원 | 저장된 통계량 + affine 역변환 |
| 학습 파라미터 | 없음 | affine_weight, affine_bias (learnable) |

### 적용 위치

[iTransformer.py](src/models/iTransformer.py) 의 `forecast()` 메서드에서 기존 수동 정규화를 RevIN 레이어로 교체하였다.

```python
from layers.RevIN import RevIN

# __init__
self.revin_layer = RevIN(configs.enc_in)

# forecast()
x_enc = self.revin_layer(x_enc, 'norm')    # 입력 정규화
# ... encoder, projection ...
dec_out = self.revin_layer(dec_out, 'denorm')  # 출력 역정규화
```

---

## Dataset

ETT (Electricity Transformer Temperature) 벤치마크 데이터셋을 사용하였다.

| 데이터셋  | 주기 (Frequency) | 변수 수 (Dim) | Train / Valid / Test  |
| ----- | -------------- | ---------- | ---------------- |
| ETTh1 | Hourly         | 7          | 12개월 / 4개월 / 4개월 |
| ETTh2 | Hourly         | 7          | 12개월 / 4개월 / 4개월 |
| ETTm1 | 15분            | 7          | 12개월 / 4개월 / 4개월 |
| ETTm2 | 15분            | 7          | 12개월 / 4개월 / 4개월 |

---

## Experiments

### Setup

- Lookback length (seq_len): 96
- Prediction length (pred_len) ∈ {96, 192, 336, 720}
- Optimizer: Adam (lr=0.0001)
- Batch size: 32
- Loss: MSE
- use_norm: true (RevIN 활성화)

### Results

**[ ETTh1 ]**

| pred_len | Original MSE | RevIN MSE | Original MAE | RevIN MAE |
| -------- | ------------ | --------- | ------------ | --------- |
| 96       | 38.75%       | 38.75%    | 40.47%       | 40.47%    |
| 192      | 44.28%       | **44.26%** | 43.68%      | **43.67%** |
| 336      | 49.00%       | **48.64%** | 45.99%      | **45.96%** |
| 720      | 51.34%       | **50.51%** | 49.44%      | **49.36%** |

**[ ETTh2 ]**

| pred_len | Original MSE | RevIN MSE | Original MAE | RevIN MAE |
| -------- | ------------ | --------- | ------------ | --------- |
| 96       | 30.14%       | **30.02%** | 35.12%      | **35.01%** |
| 192      | 37.91%       | **37.82%** | 39.89%      | **39.86%** |
| 336      | 42.01%       | **42.00%** | 43.15%      | 43.15%    |
| 720      | 42.86%       | 42.86%    | 44.71%      | 44.71%    |

**[ ETTm1 ]**

| pred_len | Original MSE | RevIN MSE | Original MAE | RevIN MAE |
| -------- | ------------ | --------- | ------------ | --------- |
| 96       | 34.41%       | **34.12%** | 37.78%      | **37.56%** |
| 192      | 38.35%       | **38.34%** | 39.58%      | 39.58%    |
| 336      | 42.04%       | 42.04%    | 41.82%      | 41.82%    |
| 720      | 48.95%       | 48.95%    | 45.66%      | **45.59%** |

**[ ETTm2 ]**

| pred_len | Original MSE | RevIN MSE | Original MAE | RevIN MAE |
| -------- | ------------ | --------- | ------------ | --------- |
| 96       | 18.53%       | **18.48%** | 27.12%      | **27.07%** |
| 192      | 25.11%       | **25.07%** | 31.29%      | **31.25%** |
| 336      | 31.50%       | 31.50%    | 35.29%      | **35.20%** |
| 720      | 41.12%       | **41.08%** | 40.52%      | **40.50%** |

**[ ETT 전체 평균 ]**

| Metric | Original | RevIN | 변화량 |
| ------ | -------- | ----- | ----- |
| MSE    | 38.51%   | **38.38%** | -0.13%p |
| MAE    | 40.09%   | **39.99%** | -0.10%p |

---

### Analysis

**1. RevIN은 전반적으로 성능을 개선하며, 성능 저하는 없다**

- 16개 실험 조합 중 MSE 기준 11개에서 개선, 5개에서 동일, 악화된 케이스는 없음
- 전체 평균 MSE 0.13%p, MAE 0.10%p 개선

**2. 장기 예측에서 더 큰 효과를 보인다 (ETTh1)**

- ETTh1에서 pred_len이 길어질수록 RevIN의 개선 폭이 증가:
  - 96-step: 동일 (0.00%p)
  - 192-step: -0.02%p
  - 336-step: **-0.36%p**
  - 720-step: **-0.83%p**
- 장기 예측일수록 분포 변화(distribution shift)가 커지므로, RevIN의 instance normalization이 더 효과적으로 작용

**3. 단기 예측에서는 ETTm1에서 가장 큰 개선**

- ETTm1 96-step에서 MSE 34.41% → 34.12% (-0.29%p)로 가장 큰 단기 예측 개선
- 15분 단위 고주파 데이터에서 instance-level 정규화가 분포 안정화에 기여

**4. RevIN의 learnable affine parameter의 역할**

- 기존 Non-stationary Transformer의 정규화는 단순 통계적 변환에 불과
- RevIN은 학습 가능한 `affine_weight`, `affine_bias`를 통해 정규화 강도를 데이터에 맞게 조절
- 이로 인해 성능이 악화되는 케이스 없이 안정적으로 개선 효과를 제공

**5. 결론**

- RevIN은 iTransformer에 **최소한의 코드 변경**으로 적용 가능하며, **성능 저하 없이 일관된 개선**을 제공한다
- 특히 장기 예측(720-step)과 분포 변동이 큰 데이터셋에서 효과가 두드러진다
- 추가 학습 파라미터(affine)가 매우 적어 계산 비용 증가 없이 적용 가능하다
