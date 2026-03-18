# Applying RevIN to Time-series Forecasting with iTransformer

본 프로젝트는 `TS/Forecasting`의 iTransformer 모델에 **RevIN(Reversible Instance Normalization)** 레이어를 적용하여, 시계열 데이터의 distribution shift 완화 및 학습 안정성 향상 효과를 실험적으로 분석하는 것을 목표로 한다.

---

## RevIN (Reversible Instance Normalization)

RevIN은 시계열 데이터에서 시간에 따라 변화하는 평균과 분산(즉, distribution shift) 문제를 완화하기 위한 정규화 기법이다.

기존의 global normalization 방식은 전체 데이터의 통계량을 기준으로 정규화를 수행하기 때문에, 시간에 따라 분포가 변하는 non-stationary 시계열에서는 효과적으로 작동하지 않을 수 있다.

RevIN은 이러한 문제를 해결하기 위해 다음과 같은 구조를 가진다:

- 입력 시 instance-wise normalization 수행
- 모델 출력 시 denormalization (역변환)을 통해 원래 scale 복원
- 추가적으로 learnable affine parameter (γ, β)를 통해 정규화 강도를 조절

즉, RevIN은 정보 손실 없이 정규화를 수행하는 reversible normalization 구조를 가진다.


### 적용 위치

`iTransformer.py`의 `forecast()` 메서드에서 기존 수동 정규화를 RevIN 레이어로 교체하였다.

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

**1. RevIN은 “안정적이지만 제한적인 개선”을 보인다**

- 16개 실험 중 MSE 기준 11개에서 개선, 5개에서 동일
- 평균적으로 MSE 0.13%p, MAE 0.10%p 감소
- 그러나 개선 폭이 작아 **유의미한 성능 향상으로 보기는 어려움**

**2. RevIN은 성능 향상보다 “stabilization” 역할에 가깝다**

- 대부분의 실험에서 성능이 개선되거나 동일하게 유지됨
- 성능 저하가 발생하지 않는다는 점에서 → regularization-like 안정화 효과를 가짐
- 이는 RevIN의 본래 목적(= distribution shift 완화)과 일치

**3. 결론**

- RevIN은 iTransformer에 간단한 구조 변경만으로 적용 가능
- 성능을 크게 향상시키지는 않지만, 전반적으로 안정적인 성능 유지 및 소폭 개선 효과를 보임
- 특히 distribution shift가 존재하는 시계열에서 robustness를 향상시키는 normalization 기법으로 해석할 수 있음
