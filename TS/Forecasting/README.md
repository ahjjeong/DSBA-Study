# DSBA Time-series pretraining: Forecasting

본 프로젝트는 iTransformer 모델을 활용하여 장기 시계열 예측(Long-term Time Series Forecasting) 실험을 수행하고, ETT 벤치마크 데이터셋에서의 성능을 재현하는 것을 목표로 한다.

논문에서 제시한 Implementation Details을 기반으로 동일한 데이터 분할 및 설정을 적용하여 실험을 진행하였다.

---

## Dataset

ETT (Electricity Transformer Temperature) 벤치마크 데이터셋을 사용하였다.

| 데이터셋  | 주기 (Frequency) | 변수 수 (Dim) | Train / Valid / Test  |
| ----- | -------------- | ---------- | ---------------- |
| ETTh1 | Hourly         | 7          | 12개월 / 4개월 / 4개월 |
| ETTh2 | Hourly         | 7          | 12개월 / 4개월 / 4개월 |
| ETTm1 | 15분            | 7          | 12개월 / 4개월 / 4개월 |
| ETTm2 | 15분            | 7          | 12개월 / 4개월 / 4개월 |

### 데이터셋 분할 방식

- Train: 첫 12개월
- Validation: 그 다음 4개월
- Test: 마지막 4개월
- 시계열 순서를 유지한 Sequential split (랜덤 분할은 사용하지 않음)

--- 

## Model

### iTransformer

기존의 시점별 토큰 대신 변수별 토큰 구조를 사용하는 Inverted Transformer 구조의 Encoder-only 시계열 예측 모델이다.

<img src="https://github.com/user-attachments/assets/24d7a176-a638-40b5-94ab-58273d75a462" width="80%" />

<br><br>

**[ 코드 ]**

```
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # invert 역할을 수행하는 핵심 코드
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)
```

---

## Experiments

### Setup

- Lookback length (seq_len): 96
- Prediction length (pred_len) ∈ {96, 192, 336, 720}
- Optimizer: Adam
- Learning rate: 0.0001
- Batch size: 32
- Loss: MSE

그 외 세부적인 하이퍼파라미터 및 실험 설정은 iTransformer 논문 저자들이 공개한 공식 GitHub의 run.py 파일을 참고하여 동일하게 적용하였다.

### Results

**[ ETTh1 ]**

| pred_len | MSE (%)    | MAE (%)    |
| -------- | ---------- | ---------- |
| 96       | **38.75%** | **40.47%** |
| 192      | 44.28%     | 43.68%     |
| 336      | 49.00%     | 45.99%     |
| 720      | 51.34%     | 49.44%     |

**[ ETTh2 ]**

| pred_len | MSE (%)    | MAE (%)    |
| -------- | ---------- | ---------- |
| 96       | **30.14%** | **35.12%** |
| 192      | 37.91%     | 39.89%     |
| 336      | 42.01%     | 43.15%     |
| 720      | 42.86%     | 44.71%     |

**[ ETTm1 ]**

| pred_len | MSE (%)    | MAE (%)    |
| -------- | ---------- | ---------- |
| 96       | **34.41%** | **37.78%** |
| 192      | 38.35%     | 39.58%     |
| 336      | 42.04%     | 41.82%     |
| 720      | 48.95%     | 45.66%     |

**[ ETTm2 ]**

| pred_len | MSE (%)    | MAE (%)    |
| -------- | ---------- | ---------- |
| 96       | **18.53%** | **27.12%** |
| 192      | 25.11%     | 31.29%     |
| 336      | 31.50%     | 35.29%     |
| 720      | 41.12%     | 40.52%     |

**[ ETT 전체 평균 ]**

| Metric  | 구현 결과 | 논문 |
| ------- | ---------- | ---------- |
| MSE | 38.51% | 38.30% |
| MAE | 40.09% | 40.70% |

> 구현 결과는 논문 reported 성능과 매우 근접한 수치를 보였다.

### Analysis

**1. 예측 길이가 길어질수록 성능 저하**

- 모든 데이터셋에서 pred_len 증가 → MSE/MAE 증가
- 특히 720-step forecasting에서 성능 저하가 뚜렷함
- 장기 예측일수록 오차 누적 효과 발생

**2. 데이터셋 난이도 비교**

- 성능 기준 (MSE 평균 기준): ETTm2 < ETTh2 < ETTm1 < ETTh1

**3. 장기 예측 민감도 차이**

- 720-step 기준 MSE 증가폭:
  - ETTh1: 38.75% → 51.34% (+12.59%p)
  - ETTh2: 30.14% → 42.86% (+12.72%p)
  - ETTm1: 34.41% → 48.95% (+14.54%p)
  - ETTm2: 18.53% → 41.12% (+22.59%p)
- ETTm2가 장기 예측에서 가장 큰 성능 저하(+22.59%p) 를 보임
  - 단기 예측(96-step)에서는 가장 낮은 오차(18.53%)로 가장 우수한 성능
  - 하지만 720-step 장기 예측에서는 오차 증가폭이 가장 큼
