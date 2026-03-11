# DSBA Time-series pretraining: Anomaly Detection

본 프로젝트는 Anomaly Transformer 모델을 활용하여 시계열 이상탐지(Time Series Anomaly Detection) 실험을 수행하고, PSM 벤치마크 데이터셋에서의 성능을 재현하는 것을 목표로 한다.

논문에서 제시한 Implementation Details을 기반으로 동일한 데이터 분할 및 설정을 적용하여 실험을 진행하였다.

---

## Dataset

PSM (Pooled Server Metrics) 데이터셋을 사용하였다.

| 데이터셋 | 형식 | 변수 수 (Dim) | Anomaly Ratio |
| ------- | ---- | ------------ | ------------- |
| PSM     | CSV  | 25           | 1%            |

### 데이터셋 구성

- `train.csv`: 학습 데이터 (정상 패턴)
- `test.csv`: 테스트 데이터 (정상 + 이상)
- `test_label.csv`: 테스트 데이터의 이상 레이블 (0: 정상, 1: 이상)

### 데이터셋 분할 방식

- Train: 전체 학습 데이터의 90%
- Validation: 전체 학습 데이터의 10% (마지막 구간)
- Test: 테스트 데이터 전체 (비중첩 윈도우, stride=win_size)
- StandardScaler: 학습 데이터로 fit, 학습/검증/테스트 모두 transform

---

## Model

### Anomaly Transformer

시계열 이상탐지를 위한 Transformer 기반 모델로, **Association Discrepancy**를 핵심 이상 탐지 기준으로 사용한다.

**[ 핵심 아이디어 ]**

1. **Anomaly Attention**: 학습 가능한 Gaussian prior와 attention series 간의 분포 차이(Association Discrepancy)를 계산
2. **Minimax Strategy**: 두 가지 loss를 사용하여 정상/이상 패턴 간의 구분을 극대화
   - `loss1 = rec_loss - k * series_loss` (series가 prior에 가까워지도록)
   - `loss2 = rec_loss + k * prior_loss` (prior가 series에서 멀어지도록)
3. **Anomaly Score**: `softmax(-(series_loss + prior_loss)) * reconstruction_loss`

**[ 코드 ]**

```python
class AnomalyAttention(nn.Module):
    def forward(self, queries, keys, values, sigma, attn_mask):
        # Attention series: softmax(Q @ K^T / sqrt(d))
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        series = self.dropout(torch.softmax(scale * scores, dim=-1))

        # Gaussian prior: learnable sigma로 parameterize
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        prior = 1.0 / (sqrt(2π) * σ) * exp(-distance² / 2σ²)

        return V, series, prior, sigma
```

---

## Experiments

### Setup

- Window size (seq_len): 100
- Batch size: 256
- Optimizer: Adam
- Learning rate: 0.0001
- Loss: MSE (reconstruction)
- Epochs: 3
- Early stopping patience: 3
- Discrepancy weight (k): 3
- Temperature: 50
- Anomaly ratio: 1%

그 외 세부적인 하이퍼파라미터 및 실험 설정은 Anomaly Transformer 논문 저자들이 공개한 공식 GitHub의 solver.py 파일을 참고하여 동일하게 적용하였다.

### 모델 구조

| 항목 | 설정값 |
| ---- | ------ |
| d_model | 512 |
| n_heads | 8 |
| e_layers | 3 |
| d_ff | 512 |
| dropout | 0.0 |
| activation | GELU |

### 평가 방식

논문과 동일한 평가 파이프라인을 적용하였다.

1. **Anomaly Score 계산**: 학습 데이터 + 테스트 데이터에 대해 anomaly score 산출
2. **Threshold 결정**: 학습/테스트 energy를 결합한 후 `percentile(100 - anomaly_ratio)` 적용
3. **Detection Adjustment**: Point-adjust 방식으로 이상 구간 내 예측값 보정
4. **평가 지표**: Accuracy, Precision, Recall, F-score

### Results

| Metric    | 논문 (PSM) |
| --------- | --------- |
| Accuracy  | 98.25%    |
| Precision | 98.10%    |
| Recall    | 97.87%    |
| F-score   | 97.83%    |

> 실험 결과는 추후 업데이트 예정

### Analysis

**1. Association Discrepancy의 효과**

- 정상 데이터: attention series가 Gaussian prior와 유사한 분포 → 낮은 discrepancy
- 이상 데이터: attention series가 prior에서 벗어남 → 높은 discrepancy
- Minimax strategy가 이 차이를 극대화하여 이상 탐지 성능 향상

**2. Anomaly Score 구성**

- Reconstruction loss만 사용할 경우 이상 탐지 성능이 제한적
- Association discrepancy (metric)와 reconstruction loss를 결합하여 보다 robust한 anomaly score 산출

**3. Detection Adjustment**

- Point-level 예측을 segment-level로 보정하여 실용적인 이상 탐지 성능 확보
- 이상 구간 내 하나라도 탐지되면 해당 구간 전체를 이상으로 판정
