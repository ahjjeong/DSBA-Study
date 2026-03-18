# Time-series Anomaly Detection with Anomaly Transformer

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

| Metric    | 논문 (PSM) | 구현 결과 |
| --------- | ---------- | --------- |
| Accuracy  | 98.25%     | 98.65%    |
| Precision | 98.10%     | 97.13%    |
| Recall    | 97.87%     | 98.04%    |
| F1-score  | 97.83%     | 97.58%    |

> 구현 결과는 논문 reported 성능과 매우 근접한 수치를 보였다.
