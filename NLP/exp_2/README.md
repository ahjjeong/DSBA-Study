# IMDB Sentiment Classification

### Batch Size Scaling with Gradient Accumulation (Torch vs Accelerate)

본 repository는 IMDB 문장 감정 분류 문제에서 effective batch size scaling이 학습 안정성과 일반화 성능에 미치는 영향을 분석하기 위한 실험을 다룬다.

특히, 동일한 모델 구조와 optimizer 설정 하에서

- 순수 PyTorch 기반 Gradient Accumulation
- HuggingFace Accelerate 기반 Gradient Accumulation

을 비교하고,

Global Batch Size 64 / 256 / 1024 중 최적의 batch size를 탐색하는 것을 목표로 한다.

---

## Directory Structure

```
NLP/
└── exp_2/
    ├── README.md                          # 프로젝트 개요 및 실험 결과 정리
    │
    ├── configs/                           # Hydra 기반 실험 설정
    │   ├── default.yaml                   # 공통 기본 설정
    │   └── model/                         # 모델 설정
    │       ├── bert.yaml
    │       └── modernbert.yaml
    │
    ├── src/                               # 핵심 로직
    │   ├── data.py                        # IMDB 데이터 로딩 및 토크나이징
    │   ├── model.py                       # EncoderForClassification 정의
    │   └── utils.py                       # seed, device, wandb 등 유틸
    │
    ├── main_torch.py                      # Torch 기반 Gradient Accumulation 학습
    ├── main_accelerate.py                 # HuggingFace Accelerate 기반 학습
    │
    └── scripts/
        ├── run_torch.sh                   # Torch 기반 batch size 실험
        ├── run_accelerate.sh              # HuggingFace Accelerate 기반 batch size 실험
        └── run_scaling_lr.sh              # Linear LR Scaling 실험
```

---

## How to Run

### 1. Weights & Biases 설정: Edit ```configs/default.yaml```

```
wandb login [API_KEY]
```


### 2. 실험 실행

- Torch 기반 batch size 실험

```
bash scripts/run_torch.sh
```

- HuggingFace Accelerate 기반 batch size 실험

```
bash scripts/run_accelerate.sh
```

---

## Dataset

### IMDB Sentiment Classification

* 전체 데이터: 50k 문장

  * Train(25k) + Test(25k) 통합 후 8:1:1 분할
* 클래스 수: 2

---

## Model

- BERT-base-uncased
- ModernBERT-base

---

## Setup

### 데이터 및 학습 설정

* Epochs: 5
* Max sequence length: 128
* Random seed: 42
* Loss: Cross Entropy Loss

### Batch Size 설정

Per-device batch size는 16으로 고정하고,

| Global Batch Size | Grad Accum Steps |
| ----------------- | ---------------- |
| 64                | 4                |
| 256               | 16               |
| 1024              | 64               |


즉,

$$ \text{Global Batch Size} = \text{per-device batch size} \times \text{grad accumulation steps} $$

### Optimizer 설정
* Optimizer: Adam
* Learning Rate: 5e-5
* Betas: (0.9, 0.999)
* Epsilon: 1e-6
* Weight Decay: 0.01

---

## Implementation

### 1. Torch 기반 Gradient Accumulation

```
loss = criterion(outputs, labels)
loss = loss / grad_accum_steps
loss.backward()

if (step + 1) % grad_accum_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

**특징**
- loss scaling 필요
- optimizer step 제어 직접 수행
- logging 시 raw loss와 scaled loss 구분 필요

### 2. HuggingFace Accelerate 기반 Gradient Accumulation

```
with accelerator.accumulate(model):
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)

    optimizer.step()
    optimizer.zero_grad()
```

**특징**
- loss scaling 자동 처리
- 분산 학습 확장 가능
- 코드 간결

## Metrics

모델 성능 평가는 다음 지표를 기준으로 수행한다.

* Accuracy

모든 실험은 **epoch 단위로 validation을 수행**하며,
validation 성능이 가장 우수한 checkpoint를 기준으로 test 성능을 측정한다.

---

## Experiments

### 실험 목표

- Batch size 64 / 256 / 1024 중 최적 batch size 탐색
- Torch vs Accelerate 구현 비교

---

### Results

## 1. Torch 기반 실험

**BERT**
<div align="center">
    <img src="https://github.com/user-attachments/assets/47553397-02c4-4e4e-96c1-5a0a1d6a5906" width="49%" />
    <img src="https://github.com/user-attachments/assets/ce13ea21-8e6c-494e-aacd-abdf9b02186d" width="41%" />
</div>

**ModernBERT**
<div align="center">
    <img src="https://github.com/user-attachments/assets/9fbd5a0c-da4f-42b7-83d8-d80ddaa2599b" width="49%" />
    <img src="https://github.com/user-attachments/assets/0f8d7c20-5ecc-408d-b3d7-1542bd36da49" width="41%" />
</div>

**Test Accuracy Summary**

| Global Batch Size | BERT   | ModernBERT |
|-------------------|:------:|:----------:|
| 64                | **82.90%** | 89.82%     |
| 256               | 74.86% | 90.54%     |
| 1024              | 49.50% | **91.02%**     |

- **BERT**
    - optimal batch size: 64
    - 배치가 커질수록 성능이 급격히 하락 (0.829 → 0.7486 → 0.4950)
    <div align="left">
        <img src="https://github.com/user-attachments/assets/3776fad2-7b5d-4a11-9fd2-a73e12c9105d" width="45%" />
        <img src="https://github.com/user-attachments/assets/8a5ceeeb-1d33-4f16-918f-c2aa64f08bd7" width="45%" />
    </div>

- **ModernBERT**
    - optimal batch size: 1024
    - 배치가 커질수록 성능이 지속적으로 향상 (0.8982 → 0.9054 → 0.9102)

## 2. HuggingFace Accelerate 기반 실험

**BERT**
<div align="center">
    <img src="https://github.com/user-attachments/assets/7dd1d762-bc6f-4a40-9d14-203c308d5e77" width="49%" />
    <img src="https://github.com/user-attachments/assets/08d76809-55b6-4a52-a959-ac5ca417323a" width="41%" />
</div>

**ModernBERT**
<div align="center">
    <img src="https://github.com/user-attachments/assets/eeebbafb-c745-40a1-ba01-f8639c8897b7" width="49%" />
    <img src="https://github.com/user-attachments/assets/c95af199-2ef3-4f0b-817c-baea63997c9a" width="41%" />
</div>

**Test Accuracy Summary**

| Global Batch Size | BERT   | ModernBERT |
|-------------------|:------:|:----------:|
| 64                | 87.46% | 90.08%     |
| 256               | 50.50% | 50.50%     |
| 1024              | 88.02% | 91.06%     |
