# IMDB Sentiment Classification Batch Size Scaling

### Batch Size Scaling with Gradient Accumulation (Torch vs Accelerate)

본 repository는 IMDB 문장 감정 분류 문제에서 effective batch size scaling이 학습 안정성과 일반화 성능에 미치는 영향을 분석하기 위한 실험을 다룬다.

특히 64 / 256 / 1024 중 **최적의 batch size**를 탐색하고,

동일한 모델 구조와 optimizer 설정 하에서

- **PyTorch** 기반 Gradient Accumulation
- **HuggingFace Accelerate** 기반 Gradient Accumulation

을 비교하는 것을 목표로 한다.

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

- **Torch 기반 batch size 실험**

```
bash scripts/run_torch.sh
```

- **HuggingFace Accelerate 기반 batch size 실험**

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

Per-device batch size는 **16**으로 고정하고,

| Batch Size | Grad Accum Steps |
| ----------------- | ---------------- |
| 64                | 4                |
| 256               | 16               |
| 1024              | 64               |


즉,

$$ \text{Batch Size} = \text{per-device batch size} \times \text{grad accumulation steps} $$

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

- ```loss / grad_accum_steps```로 loss scaling을 명시적으로 해야 함

### 2. HuggingFace Accelerate 기반 Gradient Accumulation

```
with accelerator.accumulate(model):
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

- ```accelerator.accumulate(model)```가 누적 구간을 관리

## Metrics

모델 성능 평가는 다음 지표를 기준으로 수행한다.

* Accuracy

모든 실험은 **epoch 단위로 validation을 수행**하며,
validation 성능이 가장 우수한 checkpoint를 기준으로 test 성능을 측정한다.

---

## Experiments

### 1. Torch 기반 실험

**[ BERT ]**
<div align="center">
    <img src="https://github.com/user-attachments/assets/4b1892f3-965d-4631-883b-f69687c7828a" width="45%" />
    <img src="https://github.com/user-attachments/assets/2ac199ba-e364-40f2-871f-0b4e9b2df292" width="45%" />
</div>

**[ ModernBERT ]**
<div align="center">
    <img src="https://github.com/user-attachments/assets/61a403c9-f2d1-4dc3-bd00-292e8156592f" width="45%" />
    <img src="https://github.com/user-attachments/assets/b70a002a-401e-4268-9dea-015d24aa7d6b" width="45%" />
</div>

- 큰 배치(1024)에서 성능 저하 없이 안정적으로 수렴함
- Step 단위 train accuracy를 보면
  
    - 작은 배치(64)는 변동성이 더 큼
    - 큰 배치(1024)는 더 안정적인 최적화 경향을 보임
- Validation curve 기준으로도 큰 배치가 더 높은 최종 정확도에 도달

**[ Test Accuracy Summary ]**
<div align="center">
    <img src="https://github.com/user-attachments/assets/69737444-2fe6-487f-a389-f34cbabcdc88" width="45%" />
    <img src="https://github.com/user-attachments/assets/54605dc6-fb38-4d14-873c-e79f904d693d" width="45%" />
</div>

| Batch Size | BERT   | ModernBERT |
|------------|:------:|:----------:|
| 64         | 84.06% | 89.82%     |
| 256        | 86.66% | 90.54%     |
| 1024       | **88.02%** | **91.02%**     |

- Batch size가 증가할수록 두 모델 모두 성능이 향상됨
- 64 → 1024로 증가 시
  
    - BERT: +3.96%p
    - ModernBERT: +1.20%p
- ModernBERT는 모든 batch size에서 BERT보다 높은 성능을 보임


### 2. HuggingFace Accelerate 기반 실험

<div align="center">
    <img src="https://github.com/user-attachments/assets/83df3933-9ea9-42eb-9593-bc25c5c7569e" width="45%" />
    <img src="https://github.com/user-attachments/assets/7917814b-5f32-48d4-93d9-c86814dfdab6" width="45%" />
</div>

- Batsch size가 64일 때 BERT의 결과를 예시로 첨부함
- 나머지 설정에서도 torch와 accelerate의 결과가 완전히 일치함


### 3. Learning Rate scaling 실험

```
bash scripts/run_scaling_lr.sh
```

앞선 실험에서는 Global Batch Size를 64 → 256 → 1024로 증가시켰지만, learning rate는 5e-5로 고정하였다.

하지만 
- Batch size가 커질수록 gradient noise가 감소함
- Learning rate를 고정하면 large batch는 상대적으로 작은 탐색 효과를 가짐

> 따라서 **batch size 증가에 따라 learning rate도 함께 조정**하는 것이 더 공정한 비교에 가깝다.

**[ Square-root Learning Rate Scaling Rule ]**

본 실험에서는 초기 linear scaling을 적용하였으나, large batch에서 learning rate 증가폭이 과도하여 성능 감소가 크게 나타났다.

따라서 보다 완만한 learning rate 증가를 위해 다음과 같은 sqrt scaling rule을 적용하였다.

$$ \text{LR}_\text{new} = \text{LR}_\text{base} × \sqrt{\dfrac{\text{B}_\text{new}}{\text{B}_\text{base}}} $$


| Batch Size | Learning Rate | BERT    | ModernBERT       |
|------------|--------------|----------|----------------|
| 64         | 5e-5         | **84.06%**    | 89.82%          |
| 256        | 1e-4         | 83.74%    | **90.78%**          |
| 1024       | 2e-4         | 82.54%    | 90.62%          |

- **BERT**
    - Batch size가 증가할수록 성능이 점진적으로 감소
    - Small batch 환경에서 가장 높은 성능을 기록
  
- **ModernBERT**
    - Batch size 256에서 최고 성능 달성
    - 1024에서는 소폭 감소하였으나 큰 성능 붕괴는 없음

> Batch size와 learning rate 간의 상호작용이 모델 구조에 따라 다르게 작용할 수 있다.
