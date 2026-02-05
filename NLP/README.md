# IMDB Sentiment Classification with Encoder Models

본 repository는 **문장 분류(sentence classification)** 문제에서
서로 다른 **Encoder 기반 언어 모델**이 성능 및 학습 안정성 측면에서 어떤 차이를 보이는지를 분석하기 위한 실험을 다룬다.

특히, **표준 BERT 계열 모델과 ModernBERT 계열 모델을 동일한 학습 파이프라인에서 비교**하여
모델 구조 및 사전학습 방식 차이가 감정 분류 성능에 미치는 영향을 분석하는 것을 목표로 한다.

---

## Directory Structure

```
NLP
├── README.md                     # 프로젝트 개요 및 실험 결과 정리
│
├── configs                       # 실험 설정 관리 (Hydra 기반)
│   ├── default.yaml              # 공통 기본 설정
│   └── model                     # 모델 설정
│       ├── bert.yaml
│       └── modernbert.yaml
│
├── src                           # 핵심 로직 코드
│   ├── data.py                   # IMDB 데이터 로딩 및 토크나이징
│   ├── model.py                  # EncoderForClassification 정의
│   └── utils.py                  # seed, device, config, wandb util
│
├── main.py                       # 전체 학습 / 평가 entry point
└── scripts
    └── run_all.sh                # BERT / ModernBERT 실험 자동 실행
```

---

## How to Run

### 1. Weights & Biases 설정: Edit configs/default.yaml

```
wandb login [API_KEY]
```


### 2. 실험 실행

BERT 및 ModernBERT를 순차적으로 실행

```
bash scripts/run_all.sh
```

---

## Dataset

### IMDB Sentiment Classification

IMDB 데이터셋은 영화 리뷰 텍스트에 대한 **이진 감정 분류(positive / negative)** 문제로 구성된 대표적인 NLP 벤치마크이다.

* 전체 데이터: 50k 문장

  * Train(25k) + Test(25k) 통합 후 **8:1:1 분할**
* 클래스 수: 2
* 장점:

  * 문장 간 문맥 이해 능력 평가에 적합
  * Encoder 기반 representation learning 성능 분석에 적절

---

## Model

본 실험에서는 **Encoder-only Transformer 계열 모델** 두 가지를 비교한다.

### 사용한 모델

#### BERT-base-uncased

* Transformer encoder 기반
* NSP + MLM 기반 사전학습
* WordPiece tokenizer
* 12 layers / 768 hidden / 12 heads

#### ModernBERT-base

* Transformer encoder 기반 BERT 계열 모델
* NSP 제거, MLM 중심의 사전학습
* 최신 실행 및 연산 최적화 기법을 통한 학습·추론 효율 개선

---

### Classification Head

모든 실험에서 동일한 분류 구조를 사용한다.

* Encoder의 **[CLS] token representation** 사용
* Dropout
* Linear layer → 2-class logits

---

## Setup

### 데이터 및 학습 설정

* Epochs: 3
* Batch size: 32
* Max sequence length: 128
* Random seed: 42 (재현성 보장)
* Loss: Cross Entropy Loss

## Optimizer 설정
* Optimizer: Adam
* Learning Rate: 5e-5
* Betas: (0.9, 0.999)
* Epsilon: 1e-6
* Weight Decay: 0.01

---

## Metrics

모델 성능 평가는 다음 지표를 기준으로 수행한다.

* Accuracy

모든 실험은 **epoch 단위로 validation을 수행**하며,
validation 성능이 가장 우수한 checkpoint를 기준으로 test 성능을 측정한다.

---

## Experiments

### Loss
<div align="center">
    <img src="https://github.com/user-attachments/assets/a0160e12-6008-412f-a12e-333266f450d5" width="92%" />
</div>

<div align="center">
    <img src="https://github.com/user-attachments/assets/69a09ec4-4fe9-4b89-89c4-cc937ea2b386" width="45%" />
    <img src="https://github.com/user-attachments/assets/ac412801-0503-4f19-b03c-f1fc4c44f0d9" width="45%" />
</div>

<br/>

### Accuracy

<div align="center">
    <img src="https://github.com/user-attachments/assets/4aac937d-6aa9-4700-8f1a-de9bf3942403" width="92%" />
</div>

<div align="center">
    <img src="https://github.com/user-attachments/assets/e417bc9b-fa3f-460f-aee0-7bf9a6e84e44" width="45%" />
    <img src="https://github.com/user-attachments/assets/35c2cf4a-9c08-45e2-8052-dd131f561679" width="45%" />
</div>

<br/>

### Observations

#### 1. Step-level 관찰 (Training Dynamics)

- **ModernBERT**

    - 전반적으로 높은 accuracy band(≈0.8~0.9)를 안정적으로 유지
    
    - 초반 수렴이 빠르고, 큰 붕괴 없이 유지됨

- **BERT**

    - 중간 구간에서 뚜렷한 accuracy drop 발생
    
    - 이후 회복은 하지만, 변동성이 더 큼

> **ModernBERT는 step-level에서 학습 안정성이 더 높음**

#### 2. Epoch-level 비교 (Generalization 관점)
**[ Train ]**

- **ModernBERT**

    - epoch이 진행되어도 성능 저하 없이 안정적

- **BERT**

    - epoch 2에서 성능 하락 후 회복

**[ Validation / Test ]**

| Model      | Validation Acc | Test Acc |
| :----------: | :--------------: | :--------: |
| BERT-base  | 82.18%         | 81.46%   |
| ModernBERT | 88.84%         | 83.92%   |

> **ModernBERT는 Validation에서 +6.66%p, Test에서 +2.46%p**


---

## Conclusions

본 실험에서는 IMDB sentence classification task에서
BERT-base-uncased와 ModernBERT-base를 동일한 학습 설정과 분류 구조 하에서 비교하였다.

실험 결과, ModernBERT는 step-level과 epoch-level 모두에서 더 안정적인 학습 양상을 보였으며,
Validation 및 Test 성능에서도 BERT-base 대비 일관되게 우수한 결과를 기록하였다.
특히 Validation Accuracy에서 +6.66%p, Test Accuracy에서 +2.46%p의 개선이 관찰되었다.

이러한 결과는 ModernBERT가 Transformer encoder 구조 자체를 변경하지 않고도,
사전학습 목표 단순화(NSP 제거)와 실행 및 연산 최적화를 통해
representation quality와 일반화 성능을 동시에 개선할 수 있음을 시사한다.

종합적으로, 본 실험은 ModernBERT가 단순한 BERT 변형이 아니라
실제 downstream task에서 의미 있는 성능 및 안정성 향상을 제공하는 모델임을 확인하였다.
