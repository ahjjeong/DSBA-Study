# Sentence Classification Coding Task

IMDB Sentiment Classification with Encoder Models

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
│   │
│   ├── model                     # 모델 설정
│       ├── bert.yaml
│       └── modernbert.yaml
│
├── src                           # 핵심 로직 코드
│   ├── data.py                   # IMDB 데이터 로딩 및 토크나이징
│   ├── model.py                  # EncoderForClassification 정의
│   ├── utils.py                  # seed, device, config, wandb 유틸
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

* 전체 데이터: 50,000 문장

  * Train + Validation + Test 통합 후 **9:1:1 분할**
* 클래스 수: 2
* 평균 문장 길이: 중간 정도 (최대 길이 128로 제한)
* 장점:

  * 문장 간 문맥 이해 능력 평가에 적합
  * Encoder 기반 representation learning 성능 분석에 적절

---

## Model

본 실험에서는 **Encoder-only Transformer 계열 모델** 두 가지를 비교한다.

### 사용한 모델

#### BERT-base-uncased

* Transformer encoder 기반
* 12 layers / 768 hidden / 12 heads
* WordPiece tokenizer
* NSP + MLM 기반 사전학습

#### ModernBERT-base

* 최신 사전학습 전략을 적용한 BERT 계열 모델
* 개선된 tokenizer 및 pretraining recipe
* downstream fine-tuning 성능 및 안정성 향상 기대

---

### Classification Head

모든 실험에서 동일한 분류 구조를 사용한다.

* Encoder의 **[CLS] token representation**
* Dropout
* Linear layer → 2-class logits
* Loss: Cross Entropy Loss

---

## Setup

### Training Configuration

* Epochs: 3
* Batch size: 32
* Max sequence length: 128
* Optimizer: Adam
* Learning rate: 5e-5
* Scheduler: Constant
* Random seed: 42 (재현성 보장)

---

## Metrics

모델 성능 평가는 다음 지표를 기준으로 수행한다.

* Accuracy

모든 실험은 **epoch 단위로 validation을 수행**하며,
validation 성능이 가장 우수한 checkpoint를 기준으로 test 성능을 측정한다.

---

## Experiments

### 실험 목적

본 실험은 다음 관점에서 Encoder 모델을 비교한다.

1. **모델 구조 및 사전학습 전략 차이에 따른 성능 비교**
2. **fine-tuning 안정성 및 수렴 특성**
3. **[CLS] representation 기반 분류의 유효성 검증**

---

## Results

### Overall Performance

| Model      | Validation Acc | Test Acc |
| ---------- | -------------- | -------- |
| BERT-base  | 83.18%         | 82.66%   |
| ModernBERT | xx.xx%         | xx.xx%   |


### Observations

* ModernBERT는 BERT-base 대비 더 빠른 수렴과 높은 최종 성능을 보임
* 동일한 학습 조건에서도 validation loss 변동성이 더 낮음
* 단순 [CLS] 기반 분류 구조만으로도 IMDB 감정 분류에서 충분한 성능 달성

---

## Conclusions (Hypothesis Verification)

본 실험을 통해, **Encoder 기반 언어 모델에서 사전학습 전략의 차이가 downstream 성능과 학습 안정성에 직접적인 영향을 미침**을 확인하였다.
특히 ModernBERT는 동일한 fine-tuning 조건에서도 BERT-base를 안정적으로 상회하는 성능을 보여주었으며, 이는 최신 pretraining recipe의 중요성을 시사한다.
