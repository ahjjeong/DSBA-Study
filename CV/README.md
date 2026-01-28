# Image Classification Coding Task
### **CIFAR 기반 이미지 분류 실험**

본 repository는 학습 데이터의 규모 및 난이도 변화에 대해 이미지 분류 모델이 얼마나 강건하게 성능을 유지하는지를 분석하기 위한 실험을 다룬다.

### Hypotheses

본 실험은 데이터 규모 및 난이도 변화에 대한 모델 강건성을 중심으로 다음 가설들을 설정하였다.

- **H1.** Pre-training은 데이터 규모 감소 환경에서 모델의 성능 저하를 완화하여 강건성을 향상시킨다.

- **H2.** Vision Transformer는 CNN보다 inductive bias가 약해, pre-training 없이 학습할 경우 저데이터 환경에서 더 취약하다.

- **H3.** Dataset 난이도(CIFAR-10 → CIFAR-100)가 증가할수록 ViT는 ResNet보다 성능 악화에 더 민감하다.

----


# Directory Structure


```text
CV
├── README.md                     # 프로젝트 개요 및 실험 결과 정리
├── requirements.txt              # Python 패키지
│
├── configs                       # 실험 설정 관리 (YAML)
│   ├── default.yaml              # 공통 기본 설정
│   │
│   ├── dataset                   # 데이터셋 관련 설정
│   │   ├── cifar10.yaml
│   │   └── cifar100.yaml
│   │
│   ├── model                     # 모델 아키텍처 설정
│   │   ├── resnet50_im1k.yaml
│   │   ├── resnet50_scratch.yaml
│   │   ├── vit_s16_im1k.yaml
│   │   └── vit_s16_scratch.yaml
│   │
│   └── optimizer                 # Optimizer 설정
│       └── adam.yaml
│
├── src                           # 핵심 로직 코드
│   ├── data.py                   # 데이터 로딩 및 전처리
│   ├── models.py                 # 모델 생성 (config 기반)
│   ├── train_eval.py             # 학습 및 평가 루프
│   ├── metrics.py                # metric 계산
│   ├── utils.py                  # 공통 유틸리티
│   └── wandb_utils.py            # Weights & Biases 로깅
│
├── main.py                       # 전체 실험 entry point
├── run_scale_all.sh              # train fraction 실험 자동 실행
└── run_cifar100_all.sh           # CIFAR-100 실험 자동 실행
```

----

# How to Run
**1. 라이브러리 설치**
```text
pip install -r requirements.txt
```
**2. WandB API key 설정:** Edit ```configs/default.yaml```
```text
wandb login [API_KEY]
```
**3. Run training jobs**
- Train fraction 실험
```text
bash run_scale_all.sh
```
- CIFAR-100 실험
```text
bash run_cifar100_all.sh
```

----

# Dataset
## CIFAR-10
CIFAR-10은 이미지 분류 분야에서 널리 사용되는 대표적인 벤치마크 데이터셋으로, 일상적인 사물 이미지를 10개의 클래스로 분류하는 문제로 구성되어 있다.
- **Total:** 60k (Train 50k / Test 10k)
- **클래스 수:** 10
- **이미지 크기:** 32×32
→ 상대적으로 난이도가 낮아 데이터 효율성 및 저데이터 강건성 분석에 적합

## CIFAR-100
- **Total:** 60k (Train 50k / Test 10k)
- **클래스 수:** 100
- **이미지 크기:** 32×32
→ 클래스 수 증가로 인해 dataset 난이도가 상승, 모델의 class complexity에 대한 강건성을 평가하기에 적합


----

# Model
본 실험에서는 서로 다른 구조적 특성을 가진 두 가지 vision 모델을 선정하고, 각각에 대해 pretraining 여부에 따른 성능 차이를 함께 비교한다.

## 사용한 아키텍처
- **ResNet50 (CNN)**
  - 강한 locality 및 translation equivariance
  - 저해상도·고난이도 환경에서 상대적으로 안정적인 학습 특성

- **ViT-S/16 (Transformer)**
  - 약한 inductive bias, global attention 기반
  - pre-training 의존도가 높은 구조

## 모델 구성 (총 4가지)
| Model     | Architecture | Pre-trained |
|:---------:|:------------:|-------------|
| ResNet50  | CNN          | No (Scratch) |
| ResNet50  | CNN          | Yes (ImageNet-1k) |
| ViT-S/16  | Transformer  | No (Scratch) |
| ViT-S/16  | Transformer  | Yes (ImageNet-1k) |

이를 통해 아키텍처 차이(CNN vs Transformer)와 사전학습 효과를 동시에 분석할 수 있다.

----

# Setup
## 데이터 및 학습 설정
- **데이터셋:** CIFAR-10, CIFAR-100
- **입력 이미지 크기:** 224 × 224
- **학습 epoch 수:** 20
- **배치 크기:** 64
- **손실 함수:** Cross Entropy Loss


## Optimizer 설정
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Learning Rate Scheduler:** Cosine Annealing
- **Betas:** (0.9, 0.999)
- **Epsilon:** 1e-8
- **Weight Decay:** 1e-4


## Train / Validation Split 방식

Training 데이터에서 validation set을 분리할 때, 클래스 비율을 유지하기 위해 **stratified split** 방식을 적용한다. 이를 통해 training set과 validation set이 **각 클래스에 대해 유사한 분포**를 가지도록 보장하며, 데이터 분할로 인한 성능 편차를 최소화한다.


## Training Data Subsampling

학습 데이터 크기 변화 실험을 위해, training set에 대해 **사용 비율(train fraction)을 조절하는 방식의 subsampling**을 수행한다.

- 지정된 train fraction에 따라 training data를 부분적으로 사용
- Subsampling 과정에서도 **stratified sampling**을 적용  

하여 각 클래스의 비율이 유지되도록 한다

이를 통해 학습 데이터의 절대적인 양이 줄어들더라도, 데이터 분포 왜곡 없이 모델의 데이터 효율성을 비교할 수 있도록 설계하였다.

----

# Metrics
모델의 성능은 다음 지표를 기준으로 평가한다.
- Accuracy (Top-1)
- Top-1 Error (=1-Accuracy)
- Top-5 Error

이때 Top-5 Error는 모델이 예측한 상위 5개 클래스 중 정답이 포함되지 않은 비율을 의미한다.

----

# Experiments
## 실험 목적
본 실험은 다음 세 가지 강건성 관점에서 설계되었다.

**1. 데이터 규모 감소에 대한 강건성**
- 학습 데이터 비율(train fraction)을 줄였을 때 성능이 얼마나 안정적으로 유지되는가

**2. Pre-training에 따른 강건성 변화**
- pre-training이 저데이터 환경 및 성능 변동성 감소에 얼마나 기여하는가

**3. Dataset 난이도 증가에 대한 강건성**
- CIFAR-10 → CIFAR-100으로 난이도가 증가했을 때 모델 성능이 얼마나 악화되는가


## 실험 결과
### 1. Pretraining에 따른 강건성 개선
|   Model  | Train Fraction | Scratch | Pretrained | Δ (Pretrained-Scratch) |
| :------: | :------------: | :-----: | :--------: | :--------------------: |
| ResNet50 |      1.0       |  0.104  |   0.039    |      **−6.50%**        |
| ResNet50 |      0.5       |  0.149  |   0.043    |     **−10.62%**        |
| ResNet50 |      0.2       |  0.256  |   0.050    |     **−20.61%**        |
| ResNet50 |      0.1       |  0.341  |   0.067    |     **−27.46%**        |
| ViT-S/16 |      1.0       |  0.320  |   0.066    |     **−25.43%**        |
| ViT-S/16 |      0.5       |  0.373  |   0.072    |     **−30.08%**        |
| ViT-S/16 |      0.2       |  0.472  |   0.091    |     **−38.05%**        |
| ViT-S/16 |      0.1       |  0.545  |   0.093    |     **−45.19%**        |

*Note. Δ(Pretrained–Scratch)는 Top-1 error의 절대 감소량 × 100을 의미함*

- Pretraining은 모든 설정에서 성능을 개선하며, 데이터 감소에 대한 강건성을 크게 향상
- 데이터가 적어질수록 pretraining의 효과는 더욱 커짐
- ViT는 scratch 상태에서 매우 취약하지만, pretraining을 통해 강건성이 크게 회복됨


### 2. 학습 데이터 규모 감소에 대한 민감도
<div align="center">
<img width="80%" height="1380" alt="image" src="https://github.com/user-attachments/assets/3bab8366-6f09-41cd-ba28-35b8bd10332e" />
</div>

- 데이터 감소에 따른 성능 변화는 pretraining 여부에 의해 가장 강하게 구분됨
- Pretrained 모델은 train fraction 감소에도 성능 저하 폭이 작아 안정적
- 모델 구조(ResNet vs ViT)에 따른 민감도 차이는 상대적으로 작음

### 3. Dataset 난이도 증가에 대한 강건성 (CIFAR-10 → CIFAR-100)
|   Model  | Pretrained |  CIFAR-10 | CIFAR-100 | Δ (C100 − C10) |
| :------: | :--------: | :-------: | :-------: | :------------: |
| ResNet50 |      ❌     |   0.104   |   0.323   |   **+21.88%**  |
| ViT-S/16 |      ❌     |   0.320   |   0.582   |   **+26.15%**  |
| ResNet50 |      ✅     | **0.039** | **0.161** |   **+12.25%**  |
| ViT-S/16 |      ✅     |   0.066   |   0.227   |   **+16.13%**  |

- Dataset 난이도 증가 시 모든 모델의 성능이 악화됨
- ResNet는 ViT보다 난이도 증가에 더 강건
- Pretraining은 두 모델 모두에서 난이도 증가에 따른 성능 악화를 완화

----

# Conclusions (Hypothesis Verification)

실험 결과를 통해 위 가설들은 다음과 같이 검증되었다.

- **H1 검증**
  - Pretrained 모델은 모든 train fraction에서 scratch 모델보다 낮은 error를 기록했으며, 데이터가 줄어들수록 성능 격차가 확대되었다. 이는 pre-training이 데이터 감소에 대한 강건성을 실질적으로 향상시킴을 보여준다.

- **H2 검증**
  - ViT는 scratch 상태에서 저데이터 환경에서 급격한 성능 저하를 보였으나, pre-training 적용 시 성능이 크게 회복되었다. 이는 ViT가 구조적으로 pre-training에 강하게 의존함을 확인한다.

- **H3 검증**
  - CIFAR-10 대비 CIFAR-100에서 ViT의 성능 악화 폭이 ResNet보다 더 크게 나타났으며, 이는 class complexity 증가에 대한 민감도가 ViT에서 더 큼을 의미한다.

----

> 데이터 규모 감소 및 dataset 난이도 증가 환경에서는 ResNet이 전반적으로 더 안정적인 강건성을 유지함을 확인하였다.
그러나 pre-training을 적용한 ViT는 전체 학습 데이터의 10%만으로도 scratch 상태의 ResNet(100%)을 능가하는 성능을 보였다.
이는 **pre-training이 아키텍처 차이를 넘어 모델 강건성을 좌우하는 핵심 요인**임을 명확히 보여준다.
