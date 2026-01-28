# Image Classification Coding Task
### **CIFAR-10 기반 이미지 분류 실험**

본 repository는 학습 데이터 크기 변화에 따른 모델 성능 변화를 분석하기 위한 이미지 분류 실험을 다룬다.
특히 CNN 기반 모델과 Transformer 기반 모델의 특성 차이, 그리고 pre-training의 효과를 비교·분석하는 것을 목표로 한다.

----

# Dataset
### **CIFAR-10**

CIFAR-10은 이미지 분류 분야에서 널리 사용되는 대표적인 벤치마크 데이터셋으로, 일상적인 사물 이미지를 10개의 클래스로 분류하는 문제로 구성되어 있다.
- **Total:** 60k
- **Train:** 50k
- **Test:** 10k
- **클래스 수**: 10개 (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)
- **이미지 크기:** 32×32

CIFAR-10은 데이터 규모가 비교적 작고 클래스 분포가 균형 잡혀 있어, 모델의 데이터 효율성 및 일반화 성능을 비교하는 실험에 적합한 데이터셋이다.

----

# Model
본 실험에서는 서로 다른 구조적 특성을 가진 두 가지 vision 모델을 선정하고,
각각에 대해 pretraining 여부에 따른 성능 차이를 함께 비교한다.

### 사용한 아키텍처
- ResNet50
- ViT-S/16 (Vision Transformer, Patch Size 16)

### 모델 구성 (총 4가지)
| Model     | Architecture | Pre-trained |
|-----------|--------------|-------------|
| ResNet50  | CNN          | No (Scratch) |
| ResNet50  | CNN          | Yes (ImageNet-1k) |
| ViT-S/16  | Transformer  | No (Scratch) |
| ViT-S/16  | Transformer  | Yes (ImageNet-1k) |

이를 통해 아키텍처 차이(CNN vs Transformer)와 사전학습 효과를 동시에 분석할 수 있다.

----

# Setup
### 데이터 및 학습 설정
- **데이터셋:** CIFAR-10
- **입력 이미지 크기:** 224 × 224
- **학습 epoch 수:** 20
- **배치 크기:** 64
- **손실 함수:** Cross Entropy Loss

### Optimizer 설정
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Learning Rate Scheduler:** Cosine Annealing
- **Betas:** (0.9, 0.999)
- **Epsilon:** 1e-8
- **Weight Decay:** 1e-4

### Train / Validation Split 방식

Training 데이터에서 validation set을 분리할 때,  
클래스 비율을 유지하기 위해 **stratified split** 방식을 적용한다.

Stratified split을 사용함으로써,  
training set과 validation set이 **각 클래스에 대해 유사한 분포**를 가지도록 보장하며,  
데이터 분할로 인한 성능 편차를 최소화한다.

### Training Data Subsampling

학습 데이터 크기 변화 실험을 위해,  
training set에 대해 **사용 비율(train fraction)을 조절하는 방식의 subsampling**을 수행한다.

- 지정된 train fraction에 따라 training data를 부분적으로 사용
- Subsampling 과정에서도 **stratified sampling**을 적용하여  
  각 클래스의 비율이 유지되도록 한다

이를 통해 학습 데이터의 절대적인 양이 줄어들더라도,  
데이터 분포 왜곡 없이 모델의 데이터 효율성을 비교할 수 있도록 설계하였다.

----

# Metrics
모델의 성능은 다음 지표를 기준으로 평가한다.
- Accuracy (Top-1)
- Top-1 Error (=1-Accuracy)
- Top-5 Error

이때 Top-5 Error는 모델이 예측한 상위 5개 클래스 중 정답이 포함되지 않은 비율을 의미한다.

----

# Experiments
### 실험 목적
본 실험에서는 학습 데이터의 크기가 증가함에 따라 모델 성능이 어떻게 변화하는지를 분석하고자 한다.
특히 다음과 같은 관점에서 비교를 수행한다.

**1. CNN과 Vision Transformer의 특성 비교**

CNN 기반 모델은 locality과 translation equivariance와 같은 강한 inductive bias를 가지고 있어,
상대적으로 적은 데이터에서도 안정적인 학습이 가능하다.

반면 Vision Transformer는 이러한 inductive bias가 약한 대신,
충분한 데이터가 주어질 경우 더 높은 표현력을 발휘하는 경향이 있다.

**2. Pre-training의 효과 분석**

ImageNet 기반 사전학습은 대규모 데이터에서 학습된 일반적인 시각적 표현을 제공하여,
소규모 데이터셋에서도 성능 향상 및 빠른 수렴을 가능하게 한다.

특히 Vision Transformer는 사전학습 여부에 따라 데이터 효율성 차이가 크게 나타날 가능성이 있다.

**3. 학습 데이터 크기 변화 실험의 적합성**

학습 데이터의 사용 비율을 점진적으로 변화시키는 방식은,

- 모델의 data efficiency
- pre-training이 저데이터 환경에서 얼마나 효과적인지
- 아키텍처 간 성능 격차가 데이터 규모에 따라 어떻게 변화하는지

를 분석하기에 적합한 설정이다.
