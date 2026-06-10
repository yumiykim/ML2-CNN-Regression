# ML2 Regression — 1D CNN + MAPE-Aligned Loss

> 동국대학교 머신러닝2 수업 회귀 예측 프로젝트  
> **리더보드 최종 1위 달성 · MAPE 27.57** (조교 CNN 베이스라인 27.67 대비 0.1p 개선)

---

## Overview

12,288차원 고차원 벡터에서 연속형 타깃값을 예측하는 회귀 문제입니다.  
평가 지표는 **MAPE(Mean Absolute Percentage Error)** 이며, 수강생 전체 리더보드 기준 1위를 달성했습니다.

단순 MSE 학습에서 시작해, **평가 지표에 직접 정렬된 Custom Loss 설계**와 **Multi-seed Ensemble** 전략을 통해 성능을 단계적으로 끌어올렸습니다.

---

## Dataset

데이터는 수업에서 제공된 자료이므로 이 레포지토리에 포함되지 않습니다.  
실행 시 `train.npy`와 `test_x.npy`를 프로젝트 루트에 직접 위치시켜야 합니다.

| 파일 | Shape | 설명 |
|------|:-----:|------|
| `train.npy` | (4310, 12289) | 마지막 열이 타깃 y |
| `test_x.npy` | (4311, 12288) | 예측 대상 (타깃 없음) |

**타깃 y 분포:** min=0.0 / max=0.685 / mean≈0.209  
> 값이 0 근처에 몰려 있어 MAPE 계산 시 분모 불안정 문제가 발생할 수 있습니다.

---

## Approach

### 1. 1D CNN on High-Dimensional Vectors

12,288개 피처를 독립적인 변수로 보는 MLP 대신, **1차원 시퀀스로 간주하고 `Conv1d`를 적용**했습니다.  
stride=2 합성곱을 반복해 차원을 줄이면서 로컬 패턴을 추출합니다.  
파라미터 효율성과 표현력을 동시에 확보하는 구조입니다.

### 2. MAPE-Aligned Custom Loss

이 프로젝트에서 가장 큰 성능 도약을 만든 전략입니다.

초기 `MSELoss` 학습에서는 학습 목표(MSE 최소화)와 평가 지표(MAPE 최소화)가 달라,  
검증 점수와 리더보드 점수 사이에 지속적인 괴리가 있었습니다.

이를 해결하기 위해 **평가 지표와 동일한 형태의 손실 함수**를 직접 설계했습니다:

```python
def mape_loss_for_training(y_true, y_pred, eps=0.01, lambda_mse=0.1):
    mape_term = torch.mean(torch.abs((y_true - y_pred) / (y_true + eps)))
    mse_term  = torch.mean((y_true - y_pred) ** 2)
    return mape_term + lambda_mse * mse_term
```

| 항목 | 역할 |
|------|------|
| `MAPE term` | 리더보드 평가 지표를 직접 최적화 |
| `eps=0.01` | y ≈ 0 근처에서 분모 폭주 방지 |
| `MSE × 0.1` | 소량 혼합으로 학습 안정화 |

### 3. Multi-seed Ensemble

`random_state = 10, 11, 12`으로 동일한 구조의 모델을 각각 학습한 뒤,  
test 예측값을 단순 평균했습니다.  
단일 데이터 분할에 의존하는 분산을 줄여 리더보드 성능이 안정적으로 향상됐습니다.

---

## Model Architecture — `CNNRegressorV1Drop`

```
Input  : (batch, 12288)
           ↓ unsqueeze
         (batch, 1, 12288)
           ↓ Conv1d(1→16, k=5, stride=2) + ReLU
           ↓ Conv1d(16→32, k=5, stride=2) + ReLU
           ↓ Conv1d(32→64, k=5, stride=2) + ReLU
           ↓ Flatten  →  (batch, 64×1536)
           ↓ Linear(98304 → 256) + ReLU + Dropout(0.2)
           ↓ Linear(256 → 1)
Output : (batch, 1)
```

---

## Experiments & Results

### 제출 이력

| 제출 | 전략 | Leaderboard MAPE |
|:----:|------|:----------------:|
| 1차 | 초기 CNN + MSE Loss | 43.31 |
| 2차 | CNNRegressorV1Drop + MSE Loss | 34.19 |
| **최종** | **CNNRegressorV1Drop + MAPE Loss + 3-seed Ensemble** | **27.57** |

### 단일 모델 검증 결과 (MAPE-loss 적용 후)

| Seed | Best Val MAPE |
|:----:|:-------------:|
| 10 | 22.981 |
| 11 | 24.457 |
| 12 | 23.836 |

### 리더보드 최종 순위 (일부)

```
1위  peterparker           27.57   (제출자 본인)
2위  조교(CNN Regression)  27.67
3위  c00                   27.84
4위  쌍00                28.07
```

---

## Training Config

| 항목 | 값 |
|------|:---:|
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Weight decay | 1e-6 |
| Batch size | 16 |
| Max epochs | 80 |
| Early stopping | patience=10 (val MAPE 기준) |
| Ensemble seeds | 10, 11, 12 |

---

## How to Run

```bash
# 의존성 설치
pip install -r requirements.txt

# 학습 및 예측 실행
# (train.npy, test_x.npy를 프로젝트 루트에 위치시킨 후 실행)
python train.py
```

실행 후 `submission.csv`가 생성됩니다. (4311×1, 헤더 없음)

---

## Key Takeaways

**1. 모델 구조보다 평가 지표 이해가 먼저였습니다.**  
더 깊은 CNN, StandardScaler, 2D CNN 등 다양한 구조를 실험했지만,  
결정적인 도약은 구조 개선이 아니라 *metric mismatch 문제를 인식하고 손실 함수를 재설계*한 시점에서 나왔습니다.

**2. Validation score를 맹신하면 안 됩니다.**  
MSE로 학습할 때는 val MAPE와 리더보드 MAPE 사이에 지속적인 괴리가 있었습니다.  
리더보드 피드백을 기준으로 전략을 수정하는 과정이 중요했습니다.

**3. 간단한 앙상블이 안정적으로 작동합니다.**  
복잡한 weighted ensemble 없이, seed만 달리한 단순 평균이 최종 성능 안정화에 기여했습니다.

---

## Stack

`Python` `PyTorch` `NumPy` `scikit-learn`  
`1D CNN` `Custom Loss Function` `MAPE` `Metric-aware Optimization` `Multi-seed Ensemble`

---

*동국대학교 산업시스템공학과 · 머신러닝2 (2025-2학기)*
