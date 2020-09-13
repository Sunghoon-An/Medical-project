# Doctor miss handling decision classification

의사들이 잘못 내린 판단을 탐지하기 위한 모형 개발. 
Pannel Data를 기반으로 Classification Model 개발을 진행했으며, Model은 MLP 기반 Res-Net Architecture 사용. 

## Data Preprocessing and imbalanced Handling
Normality test / Significant test를 통해 도출된 유의 변수 Imputation.
VOS(Variational OverSampling)을 통해 minority data를 handling.

## Performance Estimation
5 fold crosss validation을 진행하였으며, 각 Fold에 대한 성능 지표로는 정확도(Accuracy), 민감도(Recall), 특이도(Precision), AUROC를 사용하여 학습하고 평가에 사용.
 
## Prerequisites
- **Python** ≥ `3.6.8`
- **Keras**     ≥            `2.2.4`
- **matplotlib**    ≥       `3.0.2`
- **numpy**           ≥      `1.16.0`
- **opencv-python**    ≥     `3.4.3.18`
- **pandas**        ≥        `0.23.4`
- **polarTransform**    ≥    `2.0.0`
- **scikit-learn** ≥         `0.20.2`
- **scipy**      ≥           `1.2.0`
- **sklearn**      ≥         `0.0`
- **tensorboard**   ≥        `1.12.2`
- **tensorflow**  ≥         `1.12.0`
- **tflearn**      ≥         `0.3.2`
- **tqdm**         ≥         `4.31.1`
