# RecSys 9조 아파트 전세가 가격 예측 프로젝트
### Team

<table>
    <tr height="140px">
        <td align="center" width="130px">	
            <a href="https://github.com/minhappy68"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/127316585?v=4"/></a>
            <br />
            <a href="https://github.com/minhappy68">minhappy68
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/imnoans"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/121077194?v=4"/></a>
            <br />
            <a href="https://github.com/imnoans">imnoans
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/eatingrabbit"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/81786179?v=4"/></a>
            <br />
            <a href="https://github.com/eatingrabbit">eatingrabbit
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/hyeonjinha"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/65064566?v=4"/></a>
            <br />
            <a href="https://github.com/hyeonjinha">hyeonjinha
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/hansg931"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/118149994?v=4"/></a>
            <br />
            <a href="https://github.com/hansg931">hansg931
        </td>
    </tr>
</table>


# Contents
1. [Task Description](#task-description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Directory Structure](#directory-structure)

# Task Description
약 180만건의 아파트 주거 특성, 지하철/학교/공원 데이터, 금융 지표 데이터를 바탕으로 아파트 전세 실거래가를 예측하는 프로젝트입니다. 

예측값과 실제값 간의 Mean Absolute Error (MAE)를 평가 지표로 사용합니다.

# Installation
```bash
pip install -r requirements.txt
```

# Usage

### experiment 1

```bash
python main_experiment2.py
```
- `--eval-mae`: Evaluate MAE on holdout set

### experiment 2

```bash
python main_experiment2.py
```

- `--force-preprocess`: Force reprocessing of the raw data and skip loading processed features.
- `--n-trials`: Specify the number of trials for hyperparameter optimization (default: 100).
- `--n-jobs`: Number of jobs to run in parallel (default: 1). `-1` means using all CPU threads.


# Directory Structure

![initial](https://github.com/user-attachments/assets/5866a3a9-a8ac-4987-b074-e4c7995fe1a3)

```
deposit_prediction
├── README.md
├── data            
│   ├── preprocessed
│   ├── processed_features
│   └── raw                      
├── main_experiment1.py               # 실험 1 실행 파일
├── main_experiment2.py               # 실험 2 실행 파일
├── config.yaml                       # 모델 파라미터 파일
├── notebooks
├── requirements.txt
└── src
    ├── data
    │   ├── __init__.py
    │   └── data_loader.py
    ├── evaluation
    │   ├── __init__.py
    │   └── holdout.py
    ├── features
    │   ├── __init__.py
    │   ├── one_hot_encoding.py         # category feature를 one-hot encoding
    │   ├── year_month_day.py           # year와 month와 contract_day를 합친 contract 날짜 feature 추가
    │   ├── clustering.py               # 실제 거리를 고려한 클러스터링 수행
    │   ├── cluster_deposit_per_area.py # 클러스터별 평당 평균 가격 계산 및 feature 추가
    │   ├── cluster_feature_add.py      # 수행한 결과 리턴
    │   ├── weight.py                   # 특정 데이터에 가중치 열 추가
    │   ├── age_weight.py               # age를 구간별로 나누어 weight를 추가
    │   ├── feature_engineering.py
    │   ├── nearest_public.py
    │   ├── contract_timestamp.py       # 계약 날짜 피처 생성
    │   ├── deposit_per_area.py          # 평당 전세가 피처 생성
    │   ├── feature_engineering_1.py     # 실험1을 위한 피처 엔지니어링 함수
    │   ├── feature_engineering2.py      # 실험2를 위한 피처 엔지니어링 함수
    │   ├── label_encoding.py            # 레이블 인코딩
    │   ├── nearest_school.py            # 가까운 초중고 정보 피처 생성
    │   ├── park_size.py                 # 근처 공원의 크기 피처 생성
    │   ├── recent_deposit.py            # 최근 전세가 정보 피처 생성  
    ├── models
    │   ├── __init__.py
    │   ├── lasso.py
    │   ├── lgb.py
    │   ├── linear_regression.py
    │   ├── ridge.py
    │   ├── train_model.py
    │   ├── xgb.py
    │   ├── cat.py                       # CatBoost 모델
    │   ├── lgb_batch.py                 # LightGBM 배치 학습
    │   ├── xgb_batch.py                 # XGBoost 배치 학습
    │   ├── SLSQP.py                     # 최적 가중치 탐색용 SLSQP 함수
    │   ├── train_model_batch.py         # 실험 1의 배치 학습 전체 데이터셋  훈련 함수
    │   ├── train_model_eval.py          # 실험 1의 모델 학습 및 holdout 평가 함수
    │   ├── train_model_eval_2.py          # 실험 2의 모델 학습 및 holdout 평가 함수
    ├── preprocessing
    │   ├── __init__.py
    │   ├── data_preprocessing.py
    │   ├── remove_data.py
    │   └── split_data.py
    └── utils
        ├── submission.py
        └── variables.py

```
