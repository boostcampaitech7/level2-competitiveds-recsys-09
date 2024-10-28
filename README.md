# Contents
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)

# Installation
```bash
pip install -r requirements.txt
```

# Usage
```bash
python main.py
```

## Command Line Arguments
- `--force-preprocess`: Force reprocessing of the raw data and skip loading processed features.
- `--model`Choose the model to train (options: lgb, xgb, default: lgb).
- `--n-trials`: Specify the number of trials for hyperparameter optimization (default: 100).
- `--n-jobs`: Number of jobs to run in parallel (default: 1). `-1` means using all CPU threads.

### example
```bash
python main.py --force-preprocess --model xgb --n-trials 10 --n-jobs 2
```

# Directory Structure
```
deposit_prediction
├── README.md
├── data            
│   ├── preprocessed
│   ├── processed_features
│   └── raw                      
├── main.py
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
    │   └── nearest_public.py
    ├── models
    │   ├── __init__.py
    │   ├── lasso.py
    │   ├── lgb.py
    │   ├── linear_regression.py
    │   ├── ridge.py
    │   ├── train_model.py
    │   └── xgb.py
    ├── preprocessing
    │   ├── __init__.py
    │   ├── data_preprocessing.py
    │   ├── remove_data.py
    │   └── split_data.py
    └── utils
        ├── submission.py
        └── variables.py

```