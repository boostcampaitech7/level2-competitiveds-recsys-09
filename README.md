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