import time
from src.models.lgb_batch import train_lightgbm_in_batches_for_holdout
from src.models.xgb_batch import train_xgboost_in_batches_for_holdout
from src.models.cat import train_catboost_in_batches_for_holdout
from pandas import DataFrame
from typing import Dict, Tuple

def train_and_evaluate_models(
    X_train: DataFrame, 
    y_train: DataFrame, 
    X_holdout: DataFrame, 
    y_holdout: DataFrame, 
    params_lgb: Dict, 
    params_xgb: Dict, 
    params_cat: Dict, 
    batch_num: int = 30
) -> Tuple[float, float, float]:
    """
    LightGBM, XGBoost, CatBoost 모델을 배치 학습하고 holdout MAE를 확인하는 함수.

    :param X_train: (DataFrame) 학습 데이터 특성
    :param y_train: (DataFrame) 학습 데이터 타겟
    :param X_holdout: (DataFrame) 검증 데이터 특성
    :param y_holdout: (DataFrame) 검증 데이터 타겟
    :param params_lgb: (dict) LightGBM 하이퍼파라미터
    :param params_xgb: (dict) XGBoost 하이퍼파라미터
    :param params_cat: (dict) CatBoost 하이퍼파라미터
    :param batch_num: (int) 배치 수 (기본값: 30)
    :return: (tuple) Holdout MAE for each model (LightGBM, XGBoost, CatBoost)
    """
    start_time = time.time()
    maes = {}

    # LightGBM 모델 학습 및 검증 (배치 학습)
    mae_lgb = train_lightgbm_in_batches_for_holdout(X_train, y_train, X_holdout, y_holdout, params_lgb, batch_num=batch_num)
    maes['LightGBM'] = mae_lgb
    print(f"LightGBM Holdout MAE: {maes['LightGBM']:.2f}")

    # XGBoost 모델 학습 및 검증 (배치 학습)
    mae_xgb = train_xgboost_in_batches_for_holdout(X_train, y_train, X_holdout, y_holdout, params_xgb, batch_num=batch_num)
    maes['XGBoost'] = mae_xgb
    print(f"XGBoost Holdout MAE: {maes['XGBoost']:.2f}")

    # CatBoost 모델 학습 및 검증 (배치 학습)
    mae_cat = train_catboost_in_batches_for_holdout(X_train, y_train, X_holdout, y_holdout, params_cat, batch_num=batch_num)
    maes['CatBoost'] = mae_cat
    print(f"CatBoost Holdout MAE: {mae_cat:.2f}")

    print(f"Model Training took {time.time() - start_time:.2f} seconds")	
    
    print("==============================")
    print("Model MAE Evaluation Completed")
    print("==============================")

    return maes['LightGBM'], maes['XGBoost'], maes['CatBoost']