import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from pandas import DataFrame
from typing import Tuple


def train_xgboost_in_batches_for_holdout(
    X_train: DataFrame, 
    y_train: DataFrame, 
    X_holdout: DataFrame, 
    y_holdout: DataFrame, 
    params: dict, 
    batch_num: int = 30
) -> float:
    """
    Holdout 데이터로 XGBoost 모델을 배치 학습하고 MAE를 계산하는 함수

    :param X_train: (DataFrame) 학습 데이터 특성
    :param y_train: (DataFrame) 학습 데이터 타겟
    :param X_holdout: (DataFrame) 검증 데이터 특성
    :param y_holdout: (DataFrame) 검증 데이터 타겟
    :param params: (dict) XGBoost 하이퍼파라미터 설정 딕셔너리
    :param batch_num: (int) 배치 수 (기본값: 30)
    :return: (float) Holdout 데이터에 대한 MAE
    """
    
    xgb_model = None
    batch_size = len(X_train) // batch_num

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        dtrain_batch = xgb.DMatrix(X_batch, label=y_batch)
        xgb_model = xgb.train(params, dtrain_batch, num_boost_round=10, xgb_model=xgb_model)
        print(f"Batch {i // batch_size + 1}/{batch_num} 학습 완료")

    # Holdout 데이터 예측 및 MAE 계산
    dholdout = xgb.DMatrix(X_holdout)
    y_pred_holdout = xgb_model.predict(dholdout)
    y_pred_holdout_origin = y_pred_holdout * X_holdout['area_m2']
    y_holdout_origin = y_holdout * X_holdout['area_m2']
    holdout_mae = mean_absolute_error(y_holdout_origin, y_pred_holdout_origin)
    
    print(f"Holdout MAE: {holdout_mae:.2f}")
    return holdout_mae


def train_xgboost_in_batches_full_data(
    X_train_full: DataFrame, 
    y_train_full: DataFrame, 
    params: dict, 
    batch_num: int = 30
) -> xgb.Booster:
    """
    전체 데이터를 사용하여 XGBoost 모델을 배치 학습하고 최종 모델을 반환하는 함수

    :param X_train_full: (DataFrame) 전체 학습 데이터 특성
    :param y_train_full: (DataFrame) 전체 학습 데이터 타겟
    :param params: (dict) XGBoost 하이퍼파라미터 설정 딕셔너리
    :param batch_num: (int) 배치 수 (기본값: 30)
    :return: (xgb.Booster) 학습이 완료된 XGBoost 모델
    """
    
    xgb_model = None
    batch_size = len(X_train_full) // batch_num

    for i in range(0, len(X_train_full), batch_size):
        X_batch = X_train_full[i:i + batch_size]
        y_batch = y_train_full[i:i + batch_size]
        dtrain_batch = xgb.DMatrix(X_batch, label=y_batch)
        xgb_model = xgb.train(params, dtrain_batch, num_boost_round=10, xgb_model=xgb_model)
        print(f"Batch {i // batch_size + 1}/{batch_num} 학습 완료")

    print("전체 데이터 학습 완료")
    return xgb_model