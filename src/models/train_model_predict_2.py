import time
from typing import Dict, Tuple
from pandas import DataFrame
import xgboost as xgb
from catboost import CatBoostRegressor
from src.models.lgb_batch import train_lgb_in_batches_full_data
from src.models.xgb_batch import train_xgboost_in_batches_full_data
from src.models.cat import train_cat_in_batches_full_data

def train_and_predict_models_2(
    X_train: DataFrame, 
    y_train: DataFrame, 
    X_test: DataFrame,
    params_lgb: Dict, 
    params_xgb: Dict, 
    params_cat: Dict, 
    batch_num: int = 30
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    LightGBM, XGBoost, CatBoost 모델을 배치 학습하고 테스트 데이터 예측값을 반환하는 함수.

    :param X_train: (DataFrame) 학습 데이터 특성
    :param y_train: (DataFrame) 학습 데이터 타겟
    :param X_test: (DataFrame) 테스트 데이터 특성
    :param params_lgb: (dict) LightGBM 하이퍼파라미터
    :param params_xgb: (dict) XGBoost 하이퍼파라미터
    :param params_cat: (dict) CatBoost 하이퍼파라미터
    :param batch_num: (int) 배치 수 (기본값: 30)
    :return: (tuple) 각 모델의 원본 단위로 복원된 예측값 (DataFrame 형태)
    """
    start_time = time.time()

    # LightGBM 모델 학습 및 예측 (배치 학습)
    print("Training LightGBM in batches...")
    lgb_model = train_lgb_in_batches_full_data(X_train, y_train, params_lgb, batch_num=batch_num)
    lgb_test_pred = lgb_model.predict(X_test)
    lgb_test_pred_origin = lgb_test_pred * X_test['area_m2']

    # XGBoost 모델 학습 및 예측 (배치 학습)
    print("Training XGBoost in batches...")
    xgb_model = xgb.XGBRegressor(**params_xgb)
    xgb_model.fit(X_train, y_train)
    xgb_test_pred = xgb_model.predict(X_test)
    xgb_test_pred_origin = xgb_test_pred * X_test['area_m2']

    # CatBoost 모델 학습 및 예측 (배치 학습)
    print("Training CatBoost in batches...")
    cat_model = CatBoostRegressor(**params_cat)
    cat_model.fit(X_train, y_train)
    cat_test_pred = cat_model.predict(X_test)
    cat_test_pred_origin = cat_test_pred * X_test['area_m2']

    print(f"Model Training and Prediction took {time.time() - start_time:.2f} seconds")
    print("==============================")
    print("Model Training and Prediction Completed")
    print("==============================")

    return lgb_test_pred_origin, xgb_test_pred_origin, cat_test_pred_origin
