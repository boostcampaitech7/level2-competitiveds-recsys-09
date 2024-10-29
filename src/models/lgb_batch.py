import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from pandas import DataFrame
from typing import Tuple

def train_lightgbm_in_batches_for_holdout(
    X_train: DataFrame, 
    y_train: DataFrame, 
    X_holdout: DataFrame, 
    y_holdout: DataFrame, 
    params: dict, 
    batch_num: int = 30,
    num_boost_round: int = 500
) -> Tuple[lgb.Booster, float]:
    """
    배치 학습을 사용하여 LightGBM 모델을 학습하고 Holdout 데이터셋의 MAE를 계산하는 함수

    :param X_train: (DataFrame) 학습 데이터 특성
    :param y_train: (DataFrame) 학습 데이터 타겟
    :param X_holdout: (DataFrame) 검증 데이터 특성
    :param y_holdout: (DataFrame) 검증 데이터 타겟
    :param params: (dict) LightGBM 하이퍼파라미터 설정 딕셔너리
    :param batch_num: (int) 배치 수 (기본값: 30)
    :param num_boost_round: (int) 부스팅 라운드 수 (기본값: 500)
    :return: (lgb.Booster, float) 학습이 완료된 LightGBM 모델과 Holdout MAE
    """
    
    batch_size = len(X_train) // batch_num
    lgb_model = None

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # LightGBM Dataset 생성
        train_data = lgb.Dataset(X_batch, label=y_batch, free_raw_data=False)
        valid_data = lgb.Dataset(X_holdout, label=y_holdout, reference=train_data, free_raw_data=False)

        # 모델 학습
        lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round, 
            init_model=lgb_model if i != 0 else None  # 이전 배치 모델 이어서 학습
        )
        
        # 배치 완료 출력
        print(f"Batch {i // batch_size + 1}/{batch_num} 학습 완료")

    # Holdout 데이터에 대한 MAE 계산
    preds = lgb_model.predict(X_holdout, num_iteration=lgb_model.best_iteration)
    
    # 면적 스케일로 원복한 예측값 및 실제값
    y_pred_valid_origin = preds * X_holdout['area_m2']
    y_valid_origin = y_holdout * X_holdout['area_m2']
    
     # 원복한 값으로 MAE 계산
    mae = mean_absolute_error(y_valid_origin, y_pred_valid_origin)
    
    print(f"Holdout MAE: {mae:.2f}")
    return mae


def train_lgb_in_batches_full_data(
    X_train_full: DataFrame, 
    y_train_full: DataFrame, 
    params: dict, 
    batch_num: int = 30,
    num_boost_round: int = 10
) -> lgb.Booster:
    """
    전체 데이터를 사용하여 LightGBM 모델을 배치 학습으로 최종 학습하는 함수

    :param X_train_full: (DataFrame) 전체 학습 데이터 특성
    :param y_train_full: (DataFrame) 전체 학습 데이터 타겟
    :param params: (dict) LightGBM 하이퍼파라미터 설정 딕셔너리
    :param batch_num: (int) 배치 수 (기본값: 30)
    :param num_boost_round: (int) 부스팅 라운드 수 (기본값: 500)
    :return: (lgb.Booster) 학습이 완료된 최종 LightGBM 모델
    """
    
    batch_size = len(X_train_full) // batch_num
    lgb_model = None

    for i in range(0, len(X_train_full), batch_size):
        X_batch = X_train_full[i:i + batch_size]
        y_batch = y_train_full[i:i + batch_size]

        # LightGBM Dataset 생성
        train_data = lgb.Dataset(X_batch, label=y_batch, free_raw_data=False)

        # 모델 학습
        lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round, 
            init_model=lgb_model if i != 0 else None  # 이전 배치 모델 이어서 학습
        )
        
        # 배치 완료 출력
        print(f"Batch {i // batch_size + 1}/{batch_num} 학습 완료")

    print("전체 데이터 학습 완료")
    return lgb_model