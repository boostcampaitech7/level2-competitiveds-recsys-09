import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error
from pandas import DataFrame
from numpy import ndarray

from src.utils.variables import RANDOM_SEED

def train_catboost(X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame, params: dict, batch: bool) -> CatBoostRegressor:
    """
    Train CatBoost model using the best hyperparameters
    :param X_train: (DataFrame) Feature data for training
    :param y_train: (DataFrame) Target data for training
    :param X_holdout: (DataFrame) Feature data for holdout
    :param y_holdout: (DataFrame) Target data for holdout
    :param params: (dict) Best hyperparameters
    :return: (CatBoostRegressor) Trained CatBoost model
    """
    print('==============================')
    print('Training CatBoost')
    print('==============================')
    
    if batch:
        cat_model = train_cat_in_batches_full_data(X_train, y_train, params)
    else:
        cat_model = CatBoostRegressor(**params)
        cat_model.fit(X_train, y_train, eval_set=(X_holdout, y_holdout), verbose=False)
    
    return cat_model

'''
optimize_catboost: Optuna를 사용하여 CatBoost 모델의 최적 하이퍼파라미터를 찾는 함수입니다.
objective_catboost: CatBoost의 Optuna 최적화 목적 함수로, MAE를 기준으로 평가합니다.
train_catboost_in_batches_for_holdout: 배치 학습을 통해 CatBoost 모델을 학습하고 검증 데이터의 MAE를 평가하는 함수입니다.
train_final_model: 최적의 하이퍼파라미터로 전체 데이터를 사용해 CatBoost 모델을 최종 학습하는 함수입니다.
'''
def optimize_catboost(
    X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame, n_trials=50, n_jobs=1, batch=True
) -> (dict, ndarray):
    """
    Optuna를 사용하여 CatBoost 하이퍼파라미터를 최적화하는 함수
    :param X_train: (DataFrame) 학습용 특성 데이터
    :param y_train: (DataFrame) 학습용 타겟 데이터
    :param X_holdout: (DataFrame) 검증용 특성 데이터
    :param y_holdout: (DataFrame) 검증용 타겟 데이터
    :param n_trials: (int) 최적화 시도 횟수, 기본값=50
    :param n_jobs: (int) 병렬 작업 수, 기본값=1
    :param batch: (bool) 배치 학습 사용 여부, 기본값=True
    :return: (dict) 최적의 하이퍼파라미터
    """
    print('==============================')
    print('CatBoost Optimization')
    print('==============================')
    
    # Optuna 스터디 생성 및 최적화 수행
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_catboost(trial, X_train, y_train, X_holdout, y_holdout, batch), 
                   n_trials=n_trials, n_jobs=n_jobs)

    print(f'Best trial: {study.best_trial.value}')
    print(f'Best params: {study.best_params}')
    
    catboost_model = train_catboost(X_train, y_train, X_holdout, y_holdout, study.best_params, batch)
    preds = catboost_model.predict(X_holdout)
    
    return study.best_params, preds * X_holdout['area_m2']


def objective_catboost(
    trial, X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame, batch: bool
) -> float:
    """
    CatBoost의 Optuna 최적화 목적 함수
    :param trial: (optuna.trial.Trial) Optuna의 단일 실험 객체
    :param X_train: (DataFrame) 학습용 특성 데이터
    :param y_train: (DataFrame) 학습용 타겟 데이터
    :param X_holdout: (DataFrame) 검증용 특성 데이터
    :param y_holdout: (DataFrame) 검증용 타겟 데이터
    :param batch: (bool) 배치 학습 사용 여부
    :return: (float) MAE 평가 지표
    """
    # 최적화할 하이퍼파라미터 정의
    params = {
        'random_seed': 42,
        'loss_function': 'MAE',
        'verbose': False,
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100.0, log=True),
        'random_strength': trial.suggest_int('random_strength', 1, 20),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 1e-8, 100.0, log=True),
        'border_count': trial.suggest_int('border_count', 1, 255),
    }
    
    catboost_model = train_catboost(X_train, y_train, X_holdout, y_holdout, params, batch)
    preds = catboost_model.predict(X_holdout)
    
    y_pred_valid_origin = preds * X_holdout['area_m2']
    y_valid_origin = y_holdout * X_holdout['area_m2']
    mae = mean_absolute_error(y_valid_origin, y_pred_valid_origin)
    
    return mae



def train_catboost_in_batches_for_holdout(
    X_train: DataFrame, 
    y_train: DataFrame, 
    X_valid: DataFrame, 
    y_valid: DataFrame, 
    params: dict, 
    batch_num: int = 30
) -> tuple[CatBoostRegressor, float]:
    """
    배치 학습을 사용하여 CatBoost 모델을 학습하고 검증 MAE를 계산하는 함수

    :param X_train: (DataFrame) 학습 데이터 특성
    :param y_train: (DataFrame) 학습 데이터 타겟
    :param X_valid: (DataFrame) 검증 데이터 특성
    :param y_valid: (DataFrame) 검증 데이터 타겟
    :param params: (dict) CatBoost 하이퍼파라미터 설정 딕셔너리
    :param batch_num: (int) 배치 수 (기본값: 30)
    :return: (CatBoostRegressor, float) 학습이 완료된 CatBoost 모델과 검증 MAE
    """
    
    # CatBoost 모델 초기화
    cat_model = CatBoostRegressor(**params)
    
    # 배치 크기 계산
    batch_size = len(X_train) // batch_num

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # CatBoost Pool 생성
        train_pool = Pool(X_batch, y_batch)
        
        if i == 0:
            # 첫 번째 배치에서 모델 초기화 및 학습
            cat_model.fit(train_pool)
        else:
            # 이후 배치에서는 이전 모델에 이어서 학습 (continue learning)
            cat_model.fit(train_pool, init_model=cat_model)
        
        print(f"Batch {i // batch_size + 1}/{batch_num} 학습 완료")

    # 검증 데이터로 MAE 평가
    valid_pool = Pool(X_valid, y_valid)
    preds = cat_model.predict(valid_pool)
    
     # 면적 스케일로 원복한 예측값 및 실제값
    y_pred_valid_origin = preds * X_valid['area_m2']
    y_valid_origin = y_valid * X_valid['area_m2']
    
     # 원복한 값으로 MAE 계산
    mae = mean_absolute_error(y_valid_origin, y_pred_valid_origin)
    
    print(f"Validation MAE: {mae}")
    return mae


def train_cat_in_batches_full_data(
    X_train_full: DataFrame, 
    y_train_full: DataFrame, 
    params: dict, 
    batch_num: int = 30
) -> CatBoostRegressor:
    """
    배치 학습을 사용하여 전체 데이터를 학습하는 CatBoost 모델 최종 학습 함수

    :param X_train_full: (DataFrame) 전체 학습 데이터의 특성
    :param y_train_full: (DataFrame) 전체 학습 데이터의 타겟
    :param params: (dict) CatBoost 하이퍼파라미터 설정 딕셔너리
    :param batch_num: (int) 배치 수 (기본값: 30)
    :return: (CatBoostRegressor) 학습이 완료된 CatBoost 모델
    """
    
    # CatBoost 모델 초기화
    cat_model = CatBoostRegressor(**params)
    
    # 배치 크기 계산
    batch_size = len(X_train_full) // batch_num

    for i in range(0, len(X_train_full), batch_size):
        X_batch = X_train_full[i:i + batch_size]
        y_batch = y_train_full[i:i + batch_size]

        # CatBoost Pool 생성
        train_pool = Pool(X_batch, y_batch)
        
        if i == 0:
            # 첫 번째 배치에서 모델 초기화 및 학습
            cat_model.fit(train_pool)
        else:
            # 이후 배치에서는 이전 모델에 이어서 학습 (continue learning)
            cat_model.fit(train_pool, init_model=cat_model)
        
        print(f"Batch {i // batch_size + 1}/{batch_num} 학습 완료")

    print("최종 모델 학습 완료")
    return cat_model