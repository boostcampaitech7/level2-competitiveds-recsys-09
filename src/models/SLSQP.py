from scipy.optimize import minimize
import numpy as np
from pandas import DataFrame
from typing import List, Tuple

def optimize_weights(
    y_holdout: np.ndarray,
    X_holdout: DataFrame,
    xgb_preds: np.ndarray,
    lgb_preds: np.ndarray,
    cat_preds: np.ndarray,
    initial_weights: List[float] = [0.33, 0.33, 0.34]
) -> Tuple[np.ndarray, float]:
    """
    모델 예측값의 가중치를 최적화하여 MAE를 최소화하는 함수.
    holdout 예측값으로 최적화 진행 
    
    
    :param y_holdout: 검증 데이터셋의 실제 타겟 값
    :param X_holdout: 검증 데이터셋의 특성 데이터
    :param xgb_preds: XGBoost 예측값
    :param lgb_preds: LightGBM 예측값
    :param cat_preds: CatBoost 예측값
    :param initial_weights: 초기 가중치 리스트, 기본값은 [0.33, 0.33, 0.34]
    :return: (최적의 가중치, 최소화된 MAE)
    """
    
    # 실제 값 원본 스케일로 복원
    true_values = y_holdout * X_holdout['area_m2'].values

    # 손실 함수 정의: MAE (Mean Absolute Error)를 최소화
    def objective(weights):
        weighted_prediction = (weights[0] * xgb_preds +
                               weights[1] * lgb_preds +
                               weights[2] * cat_preds)
        return np.mean(np.abs(weighted_prediction - true_values))

    # 제약 조건: 가중치의 합은 1이어야 함
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # 가중치의 범위는 0과 1 사이
    bounds = [(0, 1) for _ in range(3)]

    # 최적화 수행
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # 결과 반환
    return result.x, result.fun