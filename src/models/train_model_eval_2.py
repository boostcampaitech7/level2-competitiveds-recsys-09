import lightgbm as lgb
from numpy import ndarray
from pandas import DataFrame
import time
import xgboost as xgb

from models.cat import optimize_catboost
from src.models.lgb import optimize_lgb
from src.models.xgb import optimize_xgb


def train_and_evaluate_models_2(X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame, X_test: DataFrame,
				n_trials: int, n_jobs: int) -> (lgb.Booster, xgb.Booster, cat.CatBoostRegressor, ndarray, ndarray, ndarray):
	"""
	Train Models
	:param X_train: (DataFrame) Feature data for training
	:param y_train: (DataFrame) Target data for training
	:param X_holdout: (DataFrame) Feature data for holdout
	:param y_holdout: (DataFrame) Target data for holdout
	:param X_test: (DataFrame) Feature data for testing
	:param model: (str) Model to train
	:param n_trials: (int) Number of trials
	:param n_jobs: (int) Number of CPU threads
	:return:
	"""
	print("==============================")
	print('Training model')
	print("==============================")

	start_time = time.time()

    # 모델 hyperparameter 최적화
	lgb_best_params, lgb_preds = optimize_lgb(X_train, y_train, X_holdout, y_holdout, n_trials=n_trials, n_jobs=n_jobs)
	xgb_best_params, xgb_preds = optimize_xgb(X_train, y_train, X_holdout, y_holdout, n_trials=n_trials, n_jobs=n_jobs, batch=False)
	cb_best_params, cb_preds = optimize_catboost(X_train, y_train, X_holdout, y_holdout, n_trials=n_trials, n_jobs=n_jobs, batch=False)
    

	print(f"Model Training took {time.time() - start_time:.2f} seconds")

	print("==============================")
	print(f'Model trained and evaluated')
	print("==============================")

	return lgb_best_params, xgb_best_params, cb_best_params, lgb_preds, xgb_preds, cb_preds