from catboost import CatBoostRegressor
import lightgbm as lgb
from numpy import ndarray
from pandas import DataFrame
import time
import xgboost as xgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

from src.models.cat import optimize_cat
from src.models.lgb import optimize_lgb
from src.models.xgb import optimize_xgb


def train_model(X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame, X_test: DataFrame,
				model: str, n_trials: int, n_jobs: int) -> ndarray:
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

	if model == 'lgb':
		best_params = optimize_lgb(X_train, y_train, X_holdout, y_holdout, n_trials=n_trials, n_jobs=n_jobs)
		final_model = lgb.LGBMRegressor(**best_params, n_jobs=n_jobs)
		final_model.fit(X_train, y_train)

	elif model == 'xgb':
		best_params = optimize_xgb(X_train, y_train, X_holdout, y_holdout, n_trials=n_trials, n_jobs=n_jobs)
		final_model = xgb.XGBRegressor(**best_params, n_jobs=n_jobs)
		final_model.fit(X_train, y_train)
		
	elif model == 'cat':
		best_params = optimize_cat(X_train, y_train, X_holdout, y_holdout, n_trials=n_trials, n_jobs=n_jobs)
		final_model = CatBoostRegressor(**best_params)
		final_model.fit(X_train, y_train)

	elif model == 'ensemble':
		lgb_model = lgb.LGBMRegressor(
			**optimize_lgb(X_train, y_train, X_holdout, y_holdout, n_trials=n_trials, n_jobs=n_jobs), n_jobs=n_jobs)
		xgb_model = xgb.XGBRegressor(
			**optimize_xgb(X_train, y_train, X_holdout, y_holdout, n_trials=n_trials, n_jobs=n_jobs), n_jobs=n_jobs)
		cat_model = CatBoostRegressor(
			**optimize_cat(X_train, y_train, X_holdout, y_holdout, n_trials=n_trials, n_jobs=n_jobs))

		# Stacking ensemble with Ridge as metamodel
		estimators = [('lgb', lgb_model), ('xgb', xgb_model), ('cat', cat_model)]
		final_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())

		final_model.fit(X_train, y_train)

	else:
		raise ValueError(f"Unsupported model type: {model}")

	y_test_pred = final_model.predict(X_test)

	print(f"Model Training took {time.time() - start_time:.2f} seconds")

	print("==============================")
	print(f'{model.upper()} Model trained')
	print("==============================")

	return y_test_pred
