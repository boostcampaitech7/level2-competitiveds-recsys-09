import lightgbm as lgb
from numpy import ndarray
from pandas import DataFrame
import pandas as pd
import time
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from src.models.lgb import optimize_lgb
from src.models.xgb import optimize_xgb
from catboost import CatBoostRegressor


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
	

	params_lgb = {
			'learning_rate': 0.09979759062977589,
			'n_estimators': 436,
			'num_leaves': 134,
			'max_depth': 20,
			'min_child_samples': 22,
			'subsample': 0.7265148802823872
		}
	
	params_xgb = {
			'learning_rate': 0.020523230971679813,
			'n_estimators': 313,
			'max_depth': 20,
			'min_child_weight': 10,
			'gamma': 4.27443753117609,
			'subsample': 0.49335787451194985,
			'colsample_bytree': 0.8502375640374019
		}
	
	if model == 'lgb':
		final_model = lgb.LGBMRegressor(**params_lgb)
		final_model.fit(X_train, y_train, eval_set=[(X_holdout, y_holdout)])

	elif model == 'xgb':
		print("Training XGB...")
		final_model = xgb.XGBRegressor()
		batch_size = 1000
		for i in range(0, len(X_train), batch_size):
			if i % 10000 == 0:
				print(f"Training batch {i} to {i + batch_size}")
			end = i + batch_size if i + batch_size < len(X_train) else len(X_train)
			final_model.fit(X_train.iloc[i:end], y_train.iloc[i:end], xgb_model=final_model.get_booster() if i > 0 else None)

	elif model == 'final':
		batch_size = 1000
		all_final_model = xgb.XGBRegressor(**params_xgb)
		all_features = pd.concat([X_train, X_holdout], axis=0)
		all_target = pd.concat([y_train, y_holdout], axis=0)
		
		for i in range(0, len(all_features), batch_size):
			end = i + batch_size if i + batch_size < len(all_features) else len(all_features)
			all_final_model.fit(all_features.iloc[i:end], all_target.iloc[i:end], init_model=all_final_model if i > 0 else None)

		y_test_pred = final_model.predict(X_test)
		y_test_pred_final = y_test_pred * X_test['area_m2']

		print(f"Model Training took {time.time() - start_time:.2f} seconds")

		print("==============================")
		print(f'{model.upper()} Model trained')
		print("==============================")

		return y_test_pred_final
		
	else:
		raise ValueError(f"Unsupported model type: {model}")

	y_pred = final_model.predict(X_holdout)
	y_pred_final = y_pred * X_holdout['area_m2']
	mae = mean_absolute_error(y_holdout * X_holdout['area_m2'], y_pred_final)
	print(mae)

	# Multiply predictions by area_m2
	if 'index' in X_test.columns:
		X_test = X_test.drop(columns=['index'])
	y_test_pred = final_model.predict(X_test)
	y_test_pred_final = y_test_pred * X_test['area_m2']

	print(f"Model Training took {time.time() - start_time:.2f} seconds")

	print("==============================")
	print(f'{model.upper()} Model trained')
	print("==============================")

	return y_test_pred_final
