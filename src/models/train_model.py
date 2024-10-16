import lightgbm as lgb
from numpy import ndarray
from pandas import DataFrame
import time

from src.models.lgb import optimize_lgb


def train_model(X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame, X_test: DataFrame) -> ndarray:
	"""
	Train Models
	:param X_train: (DataFrame) Feature data for training
	:param y_train: (DataFrame) Target data for training
	:param X_holdout: (DataFrame) Feature data for holdout
	:param y_holdout: (DataFrame) Target data for holdout
	:param X_test: (DataFrame) Feature data for testing
	:return:
	"""
	print("==============================")
	print('Training model')
	print("==============================")

	start_time = time.time()

	best_params = optimize_lgb(X_train, y_train, X_holdout, y_holdout, 1)

	final_lgb = lgb.LGBMRegressor(**best_params)
	final_lgb.fit(X_train, y_train)

	y_test_pred = final_lgb.predict(X_test)

	print(f"Model Training took {time.time() - start_time:.2f} seconds")

	print("==============================")
	print('Model trained')
	print("==============================")

	return y_test_pred
