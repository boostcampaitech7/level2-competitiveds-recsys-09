import lightgbm as lgb
from numpy import ndarray
from pandas import DataFrame
import time
import xgboost as xgb

from src.models.lgb import optimize_lgb
from src.models.xgb import optimize_xgb


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

	# LightGBM
	###
	# best_params = optimize_lgb(X_train, y_train, X_holdout, y_holdout, 1)
	#
	# final_lgb = lgb.LGBMRegressor(**best_params)
	# final_lgb.fit(X_train, y_train)
	#
	# y_test_pred = final_lgb.predict(X_test)
	###

	##################################################
	# 					WARNING						 #
	#   n_jobs=-1 will use all available CPU cores   #
	##################################################
	best_params = optimize_xgb(X_train, y_train, X_holdout, y_holdout, 1, n_jobs=-1)

	final_xgb = xgb.XGBRegressor(**best_params)
	final_xgb.fit(X_train, y_train)

	y_test_pred = final_xgb.predict(X_test)

	print(f"Model Training took {time.time() - start_time:.2f} seconds")

	print("==============================")
	print('Model trained')
	print("==============================")

	return y_test_pred
