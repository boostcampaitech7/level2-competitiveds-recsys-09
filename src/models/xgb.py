import xgboost as xgb
import optuna
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error
from numpy import ndarray

from src.models.xgb_batch import train_xgboost_in_batches_for_holdout, train_xgboost_in_batches_full_data
from src.utils.variables import RANDOM_SEED


def train_xgb(X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame, params: dict, batch: bool) -> xgb.Booster:
	"""
	Train XGBoost model using the best hyperparameters
	:param X_train: (DataFrame) Feature data for training
	:param y_train: (DataFrame) Target data for training
	:param X_holdout: (DataFrame) Feature data for holdout
	:param y_holdout: (DataFrame) Target data for holdout
	:param params: (dict) Best hyperparameters
	:return: (xgb.Booster) Trained XGBoost model
	"""
	print('==============================')
	print('Training XGBoost')
	print('==============================')

	if batch:
		xgb_model = train_xgboost_in_batches_full_data(X_train, y_train, X_holdout, y_holdout, params)
	else:
		xgb_model = xgb.XGBRegressor(**params)
		xgb_model.fit(X_train, y_train)

	return xgb_model

def optimize_xgb(X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame,
				 n_trials=100, n_jobs=1, batch=True) -> (dict, ndarray):
	"""
	Optimize XGBoost hyperparameters using Optuna
	:param X_train: (DataFrame) Feature data for training
	:param y_train: (DataFrame) Target data for training
	:param X_holdout: (DataFrame) Feature data for holdout
	:param y_holdout: (DataFrame) Target data for holdout
	:param n_trials: (int) Number of trials, default=100
	:param n_jobs: (int) Number of parallel jobs, default=1
	:param batch: (bool) Use batch training, default=True
	:return: (dict) Best hyperparameters
	:return: (ndarray) Predictions
	"""
	print('==============================')
	print('XGBoost Optimization')
	print('==============================')

	study = optuna.create_study(direction='minimize')
	study.optimize(lambda trial: objective(trial, X_train, y_train, X_holdout, y_holdout, batch), n_trials=n_trials, n_jobs=n_jobs)

	print(f'Best trial: {study.best_trial.value}')
	print(f'Best params: {study.best_params}')
	
	xgb_model = train_xgb(X_train, y_train, X_holdout, y_holdout, study.best_params, batch)
	preds = xgb_model.predict(X_holdout)

	return study.best_params, preds * X_holdout['area_m2']


def objective(trial, X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame, batch: bool) -> float:
	"""
	Optuna objective function for XGBoost
	:param trial: (optuna.trial.Trial) A trial is a process of evaluating an objective function
	:param X_train: (DataFrame) Feature data for training
	:param y_train: (DataFrame) Target data for training
	:param X_holdout: (DataFrame) Feature data for holdout
	:param y_holdout: (DataFrame) Target data for holdout
	:param batch: (bool) Use batch training
	:return: (float) MAE score
	"""
	params = {
		'objective': 'reg:squarederror',
		'eval_metric': 'mae',
		'seed': RANDOM_SEED,
		'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1),
		'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
		'max_depth': trial.suggest_int('max_depth', 3, 30),
		'subsample': trial.suggest_float('subsample', 0.1, 1.0),
		'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
		'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 1e3),
		'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 1e3),
	}
	
	xgb_model = train_xgb(X_train, y_train, X_holdout, y_holdout, params, batch)
	preds = xgb_model.predict(X_holdout)
	
	y_pred_valid_origin = preds * X_holdout['area_m2']
	y_valid_origin = y_holdout * X_holdout['area_m2']
	mae = mean_absolute_error(y_valid_origin, y_pred_valid_origin)

	return mae
