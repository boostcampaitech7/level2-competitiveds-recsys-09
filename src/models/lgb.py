import lightgbm as lgb
import optuna
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error

from src.utils.variables import RANDOM_SEED


def optimize_lgb(X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame) -> dict:
	"""
	Optimize LightGBM hyperparameters using Optuna
	:param X_train: (DataFrame) Feature data for training
	:param y_train: (DataFrame) Target data for training
	:param X_holdout: (DataFrame) Feature data for holdout
	:param y_holdout: (DataFrame) Target data for holdout
	:return: (dict) Best hyperparameters
	"""
	study = optuna.create_study(direction='minimize')
	study.optimize(lambda trial: objective(trial, X_train, y_train, X_holdout, y_holdout))

	print(f'Best MAE: {study.best_value}')
	print(f'Best trial: {study.best_trial.value}')
	print(f'Best params: {study.best_params}')

	return study.best_params


def objective(trial, X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame) -> float:
	"""
	Optuna objective function for LightGBM
	:param trial: (optuna.trial.Trial) A trial is a process of evaluating an objective function
	:param X_train: (DataFrame) Feature data for training
	:param y_train: (DataFrame) Target data for training
	:param X_holdout: (DataFrame) Feature data for holdout
	:param y_holdout: (DataFrame) Target data for holdout
	:return: (float) MAE score
	"""
	params = {
		'objective': 'regression',
		'metric': 'mae',
		'random_state': RANDOM_SEED,
		'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1),
		'n_estimators': trial.suggest_int('n_estimators', 50, 500),
		'num_leaves': trial.suggest_int('num_leaves', 10, 150),
		'max_depth': trial.suggest_int('max_depth', 3, 20),
		'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
		'subsample': trial.suggest_float('subsample', 0.1, 1.0),
	}

	lgb_model = lgb.LGBMRegressor(**params)
	lgb_model.fit(X_train, y_train)

	y_pred = lgb_model.predict(X_holdout)
	mae = mean_absolute_error(y_holdout, y_pred)

	return mae
