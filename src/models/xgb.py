import xgboost as xgb
import optuna
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error

from src.utils.variables import RANDOM_SEED


def optimize_xgb(X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame,
				 n_trials=100, n_jobs=1) -> dict:
	"""
	Optimize XGBoost hyperparameters using Optuna
	:param X_train: (DataFrame) Feature data for training
	:param y_train: (DataFrame) Target data for training
	:param X_holdout: (DataFrame) Feature data for holdout
	:param y_holdout: (DataFrame) Target data for holdout
	:param n_trials: (int) Number of trials, default=100
	:param n_jobs: (int) Number of parallel jobs, default=1
	:return: (dict) Best hyperparameters
	"""
	print('==============================')
	print('XGBoost Optimization')
	print('==============================')

	study = optuna.create_study(direction='minimize')
	study.optimize(lambda trial: objective(trial, X_train, y_train, X_holdout, y_holdout), n_trials=n_trials, n_jobs=n_jobs)

	print(f'Best trial: {study.best_trial.value}')
	print(f'Best params: {study.best_params}')

	return study.best_params


def objective(trial, X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame) -> float:
	"""
	Optuna objective function for XGBoost
	:param trial: (optuna.trial.Trial) A trial is a process of evaluating an objective function
	:param X_train: (DataFrame) Feature data for training
	:param y_train: (DataFrame) Target data for training
	:param X_holdout: (DataFrame) Feature data for holdout
	:param y_holdout: (DataFrame) Target data for holdout
	:return: (float) MAE score
	"""
	params = {
		'objective': 'reg:squarederror',
		'eval_metric': 'mae',
		'seed': RANDOM_SEED,
		'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1),
		'n_estimators': trial.suggest_int('n_estimators', 50, 500),
		'max_depth': trial.suggest_int('max_depth', 3, 30),
		'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
		'gamma': trial.suggest_float('gamma', 0, 5),
		'subsample': trial.suggest_float('subsample', 0.1, 1.0),
		'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
	}

	xgb_model = xgb.XGBRegressor(**params)
	xgb_model.fit(X_train, y_train)

	y_pred = xgb_model.predict(X_holdout)
	mae = mean_absolute_error(y_holdout, y_pred)

	return mae
