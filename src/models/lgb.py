import lightgbm as lgb
import optuna
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error
from numpy import ndarray

from src.models.lgb_batch import train_lgb_in_batches_full_data, train_lightgbm_in_batches_for_holdout
from src.utils.variables import RANDOM_SEED

def train_lgb(X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame, params: dict, batch: bool) -> lgb.Booster:
    """
    Train LightGBM model using the best hyperparameters
    :param X_train: (DataFrame) Feature data for training
    :param y_train: (DataFrame) Target data for training
    :param X_holdout: (DataFrame) Feature data for holdout
    :param y_holdout: (DataFrame) Target data for holdout
    :param params: (dict) Best hyperparameters
    :return: (lgb.Booster) Trained LightGBM model
    """
    print('==============================')
    print('Training LightGBM')
    print('==============================')
    
    if batch:
        lgb_model = train_lgb_in_batches_full_data(X_train, y_train, X_holdout, y_holdout, params)
    else:
        lgb_model = lgb.LGBMRegressor(**params)
        lgb_model.fit(X_train, y_train)
		
    return lgb_model

def optimize_lgb(X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame, n_trials=100,
				 n_jobs=1, batch=True) -> (dict, ndarray):
	"""
	Optimize LightGBM hyperparameters using Optuna
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
	print('LightGBM Optimization')
	print('==============================')
	study = optuna.create_study(direction='minimize')
	study.optimize(lambda trial: objective(trial, X_train, y_train, X_holdout, y_holdout, batch), n_trials=n_trials,
				   n_jobs=n_jobs)

	print(f'Best trial: {study.best_trial.value}')
	print(f'Best params: {study.best_params}')
	
	lgb_model = train_lgb(X_train, y_train, X_holdout, y_holdout, study.best_params, batch)
	preds = lgb_model.predict(X_holdout, num_iteration=lgb_model.best_iteration)

	return study.best_params, preds * X_holdout['area_m2']


def objective(trial, X_train: DataFrame, y_train: DataFrame, X_holdout: DataFrame, y_holdout: DataFrame, batch: bool) -> float:
	"""
	Optuna objective function for LightGBM
	:param trial: (optuna.trial.Trial) A trial is a process of evaluating an objective function
	:param X_train: (DataFrame) Feature data for training
	:param y_train: (DataFrame) Target data for training
	:param X_holdout: (DataFrame) Feature data for holdout
	:param y_holdout: (DataFrame) Target data for holdout
	:param batch: (bool) Use batch training
	:return: (float) MAE score
	"""
	params = {
		'objective': 'regression',
		'metric': 'mae',
		'random_state': RANDOM_SEED,
		'num_leaves': trial.suggest_int('num_leaves', 2, 256),
		'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
		'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
		'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
		'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
		'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
	}
	
	lgb_model = train_lgb(X_train, y_train, X_holdout, y_holdout, params, batch)
	preds = lgb_model.predict(X_holdout, num_iteration=lgb_model.best_iteration)
	
	y_pred_valid_origin = preds * X_holdout['area_m2']
	y_valid_origin = y_holdout * X_holdout['area_m2']
	mae = mean_absolute_error(y_valid_origin, y_pred_valid_origin)


	return mae

