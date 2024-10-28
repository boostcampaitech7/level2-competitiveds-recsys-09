import argparse
import numpy as np

from models.SLSQP import optimize_weights
from models.train_model_eval_2 import train_and_evaluate_models_2
from src.data.data_loader import download_data, extract_data, load_raw_data, load_preprocessed_data, \
	load_processed_features
from src.evaluation.holdout import get_holdout_data
from src.features.feature_engineering2 import feature_engineering2
from src.models.train_model import train_model
from src.preprocessing.data_preprocessing import data_preprocessing
from src.utils.submission import submission_to_csv
from src.utils.variables import RANDOM_SEED

np.random.seed(RANDOM_SEED)


def main():
	parser = argparse.ArgumentParser(description='Run project with command line arguments')
	parser.add_argument('--force-reprocess', action='store_true', help='Force reprocess data')
	parser.add_argument('--n-trials', type=int, default=100,
						help='Number of trials for hyperparameter optimization (default=100)')
	parser.add_argument('--n-jobs', type=int, default=1,
						help='Number of CPU threads for hyperparameter optimization (default=1)')
	parser.add_argument('--model', type=str, default='lgb', choices=['lgb', 'xgb'], help='Model to train (default=lgb)')

	args = parser.parse_args()

	# TODO: !!!REFACTOR!!!
	if args.force_reprocess:
		print("==============================")
		print('Force reprocessing data')
		print("==============================")
		download_data()
		extract_data()

		train_data, test_data, interest_rate, subway_info, school_info, park_info = load_raw_data()
		train_data, test_data, submission, interest_rate, subway_info, school_info, park_info = data_preprocessing(
			train_data, test_data, interest_rate, subway_info, school_info, park_info)
		train_data, test_data = feature_engineering2(train_data, test_data, interest_rate, subway_info,
													school_info, park_info)
	else:
		try:
			train_data, test_data = load_processed_features()
			_, _, submission, _, _, _, _ = load_preprocessed_data()
		except FileNotFoundError:
			try:
				train_data, test_data, submission, interest_rate, subway_info, school_info, park_info = load_preprocessed_data()
				train_data, test_data = feature_engineering2(train_data, test_data, interest_rate, subway_info,
															school_info,
															park_info)
			except FileNotFoundError:
				try:
					train_data, test_data, interest_rate, subway_info, school_info, park_info = load_raw_data()
					train_data, test_data, submission, interest_rate, subway_info, school_info, park_info = data_preprocessing(
						train_data, test_data, interest_rate, subway_info, school_info, park_info)
					train_data, test_data = feature_engineering2(train_data, test_data, interest_rate, subway_info,
																school_info, park_info)
				except FileNotFoundError:
					print("==============================")
					print('Data not found. Downloading...')
					print("==============================")
					download_data()
					extract_data()

					train_data, test_data, interest_rate, subway_info, school_info, park_info = load_raw_data()
					train_data, test_data, submission, interest_rate, subway_info, school_info, park_info = data_preprocessing(
						train_data, test_data, interest_rate, subway_info, school_info, park_info)
					train_data, test_data = feature_engineering2(train_data, test_data, interest_rate, subway_info,
																school_info, park_info)

	holdout_data, without_holdout_data = get_holdout_data(train_data)

	X_train = without_holdout_data.drop(columns=['deposit_per_area', 'year', 'month'])
	y_train = without_holdout_data['deposit_per_area']
	X_holdout = holdout_data.drop(columns=['deposit_per_area', 'year', 'month'])
	y_holdout = holdout_data['deposit_per_area']
	X_test = test_data.copy()

    # holdout 데이터로 모델 학습 및 예측
	lgb_best_params, xgb_best_params, cb_best_params, lgb_preds, xgb_preds, cb_preds = train_and_evaluate_models_2(X_train, y_train, X_holdout, y_holdout, X_test, args.model, args.n_trials, args.n_jobs)
	
    # 전체 데이터로 모델 학습 및 예측
	lgb_test_pred_origin, xgb_test_pred_origin, cat_test_pred_origin = train_and_evaluate_models_2(X_train, y_train, X_test, lgb_best_params, xgb_best_params, cb_best_params)
	
    # 앙상블 진행
	ensemble_weight, fun = optimize_weights(X_holdout, y_holdout, lgb_preds, xgb_preds, cb_preds)
	y_test_pred = ensemble_weight[0] * lgb_test_pred_origin + ensemble_weight[1] * xgb_test_pred_origin + ensemble_weight[2] * cat_test_pred_origin
	
	submission_to_csv(submission, y_test_pred)


if __name__ == '__main__':
	main()
