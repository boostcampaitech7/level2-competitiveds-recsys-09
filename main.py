import argparse
import numpy as np

from src.data.data_loader import download_data, extract_data, load_raw_data, load_preprocessed_data, \
	load_processed_features
from src.evaluation.holdout import get_holdout_data
from src.features.feature_engineering import feature_engineering
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
	parser.add_argument('--model', type=str, default='lgb', choices=['lgb', 'xgb', 'cat'], help='Model to train (default=lgb)')
	parser.add_argument('--submission', type=str, default='output.csv', help='Submission file name (default=output.csv)')

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
		train_data, test_data = feature_engineering(train_data, test_data, interest_rate, subway_info,
													school_info, park_info)
	else:
		try:
			train_data, test_data = load_processed_features()
			_, _, submission, _, _, _, _ = load_preprocessed_data()
		except FileNotFoundError:
			try:
				train_data, test_data, submission, interest_rate, subway_info, school_info, park_info = load_preprocessed_data()
				train_data, test_data = feature_engineering(train_data, test_data, interest_rate, subway_info,
															school_info,
															park_info)
			except FileNotFoundError:
				try:
					train_data, test_data, interest_rate, subway_info, school_info, park_info = load_raw_data()
					train_data, test_data, submission, interest_rate, subway_info, school_info, park_info = data_preprocessing(
						train_data, test_data, interest_rate, subway_info, school_info, park_info)
					train_data, test_data = feature_engineering(train_data, test_data, interest_rate, subway_info,
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
					train_data, test_data = feature_engineering(train_data, test_data, interest_rate, subway_info,
																school_info, park_info)

	holdout_data, train_data = get_holdout_data(train_data)

	X_train = train_data.drop(columns=['deposit'])
	y_train = train_data['deposit']
	X_holdout = holdout_data.drop(columns=['deposit'])
	y_holdout = holdout_data['deposit']
	X_test = test_data.copy()

	y_test_pred = train_model(X_train, y_train, X_holdout, y_holdout, X_test, args.model, args.n_trials, args.n_jobs)

	submission_to_csv(submission, y_test_pred, args.submission)


if __name__ == '__main__':
	main()
