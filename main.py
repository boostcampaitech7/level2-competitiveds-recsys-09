import numpy as np
from src.data.data_loader import download_data, extract_data, load_raw_data
from src.evaluation.holdout import get_holdout_data
from src.features.feature_engineering import feature_engineering
from src.models.train_model import train_model
from src.preprocessing.data_preprocessing import data_preprocessing
from src.utils.submission import submission_to_csv
from src.utils.variables import RANDOM_SEED

np.random.seed(RANDOM_SEED)


def main():
	try:
		train_data, test_data, sample_submission, interest_rate, subway_info, school_info, park_info = load_raw_data()
	except FileNotFoundError:
		print("==============================")
		print('Data not found. Downloading...')
		print("==============================")
		download_data()
		extract_data()

		train_data, test_data, sample_submission, interest_rate, subway_info, school_info, park_info = load_raw_data()

	train_data, test_data = data_preprocessing(train_data, test_data, interest_rate, subway_info, school_info, park_info)
	train_data, test_data = feature_engineering(train_data, test_data)

	holdout_data = get_holdout_data(train_data)

	X_train = train_data.drop(columns=['deposit'])
	y_train = train_data['deposit']
	X_holdout = holdout_data.drop(columns=['deposit'])
	y_holdout = holdout_data['deposit']
	X_test = test_data.copy()

	y_test_pred = train_model(X_train, y_train, X_holdout, y_holdout, X_test)

	submission_to_csv(sample_submission, y_test_pred)


if __name__ == '__main__':
	main()
