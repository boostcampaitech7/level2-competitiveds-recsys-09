import numpy as np
from src.data.data_loader import download_data, extract_data, load_raw_data
from src.utils.variables import RANDOM_SEED


np.random.seed(RANDOM_SEED)


def main():
	try:
		train_data, test_data, submission, interest_rate, subway_info, school_info, park_info = load_raw_data()
	except FileNotFoundError:
		print('Data not found. Downloading...')
		download_data()
		extract_data()

		train_data, test_data, submission, interest_rate, subway_info, school_info, park_info = load_raw_data()

	X_train = train_data.drop(columns=['deposit'])
	y_train = train_data['deposit']

