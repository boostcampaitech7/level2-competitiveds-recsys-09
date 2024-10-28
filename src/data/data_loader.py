import os
import pandas as pd


def download_data():
	print('Downloading Data')

	os.makedirs('./data/raw', exist_ok=True)
	os.system(
		'wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000314/data/20240918075312/data.tar.gz')

	print('Data Downloaded')


def extract_data():
	print('Extracting Data')

	os.system('tar -xvf ./data.tar.gz --strip-components=1 -C ./data/raw/')
	os.system('rm ./data.tar.gz')

	print('Data Extracted')


def load_raw_data():
	print('Loading Raw Data')

	train_data = pd.read_csv('./data/raw/train.csv')
	test_data = pd.read_csv('./data/raw/test.csv')
	# submission = pd.read_csv('./data/raw/sample_submission.csv')
	interest_rate = pd.read_csv('./data/raw/interestRate.csv')
	subway_info = pd.read_csv('./data/raw/subwayInfo.csv')
	school_info = pd.read_csv('./data/raw/schoolinfo.csv')
	park_info = pd.read_csv('./data/raw/parkInfo.csv')

	print('Data Loaded')

	return train_data, test_data, interest_rate, subway_info, school_info, park_info


def load_preprocessed_data():
	print('Loading Preprocessed Data')

	train_data = pd.read_csv('./data/raw/train_dist.csv')
	test_data = pd.read_csv('./data/raw/test_dist.csv')
	submission = pd.read_csv('./data/raw/sample_submission.csv')

	# TODO: Load other preprocessed data
	interest_rate = pd.read_csv('./data/raw/interestRate.csv')
	subway_info = pd.read_csv('./data/raw/subwayInfo.csv')
	school_info = pd.read_csv('./data/raw/schoolinfo.csv')
	park_info = pd.read_csv('./data/raw/parkInfo.csv')

	print('Data Loaded')

	return train_data, test_data, submission, interest_rate, subway_info, school_info, park_info


def load_processed_features():
	print('Loading Processed Features')

	train_data = pd.read_csv('./data/processed_features/train_data.csv')
	test_data = pd.read_csv('./data/processed_features/test_data.csv')

	print('Data Loaded')

	return train_data, test_data