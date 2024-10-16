import os
import pandas as pd


def download_data():
	print('Downloading Data')

	os.makedirs('/data/raw', exist_ok=True)
	os.system(
		'wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000314/data/20240918075312/data.tar.gz')

	print('Data Downloaded')


def extract_data():
	print('Extracting Data')

	os.system('tar -xvf data/raw/data.tar.gz -C data/raw/')

	print('Data Extracted')


def load_raw_data():
	print('Loading Raw Data')

	train_data = pd.read_csv('/data/raw/train.csv')
	test_data = pd.read_csv('/data/raw/test.csv')
	submission = pd.read_csv('/data/raw/sample_submission.csv')
	interest_rate = pd.read_csv('/data/raw/interestRate.csv')
	subway_info = pd.read_csv('/data/raw/subwayInfo.csv')
	school_info = pd.read_csv('/data/raw/schoolinfo.csv')
	park_info = pd.read_csv('/data/raw/parkInfo.csv')

	print('Data Loaded')

	return train_data, test_data, submission, interest_rate, subway_info, school_info, park_info
