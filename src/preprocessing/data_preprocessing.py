import pandas as pd

from src.preprocessing.remove_data import remove_duplicates, latlng_boundary_filter
from src.preprocessing.split_data import split_year_month


def data_preprocessing(train_data, test_data, sample_submission, interest_rate, subway_info, school_info, park_info):
	print("==============================")
	print('Data Preprocessing')
	print("==============================")
	train_data = remove_duplicates(train_data)
	test_data = remove_duplicates(test_data)
	sample_submission = remove_duplicates(sample_submission)

	#  Merge interest rate data
	train_data = pd.merge(train_data, interest_rate, left_on='contract_year_month', right_on='year_month', how='left')
	train_data.drop(columns=['year_month'], inplace=True)

	test_data = pd.merge(test_data, interest_rate, left_on='contract_year_month', right_on='year_month', how='left')
	test_data.drop(columns=['year_month'], inplace=True)

	train_data = split_year_month(train_data)
	test_data = split_year_month(test_data)

	subway_info = latlng_boundary_filter(subway_info)
	school_info = latlng_boundary_filter(school_info)
	park_info = latlng_boundary_filter(park_info)

	print("==============================")
	print('Data Preprocessed')
	print("==============================")

	train_data.to_csv('./data/preprocessed/train.csv', index=False)
	test_data.to_csv('./data/preprocessed/test.csv', index=False)
	sample_submission.to_csv('./data/preprocessed/sample_submission.csv', index=False)

	return train_data, test_data, sample_submission
