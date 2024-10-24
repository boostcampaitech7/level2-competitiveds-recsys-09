import pandas as pd

from src.preprocessing.remove_data import remove_duplicates, latlng_boundary_filter
from src.preprocessing.split_data import split_year_month


def data_preprocessing(train_data, test_data, interest_rate, subway_info, school_info, park_info):
	print("==============================")
	print('Data Preprocessing')
	print("==============================")

	train_data = remove_duplicates(train_data)
	# test_data = remove_duplicates(test_data)
	# test_data = test_data.drop(columns=['index'])
	submission = test_data.copy()
	submission['deposit'] = 0
	submission = submission['deposit']
	
	train_data["age"] = train_data["age"].abs()
	test_data["age"] = test_data["age"].abs()

	#  Sort data by contract date
	# train_data = train_data.sort_values(by=['contract_year_month', 'contract_day'], ascending=[True, True])
	# test_data = test_data.sort_values(by=['contract_year_month', 'contract_day'], ascending=[True, True])

	#  Merge interest rate data
	train_data = pd.merge(train_data, interest_rate, left_on='contract_year_month', right_on='year_month', how='left')
	train_data.drop(columns=['year_month'], inplace=True)

	test_data['original_index'] = range(len(test_data))
	test_data = pd.merge(test_data, interest_rate, left_on='contract_year_month', right_on='year_month', how='left')
	test_data = test_data.sort_values('original_index').drop('original_index', axis=1)
	test_data.drop(columns=['year_month'], inplace=True)

	subway_info = latlng_boundary_filter(subway_info)
	school_info = latlng_boundary_filter(school_info)
	park_info = latlng_boundary_filter(park_info)

	print("==============================")
	print('Data Preprocessed')
	print("==============================")

	train_data.to_csv('./data/preprocessed/train.csv', index=False)
	test_data.to_csv('./data/preprocessed/test.csv', index=False)
	submission.to_csv('./data/preprocessed/submission.csv', index=False)

	return train_data, test_data, submission, interest_rate, subway_info, school_info, park_info
