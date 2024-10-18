import pandas as pd

from src.preprocessing.remove_data import remove_duplicates, latlng_boundary_filter, remove_quantile
from src.preprocessing.split_data import split_year_month


def data_preprocessing(train_data, test_data, interest_rate, subway_info, school_info, park_info):
	print("==============================")
	print('Data Preprocessing')
	print("==============================")

	train_data = remove_duplicates(train_data)

	# TODO: Remove Duplicates
	# test_data = remove_duplicates(test_data)
	test_data = test_data.drop(columns=['index'])
	submission = test_data.copy()
	submission['deposit'] = 0
	submission = submission['deposit']

	#  Merge interest rate data
	train_data = pd.merge(train_data, interest_rate, left_on='contract_year_month', right_on='year_month', how='left')
	train_data.drop(columns=['year_month'], inplace=True)

	test_data = pd.merge(test_data, interest_rate, left_on='contract_year_month', right_on='year_month', how='left')
	test_data.drop(columns=['year_month'], inplace=True)

	
	train_data = split_year_month(train_data)
	test_data = split_year_month(test_data)
 
	# discard park data that less than 1,3 quantile
	park_info = remove_quantile(park_info, 'area', 0.25, 0.75)

	# 자르지 않는 것이 좋을 수도 있음
	'''	
 	subway_info = latlng_boundary_filter(subway_info)
	school_info = latlng_boundary_filter(school_info)
	park_info = latlng_boundary_filter(park_info)
	'''
	# divide school_info by schoolLevel (elementary_info, middle_info, high_info) 
	school_info['schoolLevel'] = school_info['schoolLevel'].str.lower()
	elementary_info = school_info[school_info['schoolLevel'] == 'elementary']
	middle_info = school_info[school_info['schoolLevel'] == 'middle']
	high_info = school_info[school_info['schoolLevel'] == 'high']
	
	# reset index 
	elementary_info = elementary_info.reset_index(drop=True)
	middle_info = middle_info.reset_index(drop=True)
	high_info = high_info.reset_index(drop=True)

	elementary_info['ID'] = elementary_info.index
	middle_info['ID'] = middle_info.index
	high_info['ID'] = high_info.index
	park_info['ID'] = park_info.index
	subway_info['ID'] = subway_info.index

	print("==============================")
	print('Data Preprocessed')
	print("==============================")

	train_data.to_csv('./data/preprocessed/train.csv', index=False)
	test_data.to_csv('./data/preprocessed/test.csv', index=False)
	submission.to_csv('./data/preprocessed/submission.csv', index=False)
	elementary_info.to_csv('./data/preprocessed/elementary_info.csv', index=False)
	middle_info.to_csv('./data/preprocessed/middle_info.csv', index=False)
	high_info.to_csv('./data/preprocessed/high_info.csv', index=False)

	return train_data, test_data, submission, interest_rate, subway_info, elementary_info, middle_info, high_info, park_info
