def feature_engineering(train_data, test_data, interest_rate, subway_info, school_info, park_info):
	columns_needed = ['area_m2', 'contract_year_month', 'contract_type', 'floor', 'latitude', 'longitude', 'deposit', 'built_year']
	columns_needed_test = ['area_m2', 'contract_year_month', 'contract_type', 'floor', 'latitude', 'longitude', 'built_year']

	train_data = train_data[columns_needed]
	test_data = test_data[columns_needed_test]

	return train_data, test_data
