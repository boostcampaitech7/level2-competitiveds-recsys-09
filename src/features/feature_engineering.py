from src.features.nearest_public import calculate_nearst_distances, transform_distances


def feature_engineering(train_data, test_data, interest_rate, subway_info, school_info, park_info):
	facilities_info = {'subway': subway_info, 'school': school_info, 'park': park_info}
	train_data = calculate_nearst_distances(train_data, facilities_info)
	test_data = calculate_nearst_distances(test_data, facilities_info)

	distance_columns = ['subway_distance', 'school_distance', 'park_distance']
	train_data = transform_distances(train_data, distance_columns)
	test_data = transform_distances(test_data, distance_columns)

	columns_needed = ['area_m2', 'contract_year_month', 'contract_type', 'floor', 'latitude', 'longitude', 'deposit', 'built_year']
	columns_needed_test = ['area_m2', 'contract_year_month', 'contract_type', 'floor', 'latitude', 'longitude', 'built_year']

	# train_data = train_data[columns_needed]
	# test_data = test_data[columns_needed_test]

	return train_data, test_data
