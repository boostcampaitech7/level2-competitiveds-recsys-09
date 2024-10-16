from src.preprocessing.remove_data import remove_duplicates, latlng_boundary_filter
from src.preprocessing.split_data import split_year_month


def data_preprocessing(train_data, test_data, interest_rate, subway_info, school_info, park_info):
	train_data = remove_duplicates(train_data)
	test_data = remove_duplicates(test_data)

	train_data = split_year_month(train_data)
	test_data = split_year_month(test_data)

	subway_info = latlng_boundary_filter(subway_info)
	school_info = latlng_boundary_filter(school_info)
	park_info = latlng_boundary_filter(park_info)

	return train_data, test_data, interest_rate, subway_info, school_info, park_info
