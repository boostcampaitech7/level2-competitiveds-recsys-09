import time

import numpy as np
from sklearn.cluster import KMeans

from src.features.nearest_public import calculate_nearst_distances, transform_sqrt_distances, nearest_school_count_with_type


def feature_engineering(train_data, test_data, interest_rate, subway_info, school_info, park_info):
	print("==============================")
	print("Feature Engineering")
	print("==============================")

	start_time = time.time()

	park_info = park_info[park_info['area'] >= 100000]

	facilities_info = {'subway': subway_info, 'school': school_info, 'park': park_info}
	train_data = calculate_nearst_distances(train_data, facilities_info)
	test_data = calculate_nearst_distances(test_data, facilities_info)

	distance_columns = ['subway_distance', 'school_distance', 'park_distance']
	train_data = transform_sqrt_distances(train_data, distance_columns)
	test_data = transform_sqrt_distances(test_data, distance_columns)

	train_data = nearest_school_count_with_type(train_data, school_info)
	test_data = nearest_school_count_with_type(test_data, school_info)

	# TODO: Modularize
	###
	train_area_m2_min = train_data[train_data['area_m2'] > 0]['area_m2'].min()
	test_area_m2_min = test_data[test_data['area_m2'] > 0]['area_m2'].min()
	train_data['log_area_m2'] = np.where(train_data['area_m2'] > 0, np.log1p(train_data['area_m2']), train_area_m2_min)
	test_data['log_area_m2'] = np.where(test_data['area_m2'] > 0, np.log1p(test_data['area_m2']), test_area_m2_min)

	train_floor_min = train_data[train_data['floor'] > 0]['floor'].min()
	test_floor_min = test_data[test_data['floor'] > 0]['floor'].min()
	train_data['log_floor'] = np.where(train_data['floor'] > 0, np.log1p(train_data['floor']), train_floor_min)
	test_data['log_floor'] = np.where(test_data['floor'] > 0, np.log1p(test_data['floor']), test_floor_min)

	train_age_min = train_data[train_data['age'] > 0]['age'].min()
	test_age_min = test_data[test_data['age'] > 0]['age'].min()
	train_data['log_age'] = np.where(train_data['age'] > 0, np.log1p(train_data['age']), train_age_min)
	test_data['log_age'] = np.where(test_data['age'] > 0, np.log1p(test_data['age']), test_age_min)

	train_data = train_data.drop(columns=['area_m2', 'floor', 'age'])
	test_data = test_data.drop(columns=['area_m2', 'floor', 'age'])

	# K-Means Clustering
	kmeans = KMeans(n_clusters=100, random_state=42)
	train_data['region_cluster'] = kmeans.fit_predict(train_data[['latitude', 'longitude']])
	test_data['region_cluster'] = kmeans.predict(test_data[['latitude', 'longitude']])

	# Rolling Average
	train_data['interest_rate_3mo_avg'] = train_data['interest_rate'].rolling(window=3).mean()
	test_data['interest_rate_3mo_avg'] = test_data['interest_rate'].rolling(window=3).mean()
	###

	columns_needed = ['deposit', 'log_area_m2', 'year', 'month', 'log_floor', 'latitude', 'longitude',
					  'sqrt_subway_distance', 'sqrt_school_distance', 'sqrt_park_distance', 'log_age', 'region_cluster',
					  'interest_rate_3mo_avg']
	columns_needed_test = ['log_area_m2', 'year', 'month', 'log_floor', 'latitude', 'longitude',
						   'sqrt_subway_distance', 'sqrt_school_distance', 'sqrt_park_distance', 'log_age',
						   'region_cluster', 'interest_rate_3mo_avg']

	train_data = train_data[columns_needed]
	test_data = test_data[columns_needed_test]

	print(f"\nFeature Engineering took {time.time() - start_time:.2f} seconds\n")

	print("==============================")
	print("Feature Engineering Completed")
	print("==============================")

	train_data.to_csv('./data/processed_features/train_data.csv', index=False)
	test_data.to_csv('./data/processed_features/test_data.csv', index=False)

	return train_data, test_data
