import time

import numpy as np
from sklearn.cluster import KMeans
from pandas import DataFrame

from src.features.nearest_public import calculate_nearst_distances, calculate_nearst_distances_count_id, transform_distances, transform_sqrt_distances
from src.features.one_hot_encoding import one_hot_encoding, fit_columns_of_train_and_test
from src.features.year_month_day import add_year_month_day
from src.features.age_weight import age_weight


def feature_engineering2(train_data: DataFrame, test_data: DataFrame, interest_rate: DataFrame, subway_info: DataFrame, school_info: DataFrame, park_info: DataFrame) -> (DataFrame, DataFrame):
	print("==============================")
	print("Feature Engineering")
	print("==============================")

	start_time = time.time()
	
    # deposit_per_area
	train_data['deposit_per_area'] = train_data['deposit'] / train_data['area_m2']

	# calculate nearest distances
	facilities_info = {'subway': subway_info, 'school': school_info, 'park': park_info}
	train_data = calculate_nearst_distances_count_id(train_data, facilities_info)
	test_data = calculate_nearst_distances_count_id(test_data, facilities_info)

	# transform dist to sqrt
	distance_columns = ['subway_distance', 'school_distance', 'park_distance']
	train_data = transform_sqrt_distances(train_data, distance_columns)
	test_data = transform_sqrt_distances(test_data, distance_columns)

    # add year_month_day
	train_data = add_year_month_day(train_data)
	test_data = add_year_month_day(test_data)
    
    # add age_weight
	train_data = age_weight(train_data)
	test_data = age_weight(test_data)
    
	# K-Means Clustering
	kmeans = KMeans(n_clusters=100, random_state=42)
	train_data['region_cluster'] = kmeans.fit_predict(train_data[['latitude', 'longitude']])
	test_data['region_cluster'] = kmeans.predict(test_data[['latitude', 'longitude']])
	
    # One-hot Encoding
	train_data = one_hot_encoding(train_data, ['region_cluster'])
	test_data = one_hot_encoding(test_data, ['region_cluster'])
	test_data = fit_columns_of_train_and_test(train_data, test_data)

	columns_needed = ['deposit_per_area', 'year', 'month', 'area_m2', 'built_year', 'latitude', 'longitude',
					  'sqrt_subway_distance', 'sqrt_school_distance', 'sqrt_park_distance', 
					  'subway_ID', 'school_ID', 'park_ID', 'interest_rate',
					  'contract_timestamp_scaled', 'final_price', 'age_weight'] + [f'region_cluster_{i}' for i in range(100)]
	columns_needed_test = ['year', 'month', 'area_m2', 'built_year', 'latitude', 'longitude',
					  'sqrt_subway_distance', 'sqrt_school_distance', 'sqrt_park_distance', 
					  'subway_ID', 'school_ID', 'park_ID', 'interest_rate',
					  'contract_timestamp_scaled', 'final_price', 'age_weight'] + [f'region_cluster_{i}' for i in range(100)]

	train_data = train_data[columns_needed]
	test_data = test_data[columns_needed_test]

	print(f"\nFeature Engineering took {time.time() - start_time:.2f} seconds\n")

	print("==============================")
	print("Feature Engineering Completed")
	print("==============================")

	train_data.to_csv('./data/processed_features/train_data.csv', index=False)
	test_data.to_csv('./data/processed_features/test_data.csv', index=False)

	return train_data, test_data
