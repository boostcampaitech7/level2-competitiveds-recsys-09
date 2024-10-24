import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.features.nearest_public import calculate_nearst_distances, transform_distances
from src.features.cluster_feature_add import perform_clustering_and_calculate_deposit
from src.features.clustering import kmeans


def feature_engineering(train_data, test_data, interest_rate, subway_info, school_info, park_info):
	print("==============================")
	print("Feature Engineering")
	print("==============================")

	start_time = time.time()

	n_clusters_list = [1500]
	max_iters = 100
	tol = 1e-6
	train_data, test_data = perform_clustering_and_calculate_deposit(train_data, test_data, n_clusters_list, max_iters, tol)

	train_cluster_features = []
	test_cluster_features = []
	train_cluster_features.extend(train_data.columns[train_data.columns.str.contains(f'cluster_1500')])
	test_cluster_features.extend(test_data.columns[test_data.columns.str.contains(f'cluster_1500')])

	train_data['deposit_per_area'] = train_data['deposit'] / train_data['area_m2']
	
	# TODO: Modularize
	###
	train = pd.read_csv('./data/preprocessed/train.csv')
	test = pd.read_csv('./data/preprocessed/test.csv')

	'''	train_data['nearest_subway_distance'] = train['nearest_subway_distance']
	train_data['nearest_school_distance'] = train['nearest_school_distance']
	train_data['nearest_park_distance'] = train['nearest_park_distance']

	test_data['nearest_subway_distance'] = test['nearest_subway_distance']
	test_data['nearest_school_distance'] = test['nearest_school_distance']
	test_data['nearest_park_distance'] = test['nearest_park_distance']'''


	train_nearest_subway_distance_min = train['nearest_subway_distance'].min()
	train_nearest_school_distance_min = train['nearest_school_distance'].min()
	train_nearest_park_distance_min = train['nearest_park_distance'].min()

	train_data['log_nearest_subway_distance'] = np.where(train['nearest_subway_distance'] > 0, train['nearest_subway_distance'], train_nearest_subway_distance_min)
	train_data['log_nearest_school_distance'] = np.where(train['nearest_school_distance'] > 0, train['nearest_school_distance'], train_nearest_school_distance_min)
	train_data['log_nearest_park_distance'] = np.where(train['nearest_park_distance'] > 0, train['nearest_park_distance'], train_nearest_park_distance_min)

	test_nearest_subway_distance_min = test['nearest_subway_distance'].min()
	test_nearest_school_distance_min = test['nearest_school_distance'].min()
	test_nearest_park_distance_min = test['nearest_park_distance'].min()

	test_data['log_nearest_subway_distance'] = np.where(test['nearest_subway_distance'] > 0, test['nearest_subway_distance'], test_nearest_subway_distance_min)
	test_data['log_nearest_school_distance'] = np.where(test['nearest_school_distance'] > 0, test['nearest_school_distance'], test_nearest_school_distance_min)
	test_data['log_nearest_park_distance'] = np.where(test['nearest_park_distance'] > 0, test['nearest_park_distance'], test_nearest_park_distance_min)
	
	train_data['contract_year_month_day'] = train_data["contract_year_month"] * 100 + train_data["contract_day"]
	test_data['contract_year_month_day'] = test_data["contract_year_month"] * 100 + test_data["contract_day"]
	train_data.drop(columns=['contract_year_month', 'contract_day'], inplace=True)
	test_data.drop(columns=['contract_year_month', 'contract_day'], inplace=True)

	print(train_data.head())
	columns_needed_test = ['contract_year_month_day', 'floor', 'latitude', 'longitude','age', 'area_m2', 
				   		'log_nearest_subway_distance', 'log_nearest_school_distance', 'log_nearest_park_distance'
						] + train_cluster_features
	columns_needed = columns_needed_test + ['deposit_per_area']
	
	train_data = train_data[columns_needed].sort_values('contract_year_month_day')
	test_data =	test_data[columns_needed_test]
	#train_data = train_data[columns_needed]
	#test_data = test_data[columns_needed_test]

	print(f"\nFeature Engineering took {time.time() - start_time:.2f} seconds\n")

	print("==============================")
	print("Feature Engineering Completed")
	print("==============================")

	train_data.to_csv('./data/processed_features/train_data.csv', index=False)
	test_data.to_csv('./data/processed_features/test_data.csv', index=False)

	return train_data, test_data
