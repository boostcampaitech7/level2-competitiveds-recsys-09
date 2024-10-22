import time

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

from src.features.nearest_public import calculate_nearst_distances_count_id, transform_distances
from src.features.weight import set_weight


def feature_engineering(train_data, test_data, interest_rate, subway_info, elementary_info, middle_info, high_info, park_info):
	print("==============================")
	print("Feature Engineering")
	print("==============================")

	start_time = time.time()

	facilities_info = {'subway': subway_info, 'elementary': elementary_info, 'middle': middle_info, 'high': high_info, 'park': park_info}
	print('start train_data distance calculation')
	train_data = calculate_nearst_distances_count_id(train_data, facilities_info)
	print('start test_data distance calculation')
	test_data = calculate_nearst_distances_count_id(test_data, facilities_info)

	'''distance_columns = ['subway_distance', 'school_distance', 'park_distance']
	train_data = transform_distances(train_data, distance_columns)
	test_data = transform_distances(test_data, distance_columns)'''

	# TODO: Modularize
	###
	'''train_area_m2_min = train_data[train_data['area_m2'] > 0]['area_m2'].min()
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
	test_data = test_data.drop(columns=['area_m2', 'floor', 'age'])'''

	'''# K-Means Clustering
	kmeans = KMeans(n_clusters=100, random_state=42)
	train_data['region_cluster'] = kmeans.fit_predict(train_data[['latitude', 'longitude']])
	test_data['region_cluster'] = kmeans.predict(test_data[['latitude', 'longitude']])'''

	'''# Rolling Average
	train_data['interest_rate_3mo_avg'] = train_data['interest_rate'].rolling(window=3).mean()
	test_data['interest_rate_3mo_avg'] = test_data['interest_rate'].rolling(window=3).mean()
	###'''

	# weight 설정
	weight = 2
	print(f'Setting weight to {weight}')
	train_data = set_weight(train_data, weight)

	# 필요한 열만 선택
	train_data_clustering = train_data[['latitude', 'longitude', 'deposit']]
	test_data_clustering = test_data[['latitude', 'longitude']]
	test_data_clustering.loc[:, 'deposit'] = 0

	# 스케일링
	scaler = StandardScaler()
	train_data_clustering_scaled = scaler.fit_transform(train_data_clustering)
	test_data_clustering_scaled = scaler.transform(test_data_clustering)

	# test_data_clustering_scaled에서 deposit에 해당하는 열 제거
	test_data_clustering_scaled = test_data_clustering_scaled[:, :-1]
	
	# KMeans 클러스터링 수행
	kmeans= KMeans(n_clusters=1000, init='k-means++', random_state=42)
	train_data['region_cluster'] = kmeans.fit_predict(train_data_clustering_scaled)

	# 클러스터 중심 계산
	centroids = train_data.groupby('region_cluster')[['latitude', 'longitude']].mean()

	# NearestCentroid를 사용하여 test_data에 클러스터 할당
	nearest_centroid = NearestCentroid()
	nearest_centroid.fit(centroids, centroids.index)
	test_data['region_cluster'] = nearest_centroid.predict(test_data[['latitude', 'longitude']])

	columns_needed = ['deposit', 'area_m2', 'year', 'month', 'floor', 'latitude', 'longitude',
						'subway_distance', 'elementary_distance', 'middle_distance', 'high_distance', 'park_distance', 
						'subway_ID', 'elementary_ID', 'middle_ID', 'high_ID', 'park_ID',
						'subway_count', 'elementary_count', 'middle_count', 'high_count', 'park_count',
						'age', 'interest_rate', 'region_cluster', 'weight']

	columns_needed_test = ['area_m2', 'year', 'month', 'floor', 'latitude', 'longitude',
						'subway_distance', 'elementary_distance', 'middle_distance', 'high_distance', 'park_distance', 
						'subway_ID', 'elementary_ID', 'middle_ID', 'high_ID', 'park_ID',
						'subway_count', 'elementary_count', 'middle_count', 'high_count', 'park_count',
						'age', 'interest_rate', 'region_cluster']

	train_data = train_data[columns_needed]
	test_data = test_data[columns_needed_test]

	print(f"\nFeature Engineering took {time.time() - start_time:.2f} seconds\n")

	print("==============================")
	print("Feature Engineering Completed")
	print("==============================")

	train_data.to_csv('./data/processed_features/train_data.csv', index=False)
	test_data.to_csv('./data/processed_features/test_data.csv', index=False)

	return train_data, test_data
