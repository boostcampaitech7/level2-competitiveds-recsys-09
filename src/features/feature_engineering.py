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

	print('start clustering')
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
	train_data.loc[:, 'region_cluster'] = kmeans.fit_predict(train_data_clustering_scaled)

	# vector similarity 계산을 위한 열
	vector_similarity_columns = ['area_m2', 'contract_date_epoch', 'contract_type', 'floor', 'built_year', 'latitude', 'longitude','age', 'interest_rate']

	scaler = StandardScaler()
	train_data_vector_scaled = scaler.fit_transform(train_data[vector_similarity_columns])
	test_data_vector_scaled = scaler.transform(test_data[vector_similarity_columns])

	train_data_vector_scaled = pd.DataFrame(train_data_vector_scaled, columns=vector_similarity_columns)
	test_data_vector_scaled = pd.DataFrame(test_data_vector_scaled, columns=vector_similarity_columns)

	train_data_vector_scaled.loc[:, 'region_cluster'] = train_data['region_cluster']

	# cluster 중심 계산
	centroids = train_data_vector_scaled.groupby('region_cluster').mean()

	# cluster 중심과의 거리 계산
	nc = NearestCentroid()
	nc.fit(centroids, centroids.index)
	test_data.loc[:, 'region_cluster'] = nc.predict(test_data_vector_scaled)
	train_data.loc[:, 'region_cluster'] = nc.predict(train_data_vector_scaled.drop('region_cluster', axis=1))

	facilities_info = {'subway': subway_info, 'elementary': elementary_info, 'middle': middle_info, 'high': high_info, 'park': park_info}
	print('start train_data distance calculation')
	train_data = calculate_nearst_distances_count_id(train_data, facilities_info)
	print('start test_data distance calculation')
	test_data = calculate_nearst_distances_count_id(test_data, facilities_info)

	# weight 설정
	# [x] TODO: Epoch time 적용
	weight = 2
	print(f'Setting weight to {weight}')
	train_data = set_weight(train_data, weight)

	columns_needed = ['deposit', 'area_m2', 'contract_date_epoch', 'contract_type', 'floor', 'built_year', 'latitude', 'longitude',
						'subway_distance', 'elementary_distance', 'middle_distance', 'high_distance', 'park_distance', 
						'subway_ID', 'elementary_ID', 'middle_ID', 'high_ID', 'park_ID',
						'subway_count', 'elementary_count', 'middle_count', 'high_count', 'park_count',
						'age', 'interest_rate', 'region_cluster', 'weight']

	columns_needed_test = ['area_m2', 'contract_date_epoch', 'contract_type', 'floor', 'built_year', 'latitude', 'longitude',
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
