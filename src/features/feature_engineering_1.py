import time
import pandas as pd
from src.features.nearest_school import add_school_distances  # 학교 거리 계산 모듈 import
from src.features.park_size import add_park_size  # 공원 크기 모듈 import
from src.features.recent_deposit import process_final_prices # 전세가 관련 모듈 import
from src.features.contract_timestamp import create_contract_date, create_contract_timestamp
from src.features.deposit_per_area import add_deposit_per_area
from src.features.one_hot_encoding import one_hot_encoding,fit_columns_of_train_and_test
from sklearn.cluster import KMeans

def feature_engineering(train_data, test_data):
    """
    학습 및 테스트 데이터를 위한 특성 엔지니어링 함수. (가까운 지하철, 학교, 공원 거리, K-means 클러스터링, 전세가 추가)
    
    Args:
        train_data (pd.DataFrame): 학습 데이터.
        test_data (pd.DataFrame): 테스트 데이터.
        subway_info (pd.DataFrame): 지하철 위치 정보.
        school_info (pd.DataFrame): 학교 위치 정보.
        park_info (pd.DataFrame): 공원 위치 정보.
        current_date (str): 현재 날짜 (예: '2023-06-30')

    Returns:
        pd.DataFrame, pd.DataFrame: 학습 및 테스트 데이터를 처리한 후 반환.
    """
    print("==============================")
    print("Feature Engineering 시작")
    print("==============================")
    
    start_time = time.time()
    
    print("==============================")
    print("deposit_per_area 추가")
    print("==============================")
    
    train_data = add_deposit_per_area(train_data)
    
    print("==============================")
    print("k-means 클러스터링")
    print("==============================")
    
    # K-Means 클러스터링
    kmeans = KMeans(n_clusters=100, random_state=42)
    train_data['region_cluster'] = kmeans.fit_predict(train_data[['latitude', 'longitude']])
    test_data['region_cluster'] = kmeans.predict(test_data[['latitude', 'longitude']])

    print("==============================")
    print("1년 평균 전세가 계산")
    print("==============================")
    
    train_data, test_data = process_final_prices(train_data, test_data,price_type = 'deposit_per_area')
    
    
    
    print("==============================")
    print("timestampe")
    print("==============================")
    
    train_data= create_contract_date(train_data)
    test_data= create_contract_date(test_data)
    
    train_data = create_contract_timestamp(train_data)
    test_data = create_contract_timestamp(test_data)
    
    print("==============================")
    print("one-hot encoding")
    print("==============================")
    
    
    train_data = one_hot_encoding(train_data,columns=['region_cluster'])
    test_data = fit_columns_of_train_and_test(train_data,test_data)
    
    
    # 필요한 컬럼만 유지
    columns_needed = ['area_m2', 'age' ,'contract_type', 'floor', 'contract_year_month',
                      'deposit','deposit_per_area','nearest_subway_distance',
                      'nearest_school_distance','interest_rate', 
                      'nearest_school_id','nearest_subway_id','final_price','contract_timestamp']
    
    #cluster_0부터 cluster_99까지 추가
    cluster_columns = [f'region_cluster_{i}' for i in range(0, 100)]
    columns_needed.extend(cluster_columns)
    
    columns_needed_test = ['area_m2', 'age' ,'contract_type', 'floor', 'contract_year_month',
                      'nearest_subway_distance','nearest_school_distance','interest_rate', 
                      'nearest_school_id','nearest_subway_id','final_price','contract_timestamp']
    
    columns_needed_test.extend(cluster_columns)
    
    
    train_data = train_data[columns_needed]
    test_data = test_data[columns_needed_test]
    
    # Boolean to int 변환
    for df in [train_data, test_data]:
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)

    # 시간순 정렬
    train_data = train_data.sort_values('contract_timestamp').reset_index(drop=True)

    print(f"\nFeature Engineering took {time.time() - start_time:.2f} seconds\n")
    print("==============================")
    print("Feature Engineering 완료")
    print("==============================")
    
    #train_data.to_csv('./data/processed_features/train_data.csv', index=False)
    #test_data.to_csv('./data/processed_features/test_data.csv', index=False)

    return train_data, test_data