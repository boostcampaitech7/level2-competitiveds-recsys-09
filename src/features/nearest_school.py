import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

"""
add_school_distances 함수 사용 예시:

초중고 각각 최단 거리 및 학교 아이디

train_data_= add_school_distances(train_data, school_data)
test_data_= add_school_distances(test_data, school_data)

"""

def prepare_location_tree(location_data):
    """
    위도와 경도를 라디안으로 변환하여 BallTree를 생성합니다.
    
    Args:
        location_data (pd.DataFrame): 위치 데이터 (latitude, longitude 컬럼 포함).
    
    Returns:
        BallTree: 하버사인 거리로 위치 트리 생성
    """
    coords = np.radians(location_data[['latitude', 'longitude']])
    tree = BallTree(coords, metric='haversine')
    return tree


def find_nearest_location_optimized(data, location_data, location_type) -> pd.DataFrame:
    """
    주어진 데이터에 대해 가장 가까운 위치를 찾습니다.
    
    Args:
        data (pd.DataFrame): 학습 또는 테스트 데이터 (latitude, longitude 컬럼 포함).
        location_data (pd.DataFrame): 위치 데이터 (latitude, longitude 컬럼 포함).
        location_type (str): 위치 유형 ('elementary', 'middle', 'high').
    
    Returns:
        pd.DataFrame: 가장 가까운 초중고 위치의 거리와 ID를 포함한 데이터프레임.
   
    """
    location_data['id'] = location_data.index
    # BallTree 준비
    tree = prepare_location_tree(location_data)
    test_coords = np.radians(data[['latitude', 'longitude']])
    
    # 가장 가까운 위치 찾기 (거리와 인덱스)
    distances, indices = tree.query(test_coords, k=1)
    
    # 거리 계산 (단위: m)
    distances_m = distances[:, 0] * 6371000  # 라디안에서 m로 변환
    
    # 가장 가까운 위치의 id 추출
    nearest_ids = location_data.iloc[indices[:, 0]]['id'].values
    
    # 결과 반환
    return pd.DataFrame({
        f'nearest_{location_type}_distance': distances_m,
        f'nearest_{location_type}_id': nearest_ids
    })


def add_school_distances(data, school_data)->pd.DataFrame:
    """
    데이터에 초등학교, 중학교, 고등학교의 거리를 추가합니다.
    
    Args:
        data (pd.DataFrame): 학습 또는 테스트 데이터 (latitude, longitude 컬럼 포함).
        school_data (pd.DataFrame): 학교 위치 데이터 (latitude, longitude, schoolLevel 컬럼 포함).
    
    Returns:
        pd.DataFrame: 초등학교, 중학교, 고등학교의 거리와 ID가 추가된 데이터프레임.
    """
    # 학교 유형별로 데이터 분리
    elementary_data = school_data[school_data['schoolLevel'] == 'elementary']
    middle_data = school_data[school_data['schoolLevel'] == 'middle']
    high_data = school_data[school_data['schoolLevel'] == 'high']
    
    # 초등학교, 중학교, 고등학교 각각에 대한 거리 계산
    nearest_elementary = find_nearest_location_optimized(data, elementary_data, 'elementary')
    nearest_middle = find_nearest_location_optimized(data, middle_data, 'middle')
    nearest_high = find_nearest_location_optimized(data, high_data, 'high')
    
    # 결과 병합 (초등, 중등, 고등학교 거리)
    data = pd.concat([data, nearest_elementary, nearest_middle, nearest_high], axis=1)
    
    return data