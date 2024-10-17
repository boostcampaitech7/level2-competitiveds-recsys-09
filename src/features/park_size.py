"""
add_park_size 함수 사용 예시:

공원크기 추가 함수

- nearest 정보가 있는 데이터에 사용 가능 

train_data_= add_park_size(train_data, park_data)
test_data_= add_park_size(test_data, park_data)

"""


def classify_park_size(area):
    """
    공원의 크기를 분류하는 함수.
    
    Args:
        area (float): 공원의 면적.
    
    Returns:
        str: 소형, 중형, 대형 공원 중 하나.
    """
    if area <= 10000:
        return 'Small Park'
    elif area <= 100000:
        return 'Medium Park'
    else:
        return 'Large Park'

def add_park_size(data, park_data):
    """
    데이터에 공원 크기 분류 정보를 추가하는 함수.
    
    Args:
        data (pd.DataFrame): 학습 또는 테스트 데이터 (nearest_park_id 포함).
        park_data (pd.DataFrame): 공원 데이터 (id 및 area 포함).
    
    Returns:
        pd.DataFrame: 공원 크기 분류 정보가 추가된 데이터.
    """
    # 공원 크기 분류
    park_data['id'] = park_data.index
    park_data['park_size'] = park_data['area'].apply(classify_park_size)
    
    # 공원 크기 정보를 데이터에 병합
    data = data.merge(park_data[['id', 'park_size']], 
                      left_on='nearest_park_id', 
                      right_on='id', 
                      how='left')
    
    # 불필요한 id 컬럼 제거
    data.drop(columns=['id'], inplace=True)
    
    return data

