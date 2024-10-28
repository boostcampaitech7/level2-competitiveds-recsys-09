import pandas as pd

"""
process_final_prices 함수 사용 예시:


train_dist와 test_dist에 대해 최근 1년간 평균 전세가와 마지막 거래 가격을 기반으로 final_price를 계산하고,
test 데이터에서 final_price가 없는 경우 클러스터의 평균값으로 대체하는 최종 함수입니다.

이 함수는 클러스터링 이후에 진행해야 하며, current_date 기본값은 '2023-12-31'로 설정되어 있습니다.
price_type 파라미터를 통해 'deposit' 또는 'deposit_per_area' 중 하나를 선택할 수 있습니다. 기본값은 deposit
'deposit_per_area' 피쳐 생성 후 사용해야함

기본값을 사용할 경우:
train_data, test_data = process_final_prices(train_data, test_data)

특정 날짜와 price_type을 지정할 경우:
train_data, test_data = process_final_prices(train_data, test_data, current_date='2023-06-30', price_type='deposit_per_area')

"""
def preprocess_contract_date(data): # 전처리로 옮기기
    """
    contract_year_month와 contract_day로부터 contract_date를 생성하는 전처리 함수.

    Parameters:
    - data (pd.DataFrame): 학습 또는 테스트 데이터프레임

    Returns:
    - data (pd.DataFrame): contract_date가 추가된 데이터프레임
    """
    data['contract_date'] = pd.to_datetime(data['contract_year_month'].astype(str) + data['contract_day'].astype(str), format='%Y%m%d')
    return data


def calculate_avg_price_last_year(train_dist: pd.DataFrame, current_date: str, price_type: str = 'deposit') -> pd.DataFrame:
    """
    최근 1년간 건물별 평균 전세가를 계산하는 함수.
    - 변수에 contract_date: contract_day + contract_year_month 변수를 필요로 합니다.
    - depoist_per_area 변수 필요
    
    Parameters:
    - train_dist (pd.DataFrame): 학습 데이터프레임
    - current_date (str): 현재 날짜 기준 (예: '2023-06-30')
    - price_type (str): 사용할 가격 타입 ('deposit' 또는 'deposit_per_area')
    
    Returns:
    - building_price_avg (pd.DataFrame): 건물별 최근 1년간 평균 전세가
    """
    
    # 최근 1년간의 데이터 필터링
    one_year_ago = pd.to_datetime(current_date) - pd.DateOffset(years=1)
    df_last_year = train_dist[train_dist['contract_date'] >= one_year_ago]

    # 가격 유형에 따라 사용할 컬럼명 설정
    price_column = price_type

    # 건물별로 위도와 경도를 기준으로 그룹화하여 평균 가격 계산
    building_price_avg = df_last_year.groupby(['latitude', 'longitude'])[price_column].mean().reset_index()

    # 열 이름 변경
    building_price_avg = building_price_avg.rename(columns={price_column: f'avg_{price_type}'})
    
    return building_price_avg


def get_last_transaction_price(train_dist: pd.DataFrame, price_type: str = 'deposit') -> pd.DataFrame:
    """
    건물별 마지막 거래 가격을 가져오는 함수.

    Parameters:
    - train_dist (pd.DataFrame): 학습 데이터프레임
    - price_type (str): 사용할 가격 타입 ('deposit' 또는 'deposit_per_area')

    Returns:
    - df_last_transaction (pd.DataFrame): 건물별 마지막 거래 가격을 포함한 데이터프레임
    """
    # 마지막 거래 데이터 가져오기
    df_last_transaction = train_dist.groupby(['latitude', 'longitude']).apply(
        lambda x: x.loc[x['contract_date'].idxmax()]
    )
    
    # 선택된 가격 유형 컬럼만 가져오기
    df_last_transaction = df_last_transaction[['latitude', 'longitude', price_type]].reset_index(drop=True)

    # 열 이름 변경
    df_last_transaction = df_last_transaction.rename(columns={price_type: f'last_{price_type}'})
    
    return df_last_transaction



def merge_final_price(building_price_avg, df_last_transaction, price_type: str = 'deposit'):
    """
    건물별 최근 1년간 평균 전세가와 마지막 거래 가격을 결합하여 최종 가격을 계산하는 함수.
    - 'deposit' 또는 'deposit_per_area' 중 price_type을 지정하여 계산 가능

    Parameters:
    - building_price_avg: 최근 1년간 평균 가격 데이터프레임
    - df_last_transaction: 마지막 거래 가격 데이터프레임
    - price_type (str): 사용할 가격 타입 ('deposit' 또는 'deposit_per_area')

    Returns:
    - result: 건물별 최종 가격을 포함한 데이터프레임
    """
    # 최근 1년 거래 없는 건물들에 대해 마지막 거래 가격을 사용
    result = pd.merge(df_last_transaction, building_price_avg, on=['latitude', 'longitude'], how='left')
    
    # 가격 유형에 따라 최종 열 이름 동적 설정
    avg_column = f'avg_{price_type}'
    last_column = f'last_{price_type}'
    
    # 최근 1년 거래가 없는 경우 마지막 거래 가격 사용
    result['final_price'] = result[avg_column].combine_first(result[last_column])

    return result[['latitude', 'longitude', 'final_price']]

def fill_missing_final_price_in_test(train_dist,test_dist):
    """
    Test 데이터에서 final_price가 없는 경우 같은 클러스터에 속한 건물들의 평균 final_price로 대체하는 함수.
    클러스터링 이후에 진행되어야 함
    
    Parameters:
    - train_dist : train 데이터프레임 -> 클러스터링 이후에 진행해야함
    - test_dist: Test 데이터프레임 -> 클러스터링 이후에 진행해야함

    Returns:
    - test_dist: final_price가 같은 클러스터의 평균값으로 대체된 데이터프레임
    """
    # 클러스터별 final_price의 평균값 계산
    cluster_avg_price_test = train_dist.groupby('region_cluster')['final_price'].mean().reset_index()

    # 결측값을 클러스터 평균으로 대체
    test_dist = pd.merge(test_dist, cluster_avg_price_test, on='region_cluster', how='left', suffixes=('', '_cluster_avg'))
    test_dist['final_price'] = test_dist['final_price'].fillna(test_dist['final_price_cluster_avg'])

    # 불필요한 열 제거
    test_dist = test_dist.drop(columns=['final_price_cluster_avg'])

    return test_dist

def process_final_prices(
    train_dist: pd.DataFrame, 
    test_dist: pd.DataFrame, 
    current_date: str = '2023-12-31', 
    price_type: str = 'deposit'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    train_dist와 test_dist에 대해 최근 1년간 평균 전세가와 마지막 거래 가격을 기반으로 final_price를 계산하고,
    test 데이터에서 final_price가 없는 경우 클러스터의 평균값으로 대체하는 최종 함수.
    
    위도 경도에 따른 클러스터 변수가 있을 때 사용 가능 

    Parameters:
    - train_dist: 학습 데이터프레임
    - test_dist: 테스트 데이터프레임
    - current_date: 날짜 기준 2023-12-31 기본값
    - price_type (str): 사용할 가격 타입 ('deposit' 또는 'deposit_per_area')
    Returns:
    - train_dist: final_price가 포함된 학습 데이터프레임
    - test_dist: final_price가 포함된 테스트 데이터프레임
    """
    train_dist = preprocess_contract_date(train_dist) 
    
    # 1. 최근 1년간 평균 전세가 계산
    building_price_avg = calculate_avg_price_last_year(train_dist, current_date,price_type=price_type)
    
    # 2. 건물별 마지막 거래 가격 계산
    df_last_transaction = get_last_transaction_price(train_dist,price_type=price_type)
    
    # 3. 평균 전세가와 마지막 거래 가격을 결합하여 최종 가격 계산
    result = merge_final_price( building_price_avg, df_last_transaction,price_type=price_type)
    
    # 4. train_dist와 test_dist에 최종 가격 정보 병합
    train_dist = pd.merge(train_dist, result[['latitude', 'longitude', 'final_price']], on=['latitude', 'longitude'], how='left')
    test_dist = pd.merge(test_dist, result[['latitude', 'longitude', 'final_price']], on=['latitude', 'longitude'], how='left')
    
    # 5. 테스트 데이터에서 결측값 처리 (클러스터 평균값으로 대체)
    test_dist = fill_missing_final_price_in_test(train_dist, test_dist)
    
    return train_dist, test_dist