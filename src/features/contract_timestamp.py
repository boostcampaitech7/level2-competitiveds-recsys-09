import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

"""
create_contract_timestamp 함수 사용 예시:

계약 년월일 정보를 초단위로 변경

1. datetime 생성

train_data_= create_contract_date(train_data)
test_data_= create_contract_date(test_data)

2. contract_timestamp 생성

train_data_= create_contract_date(train_data)
test_data_= create_contract_date(test_data)

2. contract_timestamp_scaled 생성 (선택 사항)

train_data_= scale_contract_timestamp(train_data)
test_data_= scale_contract_timestamp(test_data)



"""

def create_contract_date(df, year_month_col='contract_year_month', day_col='contract_day')  -> pd.DataFrame :
    """
    주어진 데이터프레임에서 연월과 일 정보를 결합하여 'contract_date' 열을 생성
    + add_year_month_day이랑 같으나 데이터 타입이 다름 
    
    Parameters:
        df (pd.DataFrame): 입력 데이터프레임
        year_month_col (str): 연월 컬럼명 (기본값: 'contract_year_month')
        day_col (str): 일 컬럼명 (기본값: 'contract_day')

    Returns:
        pd.DataFrame: 'contract_date' 열이 추가된 데이터프레임
    """
    df['contract_date'] = pd.to_datetime(
        df[year_month_col].astype(str) + df[day_col].astype(str), format='%Y%m%d'
    )
    return df

def create_contract_timestamp(df, date_col='contract_date')  -> pd.DataFrame :
    """
    주어진 데이터프레임에서 'contract_date' 열을 타임스탬프 형식으로 변환하여 'contract_timestamp' 열을 생성합니다.

    Parameters:
        df (pd.DataFrame): 입력 데이터프레임 
        date_col (str): 타임스탬프로 변환할 날짜 컬럼명 (기본값: 'contract_date' - datetime 데이터타입)

    Returns:
        pd.DataFrame: 'contract_timestamp' 열이 추가된 데이터프레임
    """
    df['contract_timestamp'] = df[date_col].astype(int) // 10**9
    return df


def scale_contract_timestamp(df, timestamp_col='contract_timestamp')  -> pd.DataFrame :
    """
    MinMaxScaler를 사용하여 'contract_timestamp' 열을 0과 1 사이로 정규화합니다.
    'contract_timestamp'와 'contract_timestamp_scaled'는 같은 정보를 담고 있으므로, 
     어느 열을 사용해도 결과에 차이가 없습니다.
     
    Parameters:
        df (pd.DataFrame): 입력 데이터프레임 
        timestamp_col (str): 정규화할 타임스탬프 컬럼명 (기본값: 'contract_timestamp')

    Returns:
        tuple: 정규화된 타임스탬프 열이 추가된 학습 및 테스트 데이터프레임
    """
    scaler = MinMaxScaler()
    df[f'{timestamp_col}_scaled'] = scaler.fit_transform(df[[timestamp_col]])
    
    
    return df