import pandas as pd

def add_deposit_per_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터프레임에 deposit_per_area 컬럼을 추가하는 함수.
    deposit 컬럼을 area_m2 컬럼으로 나눈 값으로 새로운 컬럼 생성.

    Parameters:
    - df (pd.DataFrame): 'deposit'와 'area_m2' 컬럼을 포함한 입력 데이터프레임

    Returns:
    - df (pd.DataFrame): 'deposit_per_area' 컬럼이 추가된 데이터프레임
    """
    df['deposit_per_area'] = df['deposit'] / df['area_m2']
    return df