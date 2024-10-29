from sklearn.preprocessing import LabelEncoder
import pandas as pd

"""
label_encode_column 함수 사용 예시:

'nearest_school_id'와 'nearest_subway_id' 컬럼에 대해 라벨 인코딩을 실험하였으나, 
 최종 모델 성능에 큰 영향을 미치지 않아 사용하지 않기로 결정.
 추후 다른 데이터셋이나 모델에서 다시 시도해 볼 수 있음.
 
train_dist = label_encode_column(train_dist, 'nearest_school_id')
test_dist = label_encode_column(test_dist, 'nearest_school_id')

"""

def label_encode_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    주어진 데이터프레임의 특정 컬럼에 대해 라벨 인코딩을 수행하여 새로운 열에 저장합니다.

    Parameters:
        df (pd.DataFrame): 입력 데이터프레임
        column_name (str): 라벨 인코딩을 적용할 컬럼명

    Returns:
        pd.DataFrame: 라벨 인코딩된 열이 추가된 데이터프레임
    """
    le = LabelEncoder()
    encoded_column_name = f"{column_name}_encoded"
    df[encoded_column_name] = le.fit_transform(df[column_name])
    
    return df