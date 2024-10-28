import pandas as pd
import numpy as np
from pandas import DataFrame

def age_weight(data: DataFrame) -> DataFrame:
    """
    Age Weight
    
    age를 구간별로 나누어 가중치를 부여한다.
    
    :param data: (DataFrame) Data to calculate age weight
    :return: (DataFrame) Data with age weight
    """
    conditions = [
        (data['age'] <= 5),     # 5년 이하
        (data['age'] <= 15),    # 5년 초과 15년 이하
        (data['age'] > 15)      # 15년 초과
    ]
    weights = [3, 2, 1]
    
    data['age_weight'] = np.select(conditions, weights)
    return data