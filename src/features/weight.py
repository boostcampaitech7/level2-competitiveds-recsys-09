from pandas import DataFrame

from src.utils.variables import WEIGHT_START, WEIGHT_END

def set_weight(train_data: DataFrame, weight: float) -> DataFrame:
    """
    Set the weight of the data
    :param data: (pd.DataFrame) Data
    :param start: (int) Start mon
    :param end: (int) End index
    :param weight: (float) Weight
    :return: (pd.DataFrame) Data with weight
    """
    print('Set Weight')
    print(f"Weight period: {WEIGHT_START} ~ {WEIGHT_END}")
    
    default_weight = 1
    
    condition = (train_data['year'] * 100 + train_data['month'] >= WEIGHT_START) & (
                train_data['year'] * 100 + train_data['month'] <= WEIGHT_END)
    
    train_data.loc[condition, 'weight'] = weight
    train_data.loc[~condition, 'weight'] = default_weight
    
    return train_data