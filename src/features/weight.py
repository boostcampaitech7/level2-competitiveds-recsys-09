from pandas import DataFrame, to_datetime

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
    
    # 문자열을 datetime 형식으로 변환
    weight_start_dt = to_datetime(WEIGHT_START)
    weight_end_dt = to_datetime(WEIGHT_END)
    
    # datetime을 Unix epoch time 형식으로 변환
    weight_start_epoch = int(weight_start_dt.timestamp())
    weight_end_epoch = int(weight_end_dt.timestamp())
    
    default_weight = 1
    
    condition = (train_data['contract_date_epoch'] >= weight_start_epoch) & (
                train_data['contract_date_epoch'] <= weight_end_epoch)
    
    train_data.loc[condition, 'weight'] = weight
    train_data.loc[~condition, 'weight'] = default_weight
    
    return train_data