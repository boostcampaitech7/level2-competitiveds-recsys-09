import pandas as pd
from pandas import DataFrame

def one_hot_encoding(data: DataFrame, columns: list) -> DataFrame:
    """
    One-hot encoding for categorical features
    주의: 기존 column을 drop함
    :param data: (DataFrame) Data to one-hot encode
    :return: (DataFrame) Data with one-hot encoded features
    """
    data = pd.get_dummies(data, columns=columns)
    return data

def fit_columns_of_train_and_test(train_data: DataFrame, test_data: DataFrame) -> DataFrame:
    """
    Fit columns of train and test data
    주의: one-hot encoding을 한 후에 필수로 사용해주기
    :param train_data: (DataFrame) Train data
    :param test_data: (DataFrame) Test data
    :return: (DataFrame) test data with same columns of train data
    """
    # train 데이터셋과 test 데이터셋의 column을 맞춰줌
    test_data = test_data.reindex(columns=train_data.columns, fill_value=0)
    # train의 column의 dtype과 test의 column의 dtype이 다른 경우가 있음
    # train의 column의 dtype을 기준으로 test의 column의 dtype을 변경
    for column in test_data.columns:
        if test_data[column].dtype != train_data[column].dtype:
            print(column)
            test_data[column] = test_data[column].astype(train_data[column].dtype)
            
    return test_data