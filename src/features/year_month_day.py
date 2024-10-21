from pandas import DataFrame

def add_year_month_day(data: DataFrame) -> DataFrame:
    """
    Add year_month_day
    :param data: (DataFrame) Data to add year_month_day
    :return: (DataFrame) Data with year_month_day
    """
    data['year_month_day'] = data["year"] * 10000 + data["month"] * 100 + data["contract_day"]
    return data