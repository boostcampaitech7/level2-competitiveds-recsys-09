from pandas import DataFrame


def split_year_month(data: DataFrame) -> DataFrame:
	"""
	Split contract_year_month to year and month
	:param data: (DataFrame) Data to split year and month
	:return: (DataFrame) Data with year and month, drop contract_year_month
	"""
	data['year'] = data['contract_year_month'] // 100
	data['month'] = data['contract_year_month'] % 100

	return data.drop(columns=['contract_year_month'])
