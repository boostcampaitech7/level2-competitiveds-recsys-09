from pandas import DataFrame

from src.utils.variables import MIN_LAT, MAX_LAT, MIN_LNG, MAX_LNG


def remove_duplicates(data: DataFrame) -> DataFrame:
	"""
	Remove duplicates except index from the DataFrame
	:param data: (DataFrame) DataFrame to remove duplicates
	:return: (DataFrame) DataFrame without duplicates
	"""
	return data[~data.drop(columns=['index']).duplicated()].drop(columns=['index'])


# TODO: Modify!
def remove_outliers(data: DataFrame, column: str, threshold: float) -> DataFrame:
	"""
	Remove outliers from the DataFrame
	:param data: (DataFrame) DataFrame to remove outliers
	:param column: (str) Column name to remove outliers
	:param threshold: (float) Threshold to remove outliers
	:return: (DataFrame) DataFrame without outliers
	"""
	return data[(data[column] < threshold) | (data[column] > data[column].quantile(1 - threshold))]


def latlng_boundary_filter(data: DataFrame) -> DataFrame:
	return data[(data['latitude'] >= MIN_LAT) & (data['latitude'] <= MAX_LAT) & (data['longitude'] >= MIN_LNG) & (
			data['longitude'] <= MAX_LNG)]

def remove_quantile(data: DataFrame, column: str, q1: float, q2: float) -> DataFrame:
	"""
	Remove outliers from the DataFrame
	:param data: (DataFrame) DataFrame to remove outliers
	:param column: (str) Column name to remove outliers
	:param q1: (float) Quantile 1
	:param q2: (float) Quantile 2
	:return: (DataFrame) DataFrame without outliers
	"""
	return data[(data[column] > data[column].quantile(q1)) & (data[column] < data[column].quantile(q2))]
