from pandas import DataFrame

from src.utils.variables import MIN_LAT, MAX_LAT, MIN_LNG, MAX_LNG


def remove_duplicates(data: DataFrame) -> DataFrame:
	"""
	Remove duplicates except index from the DataFrame
	:param data: (DataFrame) DataFrame to remove duplicates
	:return: (DataFrame) DataFrame without duplicates
	"""
	return data[~data.drop(columns=['index']).duplicated()].drop(columns=['index'])


def remove_outliers(data: DataFrame, column: str, threshold: float) -> DataFrame:
	"""
	Remove outliers from the DataFrame
	:param data: (DataFrame) DataFrame to remove outliers
	:param column: (str) Column name to remove outliers
	:param threshold: (float) Threshold to remove outliers
	:return: (DataFrame) DataFrame without outliers
	"""
	q1 = data[column].quantile(0.25)
	q3 = data[column].quantile(0.75)
	iqr = q3 - q1
	return data[(data[column] >= q1 - threshold * iqr) & (data[column] <= q3 + threshold * iqr)]


def latlng_boundary_filter(data: DataFrame) -> DataFrame:
	return data[(data['latitude'] >= MIN_LAT) & (data['latitude'] <= MAX_LAT) & (data['longitude'] >= MIN_LNG) & (
			data['longitude'] <= MAX_LNG)]


def remove_underrepresented(data: DataFrame, threshold: int) -> DataFrame:
	"""
	Remove underrepresented data from the DataFrame
	:param data: (DataFrame) DataFrame to remove underrepresented data
	:param threshold: (int) Threshold to remove underrepresented data
	:return: (DataFrame) DataFrame without underrepresented data
	"""
	group_counts = data.groupby(['latitude', 'longitude']).size()
	to_remove = group_counts[group_counts <= threshold].index

	return data[~data.set_index(['latitude', 'longitude']).index.isin(to_remove)]
