import pandas as pd

from src.utils.variables import MIN_LAT, MAX_LAT, MIN_LNG, MAX_LNG


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
	"""
	Remove duplicates except index from the DataFrame
	:param data: (pd.DataFrame) DataFrame to remove duplicates
	:return: (pd.DataFrame) DataFrame without duplicates
	"""
	return data[~data.drop(columns=['index']).duplicated()].drop(columns=['index'])


def set_latlng_boundary(data: pd.DataFrame) -> pd.DataFrame:
	return data[(data['latitude'] >= MIN_LAT) & (data['latitude'] <= MAX_LAT) & (data['longitude'] >= MIN_LNG) & (
			data['longitude'] <= MAX_LNG)]
