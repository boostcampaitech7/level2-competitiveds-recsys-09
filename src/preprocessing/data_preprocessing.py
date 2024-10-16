import pandas as pd


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
	"""
	Remove duplicates except index from the DataFrame
	:param data: (pd.DataFrame) DataFrame to remove duplicates
	:return: (pd.DataFrame) DataFrame without duplicates
	"""
	return data[~data.drop(columns=['index']).duplicated()].drop(columns=['index'])
