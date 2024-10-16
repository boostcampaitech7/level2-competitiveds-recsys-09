import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def nearest_public_balltree(data: pd.DataFrame, points: pd.DataFrame) -> list:
	"""
	Find the nearest public facility using BallTree
	:param data: (pd.DataFrame) Data
	:param points: (pd.DataFrame) Target points
	:return: (list) Nearest distance
	"""
	tree = BallTree(points[['latitude', 'longitude']], metric='haversine')
	distances, _ = tree.query(data[['latitude', 'longitude']], k=1)
	return distances.flatten()


def calculate_nearst_distances(data: pd.DataFrame, facilities_info: dict) -> pd.DataFrame:
	"""
	Calculate the nearest distance from the public facilities
	:param data: (pd.DataFrame) Data
	:param facilities_info: (dict) Public facility information
	:return: (pd.DataFrame) Data with nearest distance columns
	"""
	for facility_name, facility_info in facilities_info.items():
		data[f'{facility_name}_distance'] = nearest_public_balltree(data, facility_info)

	return data


def nearest_public_log_transform(distance: pd.Series) -> pd.Series:
	"""
	Calculate the log transform of the column
	:param distance: (pd.Series) nearest Public facility distance
	:return: (pd.Series) Log transformed column
	"""
	return np.log1p(distance)


def transform_distances(data: pd.DataFrame, columns: list) -> pd.DataFrame:
	for col in columns:
		data[f'log_{col}'] = nearest_public_log_transform(data[col])

	return data.drop(columns=columns)
