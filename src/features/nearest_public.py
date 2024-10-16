import numpy as np
from pandas import DataFrame
from sklearn.neighbors import BallTree


def nearest_public_balltree(info: DataFrame, points: DataFrame) -> list:
	"""
	Find the nearest public facility using BallTree
	:param info: (DataFrame) Public facility information
	:param points: (DataFrame) Target points
	:return: (list) Nearest distance
	"""
	tree = BallTree(points[['latitude', 'longitude']], metric='haversine')
	distances, _ = tree.query(info[['latitude', 'longitude']], k=1)
	return distances.flatten()


def nearest_public_log_transform(info: DataFrame, column: str) -> DataFrame:
	"""
	Calculate the log transform of the column
	:param info: (DataFrame) Public facility information
	:param column: (str) Column name to calculate the log transform
	:return: (DataFrame) Public facility information with the log transform
	"""
	info[f'log_{column}'] = np.log1p(info[column])
	return info
