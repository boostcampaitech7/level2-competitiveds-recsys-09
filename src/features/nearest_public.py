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
	tree = BallTree(np.radians(points[['latitude', 'longitude']]), metric='haversine')
	distances, _ = tree.query(np.radians(data[['latitude', 'longitude']]), k=1)
	return distances.flatten() * 6371


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


def transform_log_distances(data: pd.DataFrame, columns: list) -> pd.DataFrame:
	for col in columns:
		data[f'log_{col}'] = nearest_public_log_transform(data[col])

	return data


def nearest_public_sqrt_transform(distance: pd.Series) -> pd.Series:
	"""
	Calculate the square root transform of the column
	:param distance: (pd.Series) nearest Public facility distance
	:return: (pd.Series) Square root transformed column
	"""
	return np.sqrt(distance)


def transform_sqrt_distances(data: pd.DataFrame, columns: list) -> pd.DataFrame:
	for col in columns:
		data[f'sqrt_{col}'] = nearest_public_sqrt_transform(data[col])

	return data

def count_within_radius(df: pd.DataFrame, points: pd.DataFrame, radius=1):
	"""
	Count the number of points within the radius
	:param df: (pd.DataFrame) Data
	:param points: (pd.DataFrame) Target points
	:param radius: (float) Radius
	:return: (int) Count
	"""
	tree = BallTree(np.radians(points[['latitude', 'longitude']]), metric='haversine')
	count = tree.query_radius(np.radians(df[['latitude', 'longitude']]), r=radius, count_only=True)
	return count


def nearest_school_count_with_type(data: pd.DataFrame, school_info: pd.DataFrame, elementary_distance=1,
								   middle_distance=3, high_distance=3) -> pd.DataFrame:
	"""
	Count the number of schools with types within the distance
	:param data: (pd.DataFrame) Data
	:param school_info: (pd.DataFrame) School information
	:param elementary_distance: (int) Elementary school distance
	:param middle_distance: (int) Middle school distance
	:param high_distance: (int) High school distance
	:return: (pd.DataFrame) Data with school count columns
	"""
	data['elementary_school_count'] = count_within_radius(data, school_info[school_info['schoolLevel'] == 'elementary'],
														  radius=elementary_distance)
	data['middle_school_count'] = count_within_radius(data, school_info[school_info['schoolLevel'] == 'middle'],
													  radius=middle_distance)
	data['high_school_count'] = count_within_radius(data, school_info[school_info['schoolLevel'] == 'high'],
													radius=high_distance)

	return data
