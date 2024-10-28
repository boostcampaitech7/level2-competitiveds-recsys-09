from pandas import DataFrame, to_datetime, Timestamp, Timedelta

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

def continous_date(data: DataFrame) -> DataFrame:
	"""
	Convert the date to Unix time
	:param data: (DataFrame) Data
	:return: (DataFrame) Data with continuous date
	"""
	
	# contract_year_month 열과 contract_day 열을 합쳐 contract_date 열을 만들고 datetime 형식으로 변환
	data['contract_year_month'] = data['contract_year_month'].astype(str)
	data['contract_day'] = data['contract_day'].astype(str).str.zfill(2)  # day를 두 자리로 맞춤
	# contract_date 열 생성
	data['contract_date'] = data['contract_year_month'] + data['contract_day']

	# contract_date를 datetime 형식으로 변환
	data['contract_date'] = to_datetime(data['contract_date'], format='%Y%m%d')

	data.drop(['contract_year_month', 'contract_day'], axis=1, inplace=True)

	# contract_date를 Unix epoch time 형식으로 변환하여 새로운 열 추가
	data['contract_date_epoch'] = (data['contract_date'] - Timestamp("1970-01-01")) // Timedelta('1s')

	data.drop(['contract_date'], axis=1, inplace=True)

	return data