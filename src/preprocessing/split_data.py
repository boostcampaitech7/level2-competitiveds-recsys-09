from pandas import DataFrame, to_datetime, Timestamp, Timedelta


def split_year_month(data: DataFrame) -> DataFrame:
	"""
	Split contract_year_month to year and month
	:param data: (DataFrame) Data to split year and month
	:return: (DataFrame) Data with year and month, drop contract_year_month
	"""
	data['year'] = data['contract_year_month'] // 100
	data['month'] = data['contract_year_month'] % 100

	return data.drop(columns=['contract_year_month'])

def continous_date(data: DataFrame) -> DataFrame:
	"""
	Convert the date to Unix time
	:param data: (DataFrame) Data
	:return: (DataFrame) Data with continuous date
	"""
	# [ ] TODO: Unix time 적용 
	
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
