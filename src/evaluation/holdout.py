from pandas import DataFrame, to_datetime

from src.utils.variables import HOLDOUT_START, HOLDOUT_END

# [x] TODO: Apply Epoch time
def get_holdout_data(train_data: DataFrame) -> tuple[DataFrame, DataFrame]:
	"""
	Get holdout data from train data
	:param train_data: (DataFrame) Train data
	:return: (DataFrame, DataFrame) Holdout data, Train data
	"""

	print('Get Holdout Data')
	print(f"Holdout period: {HOLDOUT_START} ~ {HOLDOUT_END}")
	
	# 문자열을 datetime 형식으로 변환
	holdout_start_dt = to_datetime(HOLDOUT_START)
	holdout_end_dt = to_datetime(HOLDOUT_END)
	
	# datetime을 Unix epoch time 형식으로 변환
	holdout_start_epoch = int(holdout_start_dt.timestamp())
	holdout_end_epoch = int(holdout_end_dt.timestamp())

	condition = (train_data['contract_date_epoch'] >= holdout_start_epoch) & (
                train_data['contract_date_epoch'] <= holdout_end_epoch)

	holdout_data = train_data.loc[condition]
	train_data = train_data.loc[~condition]

	return holdout_data, train_data
