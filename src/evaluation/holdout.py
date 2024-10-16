from pandas import DataFrame

from src.utils.variables import HOLDOUT_START, HOLDOUT_END


def get_holdout_data(train_data: DataFrame) -> (DataFrame, DataFrame):
	"""
	Get holdout data from train data
	:param train_data: (DataFrame) Train data
	:return: (DataFrame, DataFrame) Holdout data, Train data
	"""
	print('Get Holdout Data')
	print(f"Holdout period: {HOLDOUT_START} ~ {HOLDOUT_END}")
	holdout_data = train_data[
		(train_data['contract_year_month'] >= HOLDOUT_START) & (train_data['contract_year_month'] <= HOLDOUT_END)]
	train_data = train_data[
		~((train_data['contract_year_month'] >= HOLDOUT_START) & (train_data['contract_year_month'] <= HOLDOUT_END))]

	return holdout_data, train_data
