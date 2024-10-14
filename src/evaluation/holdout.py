from src.utils.variables import HOLDOUT_START, HOLDOUT_END


def get_holdout_data(train_data):
	holdout_data = train_data[
		(train_data['contract_year_month'] >= HOLDOUT_START) & (train_data['contract_year_month'] <= HOLDOUT_END)]
	train_data = train_data[
		~((train_data['contract_year_month'] >= HOLDOUT_START) & (train_data['contract_year_month'] <= HOLDOUT_END))]
	return holdout_data
