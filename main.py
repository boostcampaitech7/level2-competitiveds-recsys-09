import numpy as np
import pandas as pd
from src.data.data_loader import download_data, extract_data, load_raw_data
from src.evaluation.holdout import get_holdout_data
from src.evaluation.mae import evaluate_mae
from src.features.feature_engineering import feature_engineering
from src.models.lgb import train_lgb
from src.utils.variables import RANDOM_SEED

np.random.seed(RANDOM_SEED)


def main():
	try:
		train_data, test_data, submission, interest_rate, subway_info, school_info, park_info = load_raw_data()
	except FileNotFoundError:
		print('Data not found. Downloading...')
		download_data()
		extract_data()

		train_data, test_data, submission, interest_rate, subway_info, school_info, park_info = load_raw_data()

	train_data, test_data = feature_engineering(train_data, test_data, interest_rate, subway_info, school_info, park_info)

	holdout_data = get_holdout_data(train_data)

	X_train = train_data.drop(columns=['deposit'])
	y_train = train_data['deposit']
	X_holdout = holdout_data.drop(columns=['deposit'])
	y_holdout = holdout_data['deposit']
	X_test = test_data.copy()

	lgb_model = train_lgb(X_train, y_train)
	evaluate_mae(lgb_model, X_holdout, y_holdout)

	lgb_test_pred = lgb_model.predict(X_test)
	submission['deposit'] = lgb_test_pred
	submission.to_csv('output.csv', index=False, encoding='utf-8-sig')
	output = pd.read_csv('output.csv')


if __name__ == '__main__':
	main()
