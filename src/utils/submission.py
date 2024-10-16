from pandas import DataFrame


def submission_to_csv(sample_submission: DataFrame, y_test_pred: DataFrame, file_name='output.csv') -> None:
	"""
	Save the submission file
	:param sample_submission: (DataFrame) Sample submission file
	:param y_test_pred: (DataFrame) Predicted target data
	:param file_name: (str) File name, default='output.csv'
	:return: None
	"""
	sample_submission['deposit'] = y_test_pred
	sample_submission.to_csv(file_name, index=False, encoding='utf-8-sig')

	return None