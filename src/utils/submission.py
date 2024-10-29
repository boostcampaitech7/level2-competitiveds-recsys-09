from numpy import ndarray
from pandas import DataFrame


def submission_to_csv(sample_submission: DataFrame, y_test_pred: ndarray, file_name='output.csv') -> None:
	"""
	Save the submission file
	:param sample_submission: (DataFrame) Sample submission file
	:param y_test_pred:
	:param file_name: (str) File name, default='output.csv'
	:return: None
	"""
	print("==============================")
	print('Saving Submission')
	print("==============================")
	sample_submission['deposit'] = y_test_pred
	sample_submission.to_csv(file_name, index=False, encoding='utf-8-sig')

	print('\nSubmission Saved\n')

	return None
