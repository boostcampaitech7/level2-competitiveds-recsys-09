from sklearn.metrics import mean_absolute_error


def evaluate_mae(model, holdout_data, test_data, holdout_pred):
	y_pred = model.predict(holdout_data)
	mae = mean_absolute_error(test_data, holdout_pred)

	print(f"{model} MAE: {mae}")

	return mae
