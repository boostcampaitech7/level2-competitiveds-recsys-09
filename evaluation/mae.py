from sklearn.metrics import mean_absolute_error


def evaluate_mae(model, X_holdout, y_holdout, holdout_pred):
	y_pred = model.predict(X_holdout)
	mae = mean_absolute_error(y_holdout, holdout_pred)

	return mae
