from sklearn.metrics import mean_absolute_error

from src.models.lgb import optimize_lgb
import lightgbm as lgb


def train_model(X_train, y_train, X_holdout, y_holdout, X_test):
	best_params = optimize_lgb(X_train, y_train, X_holdout, y_holdout)

	final_lgb = lgb.LGBMRegressor(**best_params)
	final_lgb.fit(X_train, y_train)

	y_holdout_pred = final_lgb.predict(X_test)
	holdout_mae = mean_absolute_error(y_holdout, y_holdout_pred)
	print(f'Holdout MAE: {holdout_mae}')

	y_test_pred = final_lgb.predict(X_test)

	return y_test_pred
