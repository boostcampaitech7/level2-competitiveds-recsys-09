from src.utils.variables import RANDOM_SEED
from xgboost import XGBRegressor

def train_xgb(X_train, y_train):
	xgb_model = XGBRegressor(random_state=RANDOM_SEED)
	xgb_model.fit(X_train, y_train)

	return xgb_model
