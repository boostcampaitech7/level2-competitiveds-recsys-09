import lightgbm as lgb
from src.utils.variables import RANDOM_SEED


def train_lgb(X_train, y_train):
	lgb_model = lgb.LGBMRegressor(random_state=RANDOM_SEED)
	lgb_model.fit(X_train, y_train)

	return lgb_model
