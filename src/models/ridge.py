from sklearn.linear_model import Ridge


def train_ridge(X_train, y_train, alpha=1.0):
	ridge_model = Ridge(alpha=alpha)
	ridge_model.fit(X_train, y_train)

	return ridge_model
