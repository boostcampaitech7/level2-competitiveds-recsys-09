from sklearn.linear_model import Lasso


def train_lasso(X_train, y_train, alpha=1.0):
	lasso_model = Lasso(alpha=alpha)
	lasso_model.fit(X_train, y_train)

	return lasso_model
