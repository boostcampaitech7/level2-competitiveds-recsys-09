from sklearn.linear_model import LinearRegression


def train_linear_regression(X_train, y_train):
	linear_model = LinearRegression()
	linear_model.fit(X_train, y_train)

	return linear_model
