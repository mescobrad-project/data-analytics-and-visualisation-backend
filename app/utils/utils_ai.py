from sklearn.linear_model import LinearRegression


def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model.
    Parameters:
    - X_train: Features for training
    - y_train: Target labels for training
    Returns:
    - linear_model: Trained linear regression model
    """
    # Initialize linear regression model
    linear_model = LinearRegression()
    # Train the model
    linear_model.fit(X_train, y_train)

    return linear_model
