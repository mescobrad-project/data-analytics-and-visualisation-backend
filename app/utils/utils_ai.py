from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

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

def train_logistic_regression(X_train, y_train):

    logistic_model = LogisticRegression()
    # Train the model
    logistic_model.fit(X_train, y_train)

    return logistic_model

def train_SVC(X_train, y_train, kernel, probability, regularization):

    SVC_model = SVC(kernel=kernel, probability=probability, C=regularization)
    # Train the model
    SVC_model.fit(X_train, y_train)

    return SVC_model
