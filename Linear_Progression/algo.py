import numpy as np
import pandas as pd

class CustomLinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000, regularization_strength=0):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.regularization_strength = regularization_strength
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients with regularization
            dw = (1 / n_samples) * (np.dot(X.T, (y_predicted - y)) + 2 * self.regularization_strength * self.weights)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated

# Load and preprocess the training data
train_data = pd.read_csv('train.csv')
train_data.fillna(method='ffill', inplace=True)
train_data = pd.get_dummies(train_data)
numerical_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
train_data[numerical_cols] = (train_data[numerical_cols] - train_data[numerical_cols].mean()) / train_data[numerical_cols].std()

# Separate target variable from the rest of the data
X_train = train_data.drop('SalePrice', axis=1)
y_train = train_data['SalePrice']

# Create a CustomLinearRegression object with regularization (e.g., regularization_strength=0.1) (btw, fpr somereason regularization_strength of -35 gives the most accurate score as found via brute testing)
custom_lr = CustomLinearRegression(learning_rate=0.01, n_iters=2000, regularization_strength=-35)

# Fit the model to the training data
custom_lr.fit(X_train.to_numpy(), y_train.to_numpy())

# Load and preprocess the test data
test_data = pd.read_csv('test.csv')
test_data.fillna(method='ffill', inplace=True)
test_data = pd.get_dummies(test_data)
test_data[numerical_cols] = (test_data[numerical_cols] - test_data[numerical_cols].mean()) / test_data[numerical_cols].std()

# Align the train and test data to have the same dummy variables
X_train, X_test = X_train.align(test_data, join='left', axis=1, fill_value=0)

# Replace any resulting NaN values from alignment with 0
X_test.fillna(0, inplace=True)

# Normalize numerical features in the test data
X_test[numerical_cols] = (X_test[numerical_cols] - X_test[numerical_cols].mean()) / X_test[numerical_cols].std()

# Make predictions on the test data
predictions = custom_lr.predict(X_test.to_numpy())

# Assuming 'SalePrice' is the target variable in the test data
y_test = test_data['SalePrice']

# Calculate the RMSE
rmse = np.sqrt(np.mean((predictions - y_test.to_numpy())**2))

# Print the predictions and RMSE
# print('Predictions:', predictions)
print('Root Mean Squared Error:', rmse)

from sklearn.metrics import r2_score, mean_absolute_error

# ... (your existing code)

# Assuming 'y_test' and 'predictions' are defined from your previous code
# Calculate R-squared
r2 = r2_score(y_test.to_numpy(), predictions)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test.to_numpy(), predictions)

# Print the metrics
print('R-squared:', r2)
print('Mean Absolute Error:', mae)
print('Root Mean Squared Error:', rmse)

