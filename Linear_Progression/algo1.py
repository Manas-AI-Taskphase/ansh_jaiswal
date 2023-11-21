import numpy as np
import pandas as pd

class CustomLinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
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
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated

# Load the training data
train_data = pd.read_csv('train.csv')

# Fill NA/NaN values using the specified method
train_data.fillna(method='ffill', inplace=True)

# Convert categorical variable into dummy/indicator variables
train_data = pd.get_dummies(train_data)

# Separate target variable from the rest of the data
X_train = train_data.drop('SalePrice', axis=1)
y_train = train_data['SalePrice']

# Normalize numerical features
numerical_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
X_train[numerical_cols] = (X_train[numerical_cols] - X_train[numerical_cols].mean()) / X_train[numerical_cols].std()

# Create a CustomLinearRegression object
custom_lr = CustomLinearRegression(learning_rate=0.01, n_iters=1000)

# Fit the model to the training data
custom_lr.fit(X_train.to_numpy(), y_train.to_numpy())

# Load the test data with SalePrices
test_data_with_prices = pd.read_csv('test.csv')

# Fill NA/NaN values using the specified method
test_data_with_prices.fillna(method='ffill', inplace=True)

# Convert categorical variable into dummy/indicator variables
test_data_with_prices = pd.get_dummies(test_data_with_prices)

# Separate the SalePrice from the test data
X_test = test_data_with_prices.drop('SalePrice', axis=1)
y_test = test_data_with_prices['SalePrice']

# Align the train and test data to have the same dummy variables
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# Replace any resulting NaN values from alignment with 0
X_test.fillna(0, inplace=True)

# Normalize numerical features in the test data
X_test[numerical_cols] = (X_test[numerical_cols] - X_test[numerical_cols].mean()) / X_test[numerical_cols].std()

# Make predictions on the test data
predictions = custom_lr.predict(X_test.to_numpy())

# Compute the RMSE
rmse = np.sqrt(np.mean((predictions - y_test.to_numpy())**2))

# Calculate R-squared manually
mean_y = np.mean(y_test.to_numpy())
total_sum_of_squares = np.sum((y_test.to_numpy() - mean_y)**2)
residual_sum_of_squares = np.sum((y_test.to_numpy() - predictions)**2)
r2_manual = 1 - (residual_sum_of_squares / total_sum_of_squares)

# Calculate MAE manually
mae_manual = np.mean(np.abs(y_test.to_numpy() - predictions))

# Print the metrics
print('Root Mean Squared Error:', rmse)
print('Manually Calculated R-squared:', r2_manual)
print('Manually Calculated Mean Absolute Error:', mae_manual)

