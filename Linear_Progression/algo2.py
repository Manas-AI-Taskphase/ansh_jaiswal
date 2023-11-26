import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CustomLinearRegression:
   def __init__(self, learning_rate=0.01, n_iters=1000):
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

# Lists to store metrics for each epoch
epochs_list = []
train_rmse_list = []
test_rmse_list = []
r2_list = []
mae_list = []

# Load the test data with SalePrices
test_data = pd.read_csv('test.csv')

# Fill NA/NaN values using the specified method
test_data.fillna(method='ffill', inplace=True)

# Convert categorical variable into dummy/indicator variables
test_data = pd.get_dummies(test_data)

# Separate the SalePrice from the test data
X_test = test_data.drop('SalePrice', axis=1)
y_test = test_data['SalePrice']

# Normalize numerical features in the test data
X_test[numerical_cols] = (X_test[numerical_cols] - X_test[numerical_cols].mean()) / X_test[numerical_cols].std()

# Training loop
for epoch in range(custom_lr.n_iters):
   # Fit the model to the training data
   custom_lr.fit(X_train.to_numpy(), y_train.to_numpy())

   # Make predictions on the training data
   train_predictions = custom_lr.predict(X_train.to_numpy())

   # Compute the training metrics
   train_rmse = np.sqrt(np.mean((train_predictions - y_train.to_numpy())**2))
   mean_y_train = np.mean(y_train.to_numpy())
   total_sum_of_squares_train = np.sum((y_train.to_numpy() - mean_y_train)**2)
   residual_sum_of_squares_train = np.sum((y_train.to_numpy() - train_predictions)**2)
   r2_train = 1 - (residual_sum_of_squares_train / total_sum_of_squares_train)
   mae_train = np.mean(np.abs(y_train.to_numpy() - train_predictions))

   # Store training metrics
   train_rmse_list.append(train_rmse)
   r2_list.append(r2_train)
   mae_list.append(mae_train)

   # Make predictions on the test data
   test_predictions = custom_lr.predict(X_test.to_numpy())

   # Compute the test metrics
   test_rmse = np.sqrt(np.mean((test_predictions - y_test.to_numpy())**2))
   mean_y_test = np.mean(y_test.to_numpy())
   total_sum_of_squares_test = np.sum((y_test.to_numpy() - mean_y_test)**2)
   residual_sum_of_squares_test = np.sum((y_test.to_numpy() - test_predictions)**2)
   r2_test = 1 - (residual_sum_of_squares_test / total_sum_of_squares_test)
   mae_test = np.mean(np.abs(y_test.to_numpy() - test_predictions))

   # Store test metrics
   test_rmse_list.append(test_rmse)

   # Print metrics for every 100 epochs
   if epoch % 100 == 0:
       print(f'Epoch {epoch}/{custom_lr.n_iters} - Train RMSE: {train_rmse:.4f} - Test RMSE: {test_rmse:.4f} - Train R2: {r2_train:.4f} - Test R2: {r2_test:.4f} - Train MAE: {mae_train:.4f} - Test MAE: {mae_test:.4f}')

   epochs_list.append(epoch)
    # Plot the metrics over epochs

   plt.plot(epochs_list, train_rmse_list, label='Train RMSE')
   plt.plot(epochs_list, test_rmse_list, label='Test RMSE')
   plt.plot(epochs_list, r2_list, label='R2')
   plt.plot(epochs_list, mae_list, label='MAE')
   plt.xlabel('Epochs')
   plt.ylabel('Metrics')
   plt.legend()
   plt.show()

