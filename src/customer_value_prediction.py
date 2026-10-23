// ...existing code...
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the e-commerce customer dataset from the data folder
df = pd.read_csv("data/Ecommerce Customers.csv")

# Display the dataset columns for quick inspection
print(df.columns)
print("\n")

# Print information about the dataset (column names, dtypes, non-null counts)
print(df.info())

# Select the independent variables (features) used to predict yearly spending
X = df[['Avg. Session Length',
        'Time on App',
        'Time on Website',
        'Length of Membership']]

# Select the target variable to be predicted
y = df['Yearly Amount Spent']

# Show the first few rows of the feature matrix
print("\nX (Features):")
print(X.head())

# Show the first few values of the target variable
print("\ny (Target):")
print(y.head())

# Split the dataset into training and testing sets
# 80% for training and 20% for testing, with a fixed random seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Display shapes of the training and testing sets
print("Training Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Print confirmation that the model was trained
print("Model trained successfully!")

# Display coefficients and intercept for the learned linear equation
print("\nCoefficients:")
print(model.coef_)

print("\nIntercept:")
print(model.intercept_)

# Predict customer values for the test set
y_pred = model.predict(X_test)

# Show the first few predictions
print("\nPredictions:")
print(y_pred[:5])

# Evaluate the model using standard regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("\nMAE:")
print(mae)
print("\nMSE:")
print(mse)
print("\nR2 Score:")
print(r2)

# Create a sample new customer record to test prediction
new_customer = pd.DataFrame({
    'Avg. Session Length': [34.5],
    'Time on App': [12.2],
    'Time on Website': [39.5],
    'Length of Membership': [4.5]
})

# Predict the yearly amount spent for the new customer
predicted_value = model.predict(new_customer)

# Print the predicted value
print("\nPredicted Customer Value:")
print(predicted_value[0])

# Plot actual vs predicted values to visually evaluate the model
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue')

# Labels and title for the visualization
plt.xlabel("Actual Customer Value")
plt.ylabel("Predicted Customer Value")
plt.title("Actual vs Predicted Customer Value")

# Save the plot to the images folder
plt.savefig("images/customer_value_prediction.png")

# Display the plot
plt.show()