import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("data/Ecommerce Customers.csv")
print(df.columns)
print("\n")
print(df.info())
X = df[['Avg. Session Length',
        'Time on App',
        'Time on Website',
        'Length of Membership']]

y = df['Yearly Amount Spent']

print("\nX (Features):")
print(X.head())

print("\ny (Target):")
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)
model = LinearRegression()

model.fit(X_train, y_train)

print("Model trained successfully!")
print("\nCoefficients:")
print(model.coef_)

print("\nIntercept:")
print(model.intercept_)
y_pred = model.predict(X_test)
print("\nPredictions:")
print(y_pred[:5])
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMAE:")
print(mae)
print("\nMSE:")
print(mse)
print("\nR2 Score:")
print(r2)

new_customer = pd.DataFrame({
    'Avg. Session Length': [34.5],
    'Time on App': [12.2],
    'Time on Website': [39.5],
    'Length of Membership': [4.5]
})

predicted_value = model.predict(new_customer)

print("\nPredicted Customer Value:")
print(predicted_value[0])

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue')

plt.xlabel("Actual Customer Value")
plt.ylabel("Predicted Customer Value")

plt.title("Actual vs Predicted Customer Value")

plt.savefig("images/customer_value_prediction.png")

plt.show()


