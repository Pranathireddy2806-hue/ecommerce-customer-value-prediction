I analyzed the available project files and created a clean README for the project.

````markdown
# E-Commerce Customer Value Prediction

A machine learning project that predicts a customer's yearly spending based on customer behavior and engagement metrics. The project uses a Linear Regression model and evaluates performance using MAE, MSE, and R² score.

## Project Overview

This project analyzes customer data from an e-commerce store and predicts the `Yearly Amount Spent` using features such as:

- Avg. Session Length
- Time on App
- Time on Website
- Length of Membership

The model is trained using a train-test split and then used to estimate the value of a new customer based on similar inputs.

## Dataset

The dataset used in this project is:

- `data/Ecommerce Customers.csv`

It contains customer-related attributes and the target value:
- `Yearly Amount Spent`

The features selected for prediction are:

```python
X = df[['Avg. Session Length',
        'Time on App',
        'Time on Website',
        'Length of Membership']]
```

The target variable is:

```python
y = df['Yearly Amount Spent']
```

## Project Structure

```text
e-commerce_customer_value_prediction/
├── data/
│   └── Ecommerce Customers.csv
├── images/
│   └── customer_value_prediction.png
├── src/
│   └── customer_value_prediction.py
├── README.md
└── requirements.txt   (if added later)
```

## Model Used

This project uses:

- `LinearRegression` from `scikit-learn`

The workflow is:

1. Load the dataset
2. Split features and target
3. Train the model on training data
4. Predict on test data
5. Calculate evaluation metrics
6. Predict value for a new customer
7. Save the actual-vs-predicted plot

## Workflow Summary

The script performs the following tasks:

- Reads the CSV file
- Displays dataset information and column names
- Splits the data into training and testing sets
- Trains the regression model
- Evaluates model accuracy with:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score
- Makes a prediction for a sample new customer
- Generates a scatter plot comparing actual and predicted values
- Saves the chart as:
  - `images/customer_value_prediction.png`

## Dependencies

Install the required packages using:

```bash
pip install pandas matplotlib scikit-learn
```

## How to Run

From the project root, run:

```bash
python src/customer_value_prediction.py
```

## Example Output

The script prints:

- Dataset columns
- Data shape
- Model coefficients
- Intercept
- Prediction values
- Evaluation results like:

```text
MAE: ...
MSE: ...
R2 Score: ...
```

It also displays and saves a plot titled:

```text
Actual vs Predicted Customer Value
```

## Model Evaluation

The model is evaluated with the following metrics:

- MAE: measures the average absolute prediction error
- MSE: measures the average squared error
- R² Score: indicates how well the model explains variance in the target variable

A higher R² value generally indicates a better fit.

## Sample Prediction

The script includes a sample customer profile:

```python
new_customer = pd.DataFrame({
    'Avg. Session Length': [34.5],
    'Time on App': [12.2],
    'Time on Website': [39.5],
    'Length of Membership': [4.5]
})
```

The model predicts the yearly customer value for this profile and prints the result.

## Insights

This project demonstrates how customer engagement metrics can be used to estimate revenue contribution. In a real-world setting, the model can help businesses:

- Identify high-value customers
- Improve retention strategies
- Personalize marketing campaigns
- Forecast customer lifetime value

## Possible Improvements

To improve the model further, you can consider:

- Adding more relevant customer features
- Trying other regression models such as Random Forest or XGBoost
- Performing feature engineering
- Scaling features if needed
- Validating with cross-validation

## Author

This project was created for e-commerce customer value prediction using supervised learning.
