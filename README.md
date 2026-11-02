# 🛍️ E-commerce Customer Spend Predictor

A machine learning project that predicts how much a customer will spend per year on an e-commerce platform, using **Linear Regression**. It includes a command-line training script and an interactive, colourful **Streamlit dashboard**.

---

## 🔍 Overview

An online store wants to know which customers are likely to spend the most, so it can decide where to focus its effort: improve the mobile app, improve the website, or invest in customer loyalty.

This project:

- trains a **Linear Regression** model on customer behaviour data,
- evaluates it using **MAE, MSE, RMSE and R²**,
- predicts the **Yearly Amount Spent** for a new customer,
- shows everything in an **interactive dashboard**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **Live prediction** | Move sliders for 4 customer attributes and instantly see the predicted yearly spend |
| 🎯 **Customer segmentation** | Labels the customer as high-value, average or below-average compared to the dataset mean |
| 📈 **Model performance** | R², MAE, MSE and RMSE cards, actual vs predicted plot, residuals histogram |
| ⚖️ **Feature impact** | Bar chart of the model coefficients showing which feature drives spending most |
| 🔎 **Data explorer** | Dataset preview, correlation heatmap and feature vs spend scatter plots with a trendline |
| ⚙️ **Adjustable training** | Change the test-set size and random seed from the sidebar |
| 📤 **CSV upload** | Use your own dataset with the same column names |

---

## 🧰 Tech Stack

- **Python 3.9+**
- **pandas / NumPy**: data handling
- **scikit-learn**: model training and metrics
- **Matplotlib**: static plot saved by the training script
- **Plotly**: interactive charts in the dashboard
- **statsmodels**: trendlines in Plotly scatter plots
- **Streamlit**: web interface

---

## 📊 Dataset

File: `data/Ecommerce Customers.csv`

| Column | Type | Used in model? | Description |
|---|---|---|---|
| Email | text | ❌ | Customer email |
| Address | text | ❌ | Customer address |
| Avatar | text | ❌ | Avatar colour name |
| **Avg. Session Length** | float | ✅ Feature | Average in-store style session length (minutes) |
| **Time on App** | float | ✅ Feature | Average minutes per day on the mobile app |
| **Time on Website** | float | ✅ Feature | Average minutes per day on the website |
| **Length of Membership** | float | ✅ Feature | Number of years as a member |
| **Yearly Amount Spent** | float | 🎯 Target | Money spent per year (what we predict) |

Text columns are not used because they carry no numeric signal for Linear Regression.

---

## 📁 Project Structure

```
e-commerce_customer_value/
│
├── data/
│   └── Ecommerce Customers.csv        # Dataset
│
├── images/
│   └── customer_value_prediction.png  # Plot saved by the training script
│
├── src/
│   └── customer_value_prediction.py   # Command-line training + evaluation script
│
├── app.py                             # Streamlit dashboard
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Files Git should ignore
└── README.md                          # Project documentation
```

> `requirements.txt` must stay in the project root so `pip install -r requirements.txt` and Streamlit Cloud can find it.

---

## ⚙️ How It Works

```
CSV data ─► select features (X) and target (y) ─► 80/20 train-test split
        ─► train LinearRegression ─► predict on test set
        ─► evaluate (MAE, MSE, RMSE, R²) ─► predict for new customers
```

1. **Load** the CSV with pandas.
2. **Select features**: `Avg. Session Length`, `Time on App`, `Time on Website`, `Length of Membership`.
3. **Split** the data into 80% training and 20% testing (`random_state=42` for reproducibility).
4. **Train** a `LinearRegression` model.
5. **Evaluate** it with regression metrics.
6. **Predict** the yearly spend of a new customer.

The model learns an equation of the form:

```
Yearly Spend = intercept + (b1 × Session Length) + (b2 × Time on App)
                         + (b3 × Time on Website) + (b4 × Length of Membership)
```

Each coefficient tells you how many dollars of yearly spend change when that feature increases by 1 unit while the others stay the same.

---

## 🛠️ Installation

### Prerequisites

- Python 3.9 or newer → check with `python --version`
- Git → check with `git --version`

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
source venv/bin/activate          # Mac / Linux
venv\Scripts\activate             # Windows (Command Prompt / VS Code terminal)
.\venv\Scripts\Activate.ps1       # Windows PowerShell

# 4. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

> Always run commands from the **project root folder** so the relative paths (`data/...`, `images/...`) work.

### Option 1: Interactive dashboard (recommended)

```bash
streamlit run app.py
```

Your browser opens at **http://localhost:8501**. Press `Ctrl + C` in the terminal to stop it.

### Option 2: Command-line training script

```bash
python src/customer_value_prediction.py
```

This prints the dataset info, model coefficients, evaluation metrics and a sample prediction. It also saves the actual vs predicted plot to `images/customer_value_prediction.png`.

---

## 📐 Model Evaluation

| Metric | Meaning | Better is |
|---|---|---|
| **MAE** (Mean Absolute Error) | Average size of the prediction error, in dollars | Lower |
| **MSE** (Mean Squared Error) | Average squared error; penalises big mistakes | Lower |
| **RMSE** | Square root of MSE, in dollars | Lower |
| **R² Score** | Share of the variation in spending explained by the model (1.0 = perfect) | Higher |

Fill in your own results after running the project:

| Metric | Value |
|---|---|
| MAE | _your value_ |
| MSE | _your value_ |
| RMSE | _your value_ |
| R² | _your value_ |

---

## 🖥️ Dashboard Guide

### Sidebar
- **Upload a CSV**: use your own data (must contain the same column names).
- **Test set size**: fraction of data held back for testing (10% to 50%).
- **Random seed**: change it to see how the split affects results.

### 🔮 Predict tab
1. Set the four sliders. They start at the dataset averages.
2. The predicted yearly spend and the difference from the average appear instantly.
3. The gauge is coloured by segment: red (below average), yellow (average), green (high value).

### 📈 Model Performance tab
- Metric cards for R², MAE, MSE and RMSE.
- **Actual vs Predicted**: points close to the red dashed line mean accurate predictions.
- **Residuals**: a bell shape centred at 0 means the errors are well behaved.
- **Coefficients**: the longest bar is the feature with the biggest effect on spending.

### 🔎 Data Explorer tab
- First 20 rows of the data.
- **Correlation heatmap**: shows which features move together with spending.
- **Scatter plot with trendline**: choose any feature from the dropdown.

---

## 🌐 Deployment (Streamlit Community Cloud)

1. Push the project to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** and select your repository.
4. Set **Branch** to `main` and **Main file path** to `app.py`.
5. Click **Deploy**.

Make sure `requirements.txt` is in the root and the CSV file is committed, otherwise the deployed app can't load the data.

---

## 🧯 Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: data/Ecommerce Customers.csv` | Run from the project root, and check the file name and folder |
| `FileNotFoundError` when saving the plot | Make sure the `images/` folder exists (`mkdir images`) |
| `ModuleNotFoundError` | Activate your virtual environment, then run `pip install -r requirements.txt` |
| `streamlit: command not found` | Use `python -m streamlit run app.py` |
| PowerShell blocks activation script | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt |
| Red squiggles in VS Code | Select the right interpreter: `Ctrl+Shift+P` → *Python: Select Interpreter* → pick `venv` |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |

---

## 🚀 Future Improvements

- Compare more models (Ridge, Lasso, Random Forest, Gradient Boosting)
- Add cross-validation and hyperparameter tuning
- Add feature scaling and show standardised coefficients
- Batch predictions: upload a CSV and download predictions
- Save the trained model with `joblib` instead of retraining on every run
- Add unit tests and a GitHub Actions workflow

---

