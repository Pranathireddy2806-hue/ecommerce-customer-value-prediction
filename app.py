import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- config
st.set_page_config(page_title="Customer Spend Predictor", page_icon="🛍️", layout="wide")

DATA_PATH = "data/Ecommerce Customers.csv"
FEATURES = ["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]
TARGET = "Yearly Amount Spent"

st.markdown(
    """
    <style>
    .hero {background: linear-gradient(90deg,#6a11cb 0%,#2575fc 100%);
           padding: 1.6rem 2rem; border-radius: 16px; color: white; margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 2.1rem;}
    .hero p {margin: .3rem 0 0 0; opacity: .9;}
    div[data-testid="stMetric"] {background: linear-gradient(135deg,#f8f9ff,#eef3ff);
           border: 1px solid #dfe6ff; padding: 14px; border-radius: 14px;}
    div[data-testid="stMetric"] * {color: #1f2a5a !important;}
    </style>
    <div class="hero"><h1>🛍️ Customer Spend Predictor</h1>
    <p>Linear Regression model that predicts a customer's yearly spending.</p></div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- data + model
@st.cache_data
def load_data(source):
    return pd.read_csv(source)


@st.cache_resource
def train_model(df, test_size, seed):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": r2_score(y_test, y_pred),
    }
    return model, X_test, y_test, y_pred, metrics


with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload a CSV (optional)", type="csv")
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
    seed = st.number_input("Random seed", value=42, step=1)

try:
    df = load_data(uploaded if uploaded else DATA_PATH)
except FileNotFoundError:
    st.error(f"Couldn't find `{DATA_PATH}`. Put the CSV in the `data/` folder or upload it in the sidebar.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The dataset is missing these columns: {missing}")
    st.stop()

model, X_test, y_test, y_pred, metrics = train_model(df, test_size, int(seed))

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Model Performance", "🔎 Data Explorer"])

# ---------------------------------------------------------------- tab 1: predict
with tab1:
    st.subheader("Enter customer details")
    cols = st.columns(2)
    inputs = {}
    for i, f in enumerate(FEATURES):
        lo, hi, mean = float(df[f].min()), float(df[f].max()), float(df[f].mean())
        inputs[f] = cols[i % 2].slider(f, round(lo, 1), round(hi, 1), round(mean, 1), 0.1)

    new_customer = pd.DataFrame([inputs])
    prediction = float(model.predict(new_customer)[0])
    avg = float(df[TARGET].mean())

    left, right = st.columns([1, 1.2])
    with left:
        st.metric("Predicted yearly spend", f"${prediction:,.2f}", f"{prediction - avg:+,.2f} vs average")
        st.caption(f"Average customer spends ${avg:,.2f} per year.")
        if prediction > avg * 1.1:
            st.success("🌟 High-value customer")
        elif prediction < avg * 0.9:
            st.warning("📉 Below-average customer")
        else:
            st.info("👍 Average customer")
    with right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            delta={"reference": avg},
            number={"prefix": "$"},
            gauge={"axis": {"range": [0, float(df[TARGET].max()) * 1.2]},
                   "bar": {"color": "#6a11cb"},
                   "steps": [{"range": [0, avg * 0.9], "color": "#ffe0e0"},
                             {"range": [avg * 0.9, avg * 1.1], "color": "#fff3cd"},
                             {"range": [avg * 1.1, float(df[TARGET].max()) * 1.2], "color": "#d4f8e0"}]},
        ))
        fig.update_layout(height=300, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- tab 2: performance
with tab2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{metrics['R2']:.4f}")
    c2.metric("MAE", f"{metrics['MAE']:.2f}")
    c3.metric("MSE", f"{metrics['MSE']:.2f}")
    c4.metric("RMSE", f"{metrics['RMSE']:.2f}")

    a, b = st.columns(2)
    with a:
        fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                         title="Actual vs Predicted", color=np.abs(y_test - y_pred),
                         color_continuous_scale="Viridis")
        lo, hi = float(y_test.min()), float(y_test.max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color="red", dash="dash"), name="Perfect fit"))
        fig.update_coloraxes(colorbar_title="Error")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.histogram(x=y_test - y_pred, nbins=30, title="Residuals (Actual − Predicted)",
                           color_discrete_sequence=["#2575fc"])
        fig.update_layout(xaxis_title="Residual", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    coef = pd.DataFrame({"Feature": FEATURES, "Coefficient": model.coef_}).sort_values("Coefficient")
    fig = px.bar(coef, x="Coefficient", y="Feature", orientation="h", color="Coefficient",
                 color_continuous_scale="Sunset", title="Effect of +1 unit of each feature on yearly spend ($)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Intercept: {model.intercept_:.2f}")

# ---------------------------------------------------------------- tab 3: explorer
with tab3:
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")

    a, b = st.columns(2)
    with a:
        corr = df[FEATURES + [TARGET]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Correlation heatmap", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        feat = st.selectbox("Feature to plot against yearly spend", FEATURES, index=3)
        fig = px.scatter(df, x=feat, y=TARGET, color="Length of Membership",
                         color_continuous_scale="Plasma", trendline="ols",
                         title=f"{feat} vs {TARGET}")
        st.plotly_chart(fig, use_container_width=True)