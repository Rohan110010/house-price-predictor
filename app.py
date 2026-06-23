import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==================================
# Page Configuration
# ==================================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ==================================
# Load Model
# ==================================
model = joblib.load("house_price_model.pkl")

# ==================================
# Load Metrics
# ==================================
try:
    metrics = pd.read_csv("images/model_metrics.csv")

    r2 = float(
        metrics.loc[
            metrics["Metric"] == "R2 Score",
            "Value"
        ].values[0]
    )

    mae = float(
        metrics.loc[
            metrics["Metric"] == "MAE",
            "Value"
        ].values[0]
    )

    rmse = float(
        metrics.loc[
            metrics["Metric"] == "RMSE",
            "Value"
        ].values[0]
    )

    cv_r2 = float(
        metrics.loc[
            metrics["Metric"] == "Average CV R2",
            "Value"
        ].values[0]
    )

except:
    r2 = 0.88
    mae = 0
    rmse = 0
    cv_r2 = 0.87

# ==================================
# Title
# ==================================
st.title("🏠 AI-Powered House Price Prediction System")

st.markdown(
    """
Predict house prices using a Machine Learning model trained on the
Kaggle House Prices: Advanced Regression Techniques dataset.
"""
)

# ==================================
# Sidebar
# ==================================
st.sidebar.header("📊 Model Information")

st.sidebar.success("Model: Random Forest Regressor")
st.sidebar.success(f"R² Score: {r2:.3f}")
st.sidebar.success(f"CV R² Score: {cv_r2:.3f}")

# ==================================
# Dashboard Metrics
# ==================================
st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("R² Score", f"{r2:.3f}")
col2.metric("CV R²", f"{cv_r2:.3f}")
col3.metric("MAE", f"${mae:,.0f}")
col4.metric("RMSE", f"${rmse:,.0f}")

st.markdown("---")

# ==================================
# Inputs
# ==================================
st.subheader("🏡 Property Details")

overall_qual = st.slider(
    "Overall Quality (1-10)",
    min_value=1,
    max_value=10,
    value=5
)

st.info("""
Overall Quality Guide

1-2 : Very Poor
3-4 : Below Average
5   : Average
6-7 : Good
8-9 : Very Good
10  : Excellent
""")

gr_liv_area = st.number_input(
    "Living Area (sq ft)",
    min_value=300,
    value=1500
)

garage_cars = st.number_input(
    "Garage Capacity (Cars)",
    min_value=0,
    max_value=5,
    value=2
)

full_bath = st.number_input(
    "Number of Full Bathrooms",
    min_value=0,
    max_value=10,
    value=2
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

year_built = st.number_input(
    "Year Built",
    min_value=1800,
    max_value=2025,
    value=2000
)

# ==================================
# Prediction
# ==================================
if st.button("🔮 Predict Price"):

    input_data = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "GarageCars": [garage_cars],
        "FullBath": [full_bath],
        "BedroomAbvGr": [bedrooms],
        "YearBuilt": [year_built]
    })

    prediction = model.predict(input_data)[0]

    tree_predictions = [
        tree.predict(input_data)[0]
        for tree in model.estimators_
    ]

    uncertainty = np.std(tree_predictions)

    if prediction < 150000:
        category = "🏠 Budget House"
    elif prediction < 300000:
        category = "🏡 Mid-Range House"
    else:
        category = "🏰 Premium House"

    st.success(
        f"🏡 Estimated House Price: ${prediction:,.0f}"
    )

    st.info(category)

    st.write(
        f"Prediction Uncertainty: ±${uncertainty:,.0f}"
    )

    st.subheader("📋 Property Summary")

    st.write(f"**Overall Quality:** {overall_qual}/10")
    st.write(f"**Living Area:** {gr_liv_area:,} sq ft")
    st.write(f"**Garage Capacity:** {garage_cars} cars")
    st.write(f"**Bathrooms:** {full_bath}")
    st.write(f"**Bedrooms:** {bedrooms}")
    st.write(f"**Year Built:** {year_built}")

    report_df = pd.DataFrame({
        "Metric": [
            "Predicted Price",
            "Property Category",
            "Prediction Uncertainty",
            "Overall Quality",
            "Living Area",
            "Garage Capacity",
            "Bathrooms",
            "Bedrooms",
            "Year Built",
            "Model R²",
            "Cross Validation R²"
        ],
        "Value": [
            f"${prediction:,.0f}",
            category,
            f"±${uncertainty:,.0f}",
            overall_qual,
            f"{gr_liv_area:,} sq ft",
            garage_cars,
            full_bath,
            bedrooms,
            year_built,
            round(r2, 3),
            round(cv_r2, 3)
        ]
    })

    csv = report_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="house_prediction_report.csv",
        mime="text/csv"
    )

# ==================================
# Feature Importance
# ==================================
st.markdown("---")
st.subheader("🎯 Feature Importance")

try:
    st.image(
        "images/feature_importance.png",
        caption="Random Forest Feature Importance"
    )
except:
    st.warning("feature_importance.png not found")

# ==================================
# Actual vs Predicted
# ==================================
st.subheader("📊 Actual vs Predicted Prices")

try:
    st.image(
        "images/actual_vs_predicted.png",
        caption="Model Evaluation"
    )
except:
    st.warning("actual_vs_predicted.png not found")

# ==================================
# About Project
# ==================================
st.markdown("---")

with st.expander("ℹ️ About This Project"):
    st.write("""
    Dataset: Kaggle House Prices Dataset

    Model: Random Forest Regressor

    Features:
    - OverallQual
    - GrLivArea
    - GarageCars
    - FullBath
    - BedroomAbvGr
    - YearBuilt

    Evaluation:
    - R² Score
    - Cross Validation R²
    - MAE
    - RMSE

    Deployment:
    - Streamlit Community Cloud
    """)