import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("house_price_model.pkl")

# Page Title
st.title("🏠 AI-powered House Price Prediction System using Machine Learning and Random Forest Regression")

st.markdown(
    "Predict house prices using a Machine Learning model trained on the Kaggle House Prices dataset."
)

# Sidebar
st.sidebar.header("Model Information")
st.sidebar.success("Model: Random Forest Regressor")
st.sidebar.success("R² Score: 0.88")

# User Inputs
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

# Prediction
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "GarageCars": [garage_cars],
        "FullBath": [full_bath],
        "BedroomAbvGr": [bedrooms],
        "YearBuilt": [year_built]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"🏡 Estimated House Price: ${prediction:,.0f}"
    )

    st.write("### Property Summary")
    st.write(f"**Overall Quality:** {overall_qual}/10")
    st.write(f"**Living Area:** {gr_liv_area:,} sq ft")
    st.write(f"**Garage Capacity:** {garage_cars} cars")
    st.write(f"**Bathrooms:** {full_bath}")
    st.write(f"**Bedrooms:** {bedrooms}")
    st.write(f"**Year Built:** {year_built}")