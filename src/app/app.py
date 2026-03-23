import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

#PAGE CONFIGURATION
st.set_page_config(
    page_title="Karachi House Predictor",
    page_icon="🏠",
    layout="centered"
)
@st.cache_resource
def load_assets():
    """Loads the model and the feature list to ensure encoding matches training."""
    model_path = "artifacts/model.joblib"
    columns_path = "artifacts/feature_metadata.pkl"
    
    if os.path.exists(model_path) and os.path.exists(columns_path):
        model = joblib.load(model_path)
        feature_cols = joblib.load(columns_path)
        return model, feature_cols
    else:
        st.error("❌ Model artifacts not found. Please run the training pipeline first!")
        return None, None

model, feature_cols = load_assets()


def format_pkr(amount):
    """Converts raw numbers into readable Pakistani currency format."""
    if amount >= 10_000_000:
        return f"{amount / 10_000_000:.2f} Crore"
    elif amount >= 100_000:
        return f"{amount / 100_000:.2f} Lakh"
    return f"{amount:,.0f} PKR"

#USER INTERFACE
st.title("🔮 Karachi House Price Predictor")
st.markdown("Enter property details below to get an AI-powered valuation.")

if model and feature_cols:
    with st.form("prediction_form"):
        st.subheader("Property Specs")
        
        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input("Area (Square Yards)", min_value=50, max_value=2000, value=120)
            bedrooms = st.slider("Bedrooms", 1, 10, 3)
        with col2:
            baths = st.slider("Baths", 1, 10, 2)
            
            locations = [c.replace("Location_", "") for c in feature_cols if c.startswith("Location_")]
            selected_location = st.selectbox("Select Location", sorted(locations))

        submit_button = st.form_submit_button(label="💰 Calculate Market Value")


    if submit_button:

        input_data = pd.DataFrame(0, index=[0], columns=feature_cols)
        
   
        input_data["Area"] = area
        input_data["Bedrooms"] = bedrooms
        input_data["Baths"] = baths

        loc_col = f"Location_{selected_location}"
        if loc_col in input_data.columns:
            input_data[loc_col] = 1
        
       
        try:
            prediction = model.predict(input_data)[0]
            
   
            st.divider()
            st.balloons()
            st.subheader("Estimated Market Price:")
            st.write(f"### :green[{format_pkr(prediction)}]")
            st.caption(f"Exact Value: PKR {prediction:,.0f}")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.sidebar.image("https://img.icons8.com/clouds/200/home.png")
st.sidebar.header("Model Intelligence")
st.sidebar.write("🟢 **Status:** Operational")
st.sidebar.write("📊 **R² Score:** 0.83")
st.sidebar.info("This model uses Random Forest Regression to analyze Karachi real estate trends.")