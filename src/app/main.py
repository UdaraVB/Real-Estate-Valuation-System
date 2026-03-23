import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Karachi House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def load_model_assets():
    
    model_path = "artifacts/model.joblib" 
    columns_path = "artifacts/feature_metadata.pkl"
    
    if os.path.exists(model_path) and os.path.exists(columns_path):
        model = joblib.load(model_path)
        model_columns = joblib.load(columns_path)
        return model, model_columns
    else:
        st.error("⚠️ Model artifacts not found! Please run the training pipeline first.")
        return None, None

model, model_columns = load_model_assets()

# === HELPER: FORMAT PRICE ===
def format_pkr(val):
    if val >= 10_000_000:
        return f"{val / 10_000_000:.2f} Crore"
    elif val >= 100_000:
        return f"{val / 100_000:.2f} Lakh"
    return f"{val:,.0f}"

# === USER INTERFACE ===
st.title("🏠 Karachi House Price Predictor")
st.markdown("""
Predict the market value of properties across Karachi using our **Random Forest** model.
""")

if model:
    with st.form("prediction_form"):
        st.subheader("Property Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            area = st.number_input("Area (Square Yards)", min_value=1, value=120)
            bedrooms = st.slider("Bedrooms", 1, 10, 3)
            
        with col2:
            location = st.selectbox("Location", [col.replace('Location_', '') for col in model_columns if 'Location_' in col])
            baths = st.slider("Baths", 1, 10, 2)

        submit = st.form_submit_button("💰 Estimate Price")

    if submit:
    
        input_df = pd.DataFrame(columns=model_columns)
        input_df.loc[0] = 0 
         
        input_df['Area'] = area
        input_df['Bedrooms'] = bedrooms
        input_df['Baths'] = baths
        
        loc_col = f"Location_{location}"
        if loc_col in input_df.columns:
            input_df[loc_col] = 1
            
        prediction = model.predict(input_df)[0]
          
        st.success(f"### Estimated Price: {format_pkr(prediction)}")
        st.info(f"Raw Value: PKR {prediction:,.0f}")

st.sidebar.header("About")
st.sidebar.info("""
This model was trained on historical Karachi real estate data. 
**Current R² Score:** 0.83
""")