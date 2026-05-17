import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Battery RUL Predictor",
    page_icon="🔋",
    layout="wide"
)

# 2. Load the Random Forest Model and Power Transformer
@st.cache_resource
def load_artifacts():
    model_filename = 'battery_rul_rf_model.pkl' 
    transformer_filename = 'power_transformer.pkl'
    
    model = joblib.load(model_filename)
    transformer = joblib.load(transformer_filename)
    return model, transformer

try:
    model, transformer = load_artifacts()
except FileNotFoundError:
    st.error("""
    ⚠️ **Required Artifact Files Not Found!** Ensure both **`battery_rul_rf_model.pkl`** and **`power_transformer.pkl`** are saved inside this exact folder.
    """)
    st.stop()

# 3. App Headers
st.title("🔋 Battery Remaining Useful Life (RUL) Predictor")
st.markdown("""
This interactive interface uses your trained **Random Forest Regressor** to estimate the number of operational life cycles 
remaining for a battery based on real-time diagnostic parameters.
""")

st.write("---")

# 4. Interactive Input Layout
st.subheader("⚙️ Input Battery Telemetry Features")

col1, col2, col3 = st.columns(3)

with col1:
    # Kept for reference UI tracking, but excluded from data sent to the model
    cycle_index = st.number_input("Cycle Index", min_value=1.0, value=100.0, step=1.0, help="Current operational cycle number.")
    discharge_time = st.number_input("Discharge Time (s)", min_value=0.0, value=2500.0, step=10.0)
    decrement_v = st.number_input("Decrement 3.6-3.4V (s)", min_value=0.0, value=1100.0, step=5.0)

with col2:
    max_v_dischar = st.number_input("Max. Voltage Discharge (V)", min_value=0.0, max_value=6.0, value=3.67, step=0.01)
    min_v_charg = st.number_input("Min. Voltage Charge (V)", min_value=0.0, max_value=6.0, value=3.21, step=0.01)
    time_415v = st.number_input("Time at 4.15V (s)", min_value=0.0, value=5400.0, step=10.0)

with col3:
    time_cc = st.number_input("Time Constant Current (s)", min_value=0.0, value=6700.0, step=10.0)
    charging_time = st.number_input("Charging Time (s)", min_value=0.0, value=10700.0, step=10.0)

st.write("---")

# 5. Prediction Engine
if st.button("🚀 Calculate Remaining Useful Life", type="primary"):
    
    # FIX: Excluded 'Cycle_Index' from the dictionary below. 
    # The remaining 7 keys perfectly match the feature strings and ordering from your notebook.
    input_df = pd.DataFrame([{
        'Discharge Time (s)': discharge_time,
        'Decrement 3.6-3.4V (s)': decrement_v,
        'Max. Voltage Dischar. (V)': max_v_dischar,
        'Min. Voltage Charg. (V)': min_v_charg,
        'Time at 4.15V (s)': time_415v,
        'Time constant current (s)': time_cc,
        'Charging time (s)': charging_time
    }])
    
    try:
        # Preprocess the input data using the loaded PowerTransformer
        transformed_features = transformer.transform(input_df)
        
        # Run transformed features through the Random Forest Regressor
        prediction = model.predict(transformed_features)
        
        # Post-process target result (RUL can't drop below 0)
        predicted_rul = int(max(0, prediction[0]))
        
        # 6. Display Operational Metrics
        st.subheader("📊 Output Prediction Summary")
        
        st.metric(
            label="Estimated Remaining Useful Life", 
            value=f"{predicted_rul} Cycles"
        )
        
        if predicted_rul > 500:
            st.success(f"✅ **Battery Status: Healthy.** At Cycle {int(cycle_index)}, the component has a significant operational lifetime remaining.")
        elif predicted_rul > 100:
            st.warning(f"⚠️ **Battery Status: Moderate Wear.** At Cycle {int(cycle_index)}, normal baseline degradation is detected. Monitor regularly.")
        else:
            st.error(f"🚨 **Battery Status: Critical.** At Cycle {int(cycle_index)}, the component is near its End of Life (EOL). Schedule immediate replacement.")
            
    except Exception as e:
        st.error(f"An error occurred during transformation or prediction: {e}")