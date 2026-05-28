import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise Battery Prognostics Platform",
    page_icon="🔋",
    layout="wide"
)

# 2. Secure ML Artifact Loading
@st.cache_resource
def load_artifacts():
    model_filename = 'battery_rul_et_model.pkl' 
    transformer_filename = 'power_transformer.pkl'
    
    model = joblib.load(model_filename)
    transformer = joblib.load(transformer_filename)
    return model, transformer

try:
    model, transformer = load_artifacts()
except FileNotFoundError:
    st.error("""
    ⚠️ **Required Artifact Files Not Found!** Ensure both **`battery_rul_et_model.pkl`** and **`power_transformer.pkl`** are saved inside this exact directory.
    """)
    st.stop()

# 3. Sidebar UI - Navigation & Layout Setup
st.sidebar.title("🎮 Dashboard Controller")
app_mode = st.sidebar.radio("Select Application Mode:", ["Single Unit Diagnostics", "Batch CSV Processing"])

# Master strict column order required by the model
feature_columns = [
    'Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Max. Voltage Dischar. (V)',
    'Min. Voltage Charg. (V)', 'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)'
]

if app_mode == "Single Unit Diagnostics":
    st.sidebar.write("---")
    st.sidebar.subheader("💡 Load Simulation Presets")
    preset = st.sidebar.selectbox(
        "Choose a telemetry profile to auto-fill:",
        ["Manual Tweaking 🛠️", "Fresh Battery Profile (High RUL) 🟢", "Severely Degraded Profile (Critical EOL) 🔴"]
    )
    
    # Preset definitions derived from physical battery decay cycles
    if "Fresh Battery" in preset:
        init_cycle = 15.0
        init_discharge = 3300.0
        init_decrement = 1250.0
        init_max_v = 3.69
        init_min_v = 3.18
        init_time_415 = 5600.0
        init_time_cc = 6900.0
        init_charging = 11000.0
    elif "Severely Degraded" in preset:
        init_cycle = 850.0
        init_discharge = 1200.0
        init_decrement = 400.0
        init_max_v = 3.52
        init_min_v = 3.39
        init_time_415 = 2100.0
        init_time_cc = 3400.0
        init_charging = 5800.0
    else:  # Manual baseline defaults
        init_cycle = 100.0
        init_discharge = 2500.0
        init_decrement = 1100.0
        init_max_v = 3.67
        init_min_v = 3.21
        init_time_415 = 5400.0
        init_time_cc = 6700.0
        init_charging = 10700.0

    st.sidebar.write("---")
    st.sidebar.subheader("⚙️ Fine-Tune Parameters")
    
    # Visual Grouping 1: Core Lifecycle Reference
    with st.sidebar.container(border=True):
        st.markdown("📋 **Asset Metadata**")
        cycle_index = st.number_input("Cycle Index Reference", min_value=1.0, value=init_cycle, step=1.0)
    
    # Visual Grouping 2: Operational Time Frameworks
    with st.sidebar.container(border=True):
        st.markdown("⏱️ **Temporal Telemetry (Seconds)**")
        discharge_time = st.number_input("Discharge Time (s)", min_value=0.0, value=init_discharge, step=10.0)
        charging_time = st.number_input("Charging Time (s)", min_value=0.0, value=init_charging, step=10.0)
        decrement_v = st.number_input("Decrement 3.6-3.4V (s)", min_value=0.0, value=init_decrement, step=5.0)
        time_415v = st.number_input("Time at 4.15V (s)", min_value=0.0, value=init_time_415, step=10.0)
        time_cc = st.number_input("Time Constant Current (s)", min_value=0.0, value=init_time_cc, step=10.0)
        
    # Visual Grouping 3: Operational Voltage Boundings
    with st.sidebar.container(border=True):
        st.markdown("⚡ **Voltage Extremes (Volts)**")
        max_v_dischar = st.number_input("Max. Voltage Discharge (V)", min_value=0.0, max_value=6.0, value=init_max_v, step=0.01)
        min_v_charg = st.number_input("Min. Voltage Charge (V)", min_value=0.0, max_value=6.0, value=init_min_v, step=0.01)

# Add Professional Developer Metadata Footer to Sidebar
st.sidebar.write("---")
st.sidebar.caption("🤖 **Engine Version:** v1.0.2-Production")
st.sidebar.caption("🎓 **Developer:** Muhammad Talha / B-29450]")


# 4. Main Panel Display Construction
st.title("🔋 Industrial Battery Remaining Useful Life (RUL) Predictor")
st.markdown("An advanced AI prognostics dashboard powered by an optimized **Extra Trees Ensemble** architecture.")
st.write("---")

# =====================================================================
# MODULE 1: Single Diagnostic Profiler
# =====================================================================
if app_mode == "Single Unit Diagnostics":
    st.subheader("📊 Live Telemetry Summary Matrix")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Target Asset ID", f"BATT-CYC-{int(cycle_index)}")
    m2.metric("Discharge Phase", f"{discharge_time}s")
    m3.metric("Charging Envelope", f"{charging_time}s")
    m4.metric("Voltage Range", f"{min_v_charg}V - {max_v_dischar}V")
    
    st.write("---")
    
    if st.button("🚀 Calculate Remaining Useful Life", type="primary", use_container_width=True):
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
            transformed_features = transformer.transform(input_df)
            prediction = model.predict(transformed_features)
            predicted_rul = int(max(0, prediction[0]))
            
            # Contextual percentage scale against standard max testing threshold (1000 cycles)
            max_expected_lifecycle = 1000
            health_percentage = min(100, int((predicted_rul / max_expected_lifecycle) * 100))
            
            st.subheader("🔮 Prognostic Estimation Outputs")
            st.write(f"**Calculated Structural Integrity Index: {health_percentage}%**")
            st.progress(health_percentage)
            
            st.metric(
                label="Predicted Remaining Useful Life (RUL)", 
                value=f"{predicted_rul} Operational Cycles"
            )
            
            if predicted_rul > 500:
                st.success(f"✅ **Asset Health: Normal Operating State.** Component retains strong lifespan capacity. Recommended Action: Log telemetry on schedule.")
            elif predicted_rul > 100:
                st.warning(f"⚠️ **Asset Health: Moderate Baseline Decay.** Mild internal wear layer indicated. Recommended Action: Increase sensor evaluation frequency.")
            else:
                st.error(f"🚨 **Asset Health: CRITICAL SYSTEM FAULT.** Battery approaching strict End-of-Life thresholds. Recommended Action: Schedule immediate operational replacement.")
                
        except Exception as e:
            st.error(f"Execution Error encountered during computation pipeline: {e}")

# =====================================================================
# MODULE 2: Fleet Batch CSV Processing
# =====================================================================
else:
    st.subheader("📂 Fleet Management Batch Ingestion Portal")
    st.markdown("""
    Upload a bulk `.csv` data export containing historical telemetry files from an entire vehicle network or grid installation. 
    The engine will execute structural transformation and return an operational prediction spreadsheet instantly.
    """)
    
    uploaded_file = st.file_uploader("Choose telemetry spreadsheet to upload", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Multi-encoding parsing protocol fallback
            try:
                batch_df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                batch_df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
                
            st.success("✅ File imported successfully! Analyzing column alignment...")
            
            # Scan file contents to ensure all required model parameters are present
            missing_cols = [col for col in feature_columns if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"❌ **Validation Error!** Uploaded CSV is missing columns required by the model: {missing_cols}")
                st.info("💡 Ensure your file columns match your training telemetry labels exactly.")
            else:
                # RE-ALIGNMENT PROCESS: Force structural sequence matching prior to computing transformation array
                processing_df = batch_df[feature_columns].copy()
                
                # Compute transformation matrix and run inference
                transformed_batch = transformer.transform(processing_df)
                batch_predictions = model.predict(transformed_batch)
                
                # Safely write variables back into display frame arrays
                batch_df['Predicted_RUL_Cycles'] = batch_predictions.astype(int)
                batch_df['Predicted_RUL_Cycles'] = batch_df['Predicted_RUL_Cycles'].clip(lower=0)
                
                st.write("---")
                st.subheader("📋 Fleet Prediction Matrix Preview (First 10 Rows)")
                st.dataframe(batch_df.head(10), use_container_width=True)
                
                # Build export stream buffer
                csv_buffer = batch_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download Full Predictive Maintenance Report (.CSV)",
                    data=csv_buffer,
                    file_name="fleet_battery_prognostics_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Failed to process file batch structure: {e}")