import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime

# ======================================================================
# 1. PAGE CONFIGURATION
# ======================================================================
st.set_page_config(
    page_title="Enterprise Battery Prognostics Platform",
    page_icon="🔋",
    layout="wide"
)

# ======================================================================
# 2. DATABASE HELPER FUNCTIONS
# ======================================================================
DB_PATH = "battery_prognostics.db"

def init_db():
    """Initialize the SQLite database and create prediction_history table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp               TEXT NOT NULL,
                mode                    TEXT NOT NULL,
                asset_id                TEXT NOT NULL,
                discharge_time          REAL,
                decrement_v             REAL,
                max_voltage_dischar     REAL,
                min_voltage_charg       REAL,
                time_415v               REAL,
                time_cc                 REAL,
                charging_time           REAL,
                predicted_rul           INTEGER,
                health_status           TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def log_prediction(mode, asset_id, discharge_time, decrement_v, max_voltage_dischar,
                   min_voltage_charg, time_415v, time_cc, charging_time,
                   predicted_rul, health_status):
    """Insert a single prediction record into the database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_history
                (timestamp, mode, asset_id, discharge_time, decrement_v,
                 max_voltage_dischar, min_voltage_charg, time_415v, time_cc,
                 charging_time, predicted_rul, health_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, mode, asset_id,
            discharge_time, decrement_v, max_voltage_dischar,
            min_voltage_charg, time_415v, time_cc, charging_time,
            predicted_rul, health_status
        ))
        conn.commit()
    finally:
        conn.close()


def fetch_history():
    """Return the full prediction history as a pandas DataFrame, newest rows first."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM prediction_history ORDER BY id DESC", conn
        )
    finally:
        conn.close()
    return df


def clear_history():
    """Drop and immediately recreate the prediction_history table."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS prediction_history")
        conn.commit()
    finally:
        conn.close()
    # Recreate the empty schema
    init_db()


def get_summary_stats():
    """Return aggregated KPI stats directly from the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM prediction_history")
        total_runs = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(predicted_rul) FROM prediction_history")
        avg_rul_raw = cursor.fetchone()[0]
        avg_rul = round(avg_rul_raw, 1) if avg_rul_raw is not None else 0.0

        cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE health_status = 'Critical'")
        critical_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE health_status = 'Warning'")
        warning_count = cursor.fetchone()[0]
    finally:
        conn.close()
    return total_runs, avg_rul, critical_count, warning_count


# ======================================================================
# 3. HEALTH STATUS CLASSIFIER (reused by both modules)
# ======================================================================
def classify_health(predicted_rul: int) -> str:
    """Map a predicted RUL integer to a categorical health status string."""
    if predicted_rul > 500:
        return "Normal"
    elif predicted_rul > 100:
        return "Warning"
    else:
        return "Critical"


# ======================================================================
# 4. ML ARTIFACT LOADING  (caching unchanged)
# ======================================================================
@st.cache_resource
def load_artifacts():
    model     = joblib.load('battery_rul_et_model.pkl')
    transformer = joblib.load('power_transformer.pkl')
    return model, transformer


try:
    model, transformer = load_artifacts()
except FileNotFoundError:
    st.error(
        "⚠️ **Required Artifact Files Not Found!** "
        "Ensure both **`battery_rul_et_model.pkl`** and **`power_transformer.pkl`** "
        "are saved inside this exact directory."
    )
    st.stop()

# Bootstrap the database on every cold start
init_db()

# Master strict column order required by the power transformer
feature_columns = [
    'Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Max. Voltage Dischar. (V)',
    'Min. Voltage Charg. (V)', 'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)'
]

# ======================================================================
# 5. SESSION STATE – NAVIGATION PERSISTENCE
# ======================================================================
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Single Unit Diagnostics"


# ======================================================================
# 6. SIDEBAR – NAVIGATION BUTTONS + CONTEXTUAL CONTROLS
# ======================================================================
st.sidebar.title("🎮 Dashboard Controller")
st.sidebar.write("---")
st.sidebar.subheader("Navigation")

# ── 3 Full-width navigation buttons ──────────────────────────────────
NAV_PAGES = {
    "Single Unit Diagnostics": "🎯 Single Unit Diagnostics",
    "Batch CSV Processing":    "📂 Batch CSV Processing",
    "Historical Analytics":    "📊 Historical Analytics & Logs",
}

for page_key, page_label in NAV_PAGES.items():
    is_active = (st.session_state.app_mode == page_key)
    # Prepend a ▶ marker on the active page label so the user knows where they are
    display_label = f"▶ {page_label}" if is_active else page_label
    if st.sidebar.button(display_label, use_container_width=True, key=f"nav_{page_key}"):
        st.session_state.app_mode = page_key
        st.rerun()

# Capture current page for downstream routing
app_mode = st.session_state.app_mode

# ── Sidebar parameter controls (Single Unit only) ────────────────────
if app_mode == "Single Unit Diagnostics":
    st.sidebar.write("---")
    st.sidebar.subheader("💡 Load Simulation Presets")
    preset = st.sidebar.selectbox(
        "Choose a telemetry profile to auto-fill:",
        ["Manual Tweaking 🛠️", "Fresh Battery Profile (High RUL) 🟢", "Severely Degraded Profile (Critical EOL) 🔴"]
    )

    if "Fresh Battery" in preset:
        init_cycle       = 15.0;    init_discharge = 3300.0;  init_decrement = 1250.0
        init_max_v       = 3.69;    init_min_v     = 3.18;    init_time_415  = 5600.0
        init_time_cc     = 6900.0;  init_charging  = 11000.0
    elif "Severely Degraded" in preset:
        init_cycle       = 850.0;   init_discharge = 1200.0;  init_decrement = 400.0
        init_max_v       = 3.52;    init_min_v     = 3.39;    init_time_415  = 2100.0
        init_time_cc     = 3400.0;  init_charging  = 5800.0
    else:
        init_cycle       = 100.0;   init_discharge = 2500.0;  init_decrement = 1100.0
        init_max_v       = 3.67;    init_min_v     = 3.21;    init_time_415  = 5400.0
        init_time_cc     = 6700.0;  init_charging  = 10700.0

    st.sidebar.write("---")
    st.sidebar.subheader("⚙️ Fine-Tune Parameters")

    with st.sidebar.container(border=True):
        st.markdown("📋 **Asset Metadata**")
        cycle_index = st.number_input("Cycle Index Reference", min_value=1.0, value=init_cycle, step=1.0)

    with st.sidebar.container(border=True):
        st.markdown("⏱️ **Temporal Telemetry (Seconds)**")
        discharge_time = st.number_input("Discharge Time (s)",         min_value=0.0, value=init_discharge, step=10.0)
        charging_time  = st.number_input("Charging Time (s)",          min_value=0.0, value=init_charging,  step=10.0)
        decrement_v    = st.number_input("Decrement 3.6-3.4V (s)",     min_value=0.0, value=init_decrement, step=5.0)
        time_415v      = st.number_input("Time at 4.15V (s)",          min_value=0.0, value=init_time_415,  step=10.0)
        time_cc        = st.number_input("Time Constant Current (s)",  min_value=0.0, value=init_time_cc,   step=10.0)

    with st.sidebar.container(border=True):
        st.markdown("⚡ **Voltage Extremes (Volts)**")
        max_v_dischar = st.number_input("Max. Voltage Discharge (V)", min_value=0.0, max_value=6.0, value=init_max_v, step=0.01)
        min_v_charg   = st.number_input("Min. Voltage Charge (V)",    min_value=0.0, max_value=6.0, value=init_min_v, step=0.01)

# ── Developer metadata footer ────────────────────────────────────────
st.sidebar.write("---")
st.sidebar.caption("🤖 **Engine Version:** v1.0.2-Production")
st.sidebar.caption("🎓 **Developer:**\n Muhammad Talha / B-29450\n Ahmad Yaqoob / B-29422")


# ======================================================================
# 7. MAIN PANEL HEADER
# ======================================================================
st.title("🔋 Industrial Battery Remaining Useful Life (RUL) Predictor")
st.markdown("An advanced AI prognostics dashboard powered by an optimized **Extra Trees Ensemble** architecture.")
st.write("---")


# ======================================================================
# MODULE A – SINGLE UNIT DIAGNOSTICS
# ======================================================================
if app_mode == "Single Unit Diagnostics":
    st.subheader("📊 Live Telemetry Summary Matrix")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Target Asset ID",    f"BATT-CYC-{int(cycle_index)}")
    m2.metric("Discharge Phase",    f"{discharge_time}s")
    m3.metric("Charging Envelope",  f"{charging_time}s")
    m4.metric("Voltage Range",      f"{min_v_charg}V – {max_v_dischar}V")

    st.write("---")

    if st.button("🚀 Calculate Remaining Useful Life", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            'Discharge Time (s)':        discharge_time,
            'Decrement 3.6-3.4V (s)':    decrement_v,
            'Max. Voltage Dischar. (V)':  max_v_dischar,
            'Min. Voltage Charg. (V)':    min_v_charg,
            'Time at 4.15V (s)':          time_415v,
            'Time constant current (s)':  time_cc,
            'Charging time (s)':          charging_time,
        }])

        try:
            transformed_features = transformer.transform(input_df)
            prediction            = model.predict(transformed_features)
            predicted_rul         = int(max(0, prediction[0]))

            # Health classification
            health_status = classify_health(predicted_rul)

            # ── Database write ────────────────────────────────────────
            asset_id = f"BATT-CYC-{int(cycle_index)}"
            log_prediction(
                mode               = "Single Unit",
                asset_id           = asset_id,
                discharge_time     = discharge_time,
                decrement_v        = decrement_v,
                max_voltage_dischar= max_v_dischar,
                min_voltage_charg  = min_v_charg,
                time_415v          = time_415v,
                time_cc            = time_cc,
                charging_time      = charging_time,
                predicted_rul      = predicted_rul,
                health_status      = health_status,
            )
            st.toast(f"✅ Prediction logged to database for asset **{asset_id}**", icon="💾")

            # ── Results display ───────────────────────────────────────
            max_expected_lifecycle = 1000
            health_percentage = min(100, int((predicted_rul / max_expected_lifecycle) * 100))

            st.subheader("🔮 Prognostic Estimation Outputs")
            st.write(f"**Calculated Structural Integrity Index: {health_percentage}%**")
            st.progress(health_percentage)

            st.metric(
                label="Predicted Remaining Useful Life (RUL)",
                value=f"{predicted_rul} Operational Cycles"
            )

            if health_status == "Normal":
                st.success("✅ **Asset Health: Normal Operating State.** Component retains strong lifespan capacity. Recommended Action: Log telemetry on schedule.")
            elif health_status == "Warning":
                st.warning("⚠️ **Asset Health: Moderate Baseline Decay.** Mild internal wear layer indicated. Recommended Action: Increase sensor evaluation frequency.")
            else:
                st.error("🚨 **Asset Health: CRITICAL SYSTEM FAULT.** Battery approaching strict End-of-Life thresholds. Recommended Action: Schedule immediate operational replacement.")

        except Exception as e:
            st.error(f"Execution Error encountered during computation pipeline: {e}")


# ======================================================================
# MODULE B – BATCH CSV PROCESSING
# ======================================================================
elif app_mode == "Batch CSV Processing":
    st.subheader("📂 Fleet Management Batch Ingestion Portal")
    st.markdown(
        "Upload a bulk `.csv` data export containing historical telemetry files from an entire "
        "vehicle network or grid installation. The engine will execute structural transformation, "
        "return an operational prediction spreadsheet, and persist every row into the database."
    )

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

            missing_cols = [col for col in feature_columns if col not in batch_df.columns]

            if missing_cols:
                st.error(f"❌ **Validation Error!** Uploaded CSV is missing required columns: `{missing_cols}`")
                st.info("💡 Ensure your file columns match your training telemetry labels exactly.")
            else:
                # RE-ALIGNMENT: Force strict sequence matching prior to transformer
                processing_df = batch_df[feature_columns].copy()

                transformed_batch  = transformer.transform(processing_df)
                batch_predictions  = model.predict(transformed_batch)

                batch_df['Predicted_RUL_Cycles'] = batch_predictions.astype(int)
                batch_df['Predicted_RUL_Cycles'] = batch_df['Predicted_RUL_Cycles'].clip(lower=0)
                batch_df['Health_Status']         = batch_df['Predicted_RUL_Cycles'].apply(classify_health)

                # ── Persist every row into SQLite ─────────────────────
                rows_written = 0
                for idx, row in batch_df.iterrows():
                    asset_id = f"BATCH-ROW-{idx + 1}"
                    log_prediction(
                        mode               = "Batch File Ingestion",
                        asset_id           = asset_id,
                        discharge_time     = float(row['Discharge Time (s)']),
                        decrement_v        = float(row['Decrement 3.6-3.4V (s)']),
                        max_voltage_dischar= float(row['Max. Voltage Dischar. (V)']),
                        min_voltage_charg  = float(row['Min. Voltage Charg. (V)']),
                        time_415v          = float(row['Time at 4.15V (s)']),
                        time_cc            = float(row['Time constant current (s)']),
                        charging_time      = float(row['Charging time (s)']),
                        predicted_rul      = int(row['Predicted_RUL_Cycles']),
                        health_status      = str(row['Health_Status']),
                    )
                    rows_written += 1

                st.toast(f"💾 {rows_written} batch records written to database.", icon="✅")

                st.write("---")
                st.subheader("📋 Fleet Prediction Matrix Preview (First 10 Rows)")
                st.dataframe(batch_df.head(10), use_container_width=True)

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


# ======================================================================
# MODULE C – HISTORICAL ANALYTICS & LOGS
# ======================================================================
elif app_mode == "Historical Analytics":
    st.subheader("📊 Operational Control Room – Historical Analytics & Logs")
    st.markdown("Live aggregated fleet intelligence drawn directly from the persistent prediction ledger.")
    st.write("---")

    # ── KPI Metrics Row ───────────────────────────────────────────────
    total_runs, avg_rul, critical_count, warning_count = get_summary_stats()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🗂️ Total Evaluation Runs",       total_runs)
    k2.metric("🔋 Fleet Avg. Predicted RUL",    f"{avg_rul} cycles")
    k3.metric("🚨 Critical Alerts (EOL)",        critical_count)
    k4.metric("⚠️ Degraded Alerts (Warning)",    warning_count)

    st.write("---")

    # ── Action Controls Row ───────────────────────────────────────────
    history_df = fetch_history()

    action_col1, action_col2 = st.columns([1, 1])

    with action_col1:
        if not history_df.empty:
            csv_export = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Full History as CSV",
                data=csv_export,
                file_name="battery_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.download_button(
                label="📥 Export Full History as CSV",
                data=b"",
                file_name="battery_prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=True
            )

    with action_col2:
        if st.button("🗑️ Clear Prediction History", type="secondary", use_container_width=True):
            clear_history()
            st.rerun()

    st.write("---")

    # ── Diagnostic Ledger ─────────────────────────────────────────────
    st.subheader("🗃️ Diagnostic Ledger")

    if history_df.empty:
        st.info(
            "📭 **No prediction records found.** "
            "Run a Single Unit Diagnostic or process a Batch CSV to begin populating the ledger."
        )
    else:
        st.dataframe(history_df, use_container_width=True)