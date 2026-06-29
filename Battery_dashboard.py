import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from skopt import gp_minimize
from skopt.space import Real
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")


# ==========================================================
#                 SUPPORT FUNCTIONS
# ==========================================================

def exp_decay(cycle, a, b, c):
    return a * np.exp(b * cycle) + c


def extract_sensitive_features(df):
    """Extract high-sensitivity battery degradation metrics."""

    # Check required columns
    required = ["Time", "Voltage_measured", "Current_measured"]
    if not all(col in df.columns for col in required):
        raise ValueError("Uploaded file must contain: Time, Voltage_measured, Current_measured")

    df_dis = df[df['Current_measured'] < -0.01].copy()
    if df_dis.empty:
        raise ValueError("No discharge points found (Current_measured < -0.01).")

    time = df_dis['Time'].values
    voltage = df_dis['Voltage_measured'].values
    current = df_dis['Current_measured'].values

    # Capacity & energy (Ah, Wh)
    capacity_Ah = -np.trapz(current, time) / 3600
    energy_Wh = -np.trapz(voltage * current, time) / 3600

    avg_v = voltage.mean()
    min_v = voltage.min()
    voltage_slope = (voltage[-1] - voltage[0]) / max((time[-1] - time[0]), 1e-6)
    voltage_var = voltage.var()

    # Internal resistance using first N points
    window = min(10, len(voltage))
    dv = voltage[:window].max() - voltage[:window].min()
    di = current[:window].max() - current[:window].min()
    r_internal = abs(dv / (di + 1e-6))

    # Discharge efficiency
    V_nom = 3.7
    theoretical_energy = V_nom * capacity_Ah
    discharge_eff = energy_Wh / theoretical_energy if theoretical_energy > 0 else 0.0

    return {
        "capacity_Ah": capacity_Ah,
        "energy_Wh": energy_Wh,
        "avg_v": avg_v,
        "min_v": min_v,
        "voltage_slope": voltage_slope,
        "voltage_var": voltage_var,
        "r_internal": r_internal,
        "discharge_efficiency": discharge_eff,
    }


def score_function(c_rate, rest, F):
    """Sensitive scoring model for surrogate training."""
    return (
        300
        - 25 * (c_rate - 1)
        + 0.45 * rest
        + 70 * (F["capacity_Ah"] - 1.9)
        + 20 * (F["energy_Wh"] - 6.5)
        - 450 * F["r_internal"]
        + 60 * (F["avg_v"] - 3.7)
        + 200 * F["voltage_slope"]
        - 8 * F["voltage_var"]
        + 350 * (F["discharge_efficiency"] - 0.92)
    )


# ==========================================================
#                     STREAMLIT APP
# ==========================================================

st.set_page_config(layout="wide")
st.title("🔋 Battery Analytics Dashboard — RUL-Driven Optimization")

# ==========================================================
#              SECTION 1 — CAPACITY / RUL (T2)
# ==========================================================
st.header("1. Baseline Capacity / RUL (T2)")

try:
    df_capacity = pd.read_csv("capacity_summary.csv")
except:
    st.error("⚠️ File 'capacity_summary.csv' not found in directory.")
    st.stop()

if "Capacity_Ah" not in df_capacity.columns or "Cycle" not in df_capacity.columns:
    st.error("capacity_summary.csv must contain columns: Cycle, Capacity_Ah")
    st.stop()

initial_capacity = df_capacity['Capacity_Ah'].iloc[0]
cycles_data = df_capacity['Cycle'].values
capacity_data = df_capacity['Capacity_Ah'].values

# Exponential decay fit
p0 = [0.3, -0.01, 1.4]
params, _ = curve_fit(exp_decay, cycles_data, capacity_data, p0=p0)
a_fit, b_fit, c_fit = params

cycle_predict = np.arange(1, 600)
capacity_predict = exp_decay(cycle_predict, a_fit, b_fit, c_fit)

EOL_CAPACITY = initial_capacity * 0.8
eol_idx = np.where(capacity_predict < EOL_CAPACITY)[0]
EOL_CYCLE = int(eol_idx[0]) if len(eol_idx) > 0 else None

last_cycle = cycles_data[-1]
baseline_RUL = (EOL_CYCLE - last_cycle) if EOL_CYCLE else None

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cycles_data, capacity_data, 'o', label="Measured")
ax.plot(cycle_predict, capacity_predict, '-', label="Fit")
ax.axhline(EOL_CAPACITY, color='r', linestyle='--', label="80% EOL")
if EOL_CYCLE:
    ax.axvline(EOL_CYCLE, color='k', linestyle=':')
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.write(f"📌 **Predicted EOL Cycle:** {EOL_CYCLE}")
st.write(f"📌 **Baseline RUL:** {baseline_RUL} cycles" if baseline_RUL else "⚠️ EOL not reached in prediction range.")


# ==========================================================
#             SECTION 2 — UPLOAD DISCHARGE CYCLE
# ==========================================================
st.header("2. Upload a Discharge Cycle (or use example)")

uploaded = st.file_uploader("Upload a discharge CSV", type="csv")
if uploaded is None:
    st.stop()

uploaded.seek(0)
df_cycle = pd.read_csv(uploaded)

st.write("#### Preview:")
st.dataframe(df_cycle.head())

# Extract features
try:
    features = extract_sensitive_features(df_cycle)
except Exception as e:
    st.error(f"Error extracting features: {e}")
    st.stop()

st.write("### Extracted Features (Highly Sensitive)")
for k, v in features.items():
    st.write(f"- **{k}**: {v}")


# ==========================================================
#          SECTION 3 — TRAIN SURROGATE MODEL
# ==========================================================
st.header("3. Surrogate Creation (Data-driven RUL Prediction)")

C_rates = np.linspace(0.5, 3.0, 12)
Rest_times = np.linspace(0, 30, 12)

grid = [[c, r] for c in C_rates for r in Rest_times]
df_sur = pd.DataFrame(grid, columns=["c_rate", "rest"])

df_sur["score"] = df_sur.apply(lambda row: score_function(row["c_rate"], row["rest"], features), axis=1)

X = df_sur[["c_rate", "rest"]]
y = df_sur["score"]

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    objective='reg:squarederror'
)
model.fit(X, y)

st.success("Surrogate model trained using extracted features!")


# ==========================================================
#                   SECTION 4 — OPTIMIZATION
# ==========================================================
st.header("4. Optimization — Maximize Predicted RUL")


def objective(params):
    c, r = params
    c = np.clip(c, 0.5, 3.0)
    r = np.clip(r, 0, 30)
    return -model.predict(np.array([[c, r]]))[0]


dims = [
    Real(0.5, 3.0, name="c_rate"),
    Real(0, 30, name="rest")
]

res = gp_minimize(objective, dims, n_calls=40, random_state=42)

opt_c, opt_r = res.x
opt_score = -res.fun

st.subheader("Optimal Charging Protocol (Based on Uploaded Cycle)")
st.write(f"**Optimal C-rate:** {opt_c:.3f} C")
st.write(f"**Optimal Rest Time:** {opt_r:.3f} min")
st.write(f"**Predicted Score:** {opt_score:.2f}")
st.write(f"- **Discharge Efficiency**: {features['discharge_efficiency']:.4f}")

baseline_score = model.predict(np.array([[1.0, 0.0]]) )[0]


if baseline_RUL:
    rul_at_opt = baseline_RUL * (1 + (opt_score - baseline_score) / 300)

    st.subheader("📈 RUL Prediction at Optimal Protocol")
    st.write(f"**Baseline RUL:** {baseline_RUL:.1f} cycles")
    st.write(f"**RUL at Optimal Condition:** {rul_at_opt:.1f} cycles")
else:
    st.warning("Baseline RUL could not be determined from capacity_summary.csv.")
