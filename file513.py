import streamlit as st
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="PR Jet Stability Simulator", layout="wide")
st.title("Photoresist Dispensing Jet Simulator")
st.markdown("Rayleigh-Plateau instability based linear stability analysis")

# -----------------------------
# Inputs
# -----------------------------
st.sidebar.header("Input Parameters")

r0_mm = st.sidebar.slider("Nozzle radius r₀ (mm)", 0.1, 2.0, 0.5, 0.1)
U = st.sidebar.slider("Jet velocity U (m/s)", 0.1, 10.0, 2.0, 0.1)
eta_mpas = st.sidebar.slider("Viscosity η (mPa·s)", 1.0, 200.0, 30.0, 1.0)
gamma_mNm = st.sidebar.slider("Surface tension γ (mN/m)", 10.0, 80.0, 30.0, 1.0)
L_cm = st.sidebar.slider("Nozzle-to-substrate distance L (cm)", 1.0, 20.0, 5.0, 0.5)

# unit conversion
r0 = r0_mm / 1000
eta = eta_mpas / 1000
gamma = gamma_mNm / 1000
L = L_cm / 100
rho = 1000
eps = r0 * 0.02

# -----------------------------
# Dispersion relation
# -----------------------------
def growth_rate(x):
    # x = k*r0
    base = (gamma / (rho * r0**3)) * x * (1 - x**2)
    base = np.maximum(base, 0)

    omega = np.sqrt(base)

    # simple viscosity damping
    damping = 1 / (1 + 8 * eta * x**2)
    return omega * damping

x = np.linspace(0.01, 1.2, 300)
omega = growth_rate(x)

idx = np.argmax(omega)
omega_max = omega[idx]
x_max = x[idx]

if omega_max < 1e-10:
    omega_max = 1e-10

k_max = x_max / r0
lambda_max = 2 * np.pi / k_max
tb = np.log(r0 / eps) / omega_max
Lb = U * tb

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Growth-rate spectrum ω(k)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=omega,
        mode="lines",
        name="ω(k)"
    ))

    fig.update_layout(
        xaxis_title="k·r₀",
        yaxis_title="Growth rate ω (1/s)",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Results")

    st.metric("Most unstable k·r₀", f"{x_max:.3f}")
    st.metric("Most unstable wavelength", f"{lambda_max*1000:.2f} mm")
    st.metric("Breakup time", f"{tb*1000:.2f} ms")
    st.metric("Breakup distance", f"{Lb*100:.2f} cm")

    if Lb > L:
        st.success("Jet reaches wafer before breakup")
    else:
        st.error("Breakup before wafer")

# -----------------------------
# Animation
# -----------------------------
st.subheader("Jet deformation")

safe_time_ms = float(min(tb * 1000, 50))
t_ms = st.slider("Time evolution (ms)", 0.0, safe_time_ms, 0.0, 0.5)
t = t_ms / 1000

z = np.linspace(0, min(Lb, 0.15), 300)

amp = eps * np.exp(omega_max * t)
amp = min(amp, 0.8 * r0)

r = r0 + amp * np.cos(k_max * z)

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=z*100,
    y=r*1000,
    mode="lines",
    name="upper"
))

fig2.add_trace(go.Scatter(
    x=z*100,
    y=-r*1000,
    mode="lines",
    fill="tonexty",
    name="lower"
))

fig2.update_layout(
    xaxis_title="Distance (cm)",
    yaxis_title="Jet radius (mm)",
    height=400
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Validation
# -----------------------------
st.subheader("Validation")

st.write("Theoretical critical wavelength:")
st.latex(r"\lambda_c = 2\pi r_0")

st.write("Most unstable wavelength from simulation:")
st.write(f"{lambda_max*1000:.3f} mm")

error = abs(lambda_max - 2*np.pi*r0) / (2*np.pi*r0) * 100
st.write(f"Difference from λc: {error:.2f}%")