"""
GeoRadar — Interactive 3D Groundwater Map Dashboard
Team Astrix · AESH 2026 · Green Radar Systems

Run with:  streamlit run dashboard.py

Displays:
  - 3D subsurface classification map (water / dry soil / rock)
  - Power comparison chart (GeoRadar vs conventional GPR)
  - CNN classification confidence per scan position
  - Live duty-cycle activation map
"""

import numpy as np
import json

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    DEPS_OK = True
except ImportError:
    DEPS_OK = False

# ── Synthetic demo data generators ────────────────────────────────────────────

def make_demo_scan_grid(nx: int = 30, ny: int = 30, seed: int = 42):
    """Generate a synthetic 2-D grid of subsurface class predictions."""
    rng = np.random.default_rng(seed)
    grid = np.zeros((nx, ny), dtype=int)   # 0 = dry soil default

    # Inject a water-bearing "pool" region
    cx, cy, r = nx // 2, ny // 2, 7
    for i in range(nx):
        for j in range(ny):
            if (i - cx) ** 2 + (j - cy) ** 2 < r ** 2:
                grid[i, j] = 1   # water

    # Random rock patches
    rock_centers = rng.integers(3, [nx - 3, ny - 3], size=(4, 2))
    for rc in rock_centers:
        ri, rj = rc
        grid[ri - 2:ri + 2, rj - 2:rj + 2] = 2   # rock

    # Confidence scores
    conf = rng.uniform(0.75, 0.99, (nx, ny)).astype(np.float32)
    conf[grid == 0] = rng.uniform(0.70, 0.92, conf[grid == 0].shape)

    return grid, conf


def make_3d_volume(grid, depths=(2.0, 4.5, 7.0), noise_std=0.3, seed=7):
    """Expand 2-D grid into 3 depth slices with slight variation."""
    rng = np.random.default_rng(seed)
    nx, ny = grid.shape
    volumes = []
    for d in depths:
        layer = grid.copy().astype(float)
        layer += rng.normal(0, noise_std, layer.shape)
        volumes.append((d, np.clip(np.round(layer), 0, 2).astype(int)))
    return volumes


# ── Plotly figure builders ────────────────────────────────────────────────────

CLASS_COLORS = {0: "#D4A96A", 1: "#1A7FCC", 2: "#8B6F47"}
CLASS_LABELS = {0: "Dry soil", 1: "Water-bearing", 2: "Rock"}


def build_3d_scatter(volumes):
    """Build a 3-D scatter plot of subsurface class predictions."""
    fig = go.Figure()
    for depth, layer in volumes:
        nx, ny = layer.shape
        x_coords, y_coords, z_coords, colors, labels = [], [], [], [], []
        for i in range(nx):
            for j in range(ny):
                cls = layer[i, j]
                x_coords.append(i)
                y_coords.append(j)
                z_coords.append(-depth)    # negative = underground
                colors.append(CLASS_COLORS[cls])
                labels.append(CLASS_LABELS[cls])

        for cls in [0, 1, 2]:
            mask = [l == CLASS_LABELS[cls] for l in labels]
            fig.add_trace(go.Scatter3d(
                x=[x_coords[k] for k in range(len(mask)) if mask[k]],
                y=[y_coords[k] for k in range(len(mask)) if mask[k]],
                z=[z_coords[k] for k in range(len(mask)) if mask[k]],
                mode="markers",
                marker=dict(size=5, color=CLASS_COLORS[cls], opacity=0.75),
                name=f"{CLASS_LABELS[cls]} @ {depth}m",
                showlegend=(depth == volumes[0][0]),
            ))

    fig.update_layout(
        title="GeoRadar — 3D Subsurface Classification Map",
        scene=dict(
            xaxis_title="Survey X (m)",
            yaxis_title="Survey Y (m)",
            zaxis_title="Depth (m)",
            bgcolor="rgba(245,248,252,1)",
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=40),
        height=500,
    )
    return fig


def build_power_chart():
    """Bar chart: GeoRadar vs conventional GPR power budget."""
    systems = ["Conventional GPR\n(always-on)", "GeoRadar\n(adaptive duty-cycle)"]
    power_w = [2.0, 0.62]    # W — matches duty_cycle.py model at 30% activation
    colors  = ["#E05C5C", "#2E86AB"]

    fig = go.Figure(go.Bar(
        x=systems, y=power_w,
        marker_color=colors,
        text=[f"{p} W" for p in power_w],
        textposition="outside",
        width=0.45,
    ))
    fig.add_annotation(
        x=1, y=0.62,
        text="<b>~70% reduction</b>",
        showarrow=True, arrowhead=2,
        ax=-80, ay=-40,
        font=dict(color="#2E86AB", size=13),
    )
    fig.update_layout(
        title="Power Consumption: GeoRadar vs Conventional GPR",
        yaxis_title="Average power (W)",
        yaxis_range=[0, 2.5],
        plot_bgcolor="white",
        height=350,
    )
    return fig


def build_activation_map(grid, conf):
    """Heatmap of CNN confidence scores across the survey grid."""
    fig = px.imshow(
        conf,
        color_continuous_scale="Blues",
        zmin=0.7, zmax=1.0,
        labels=dict(color="CNN confidence"),
        title="CNN Classification Confidence Map (top view)",
        aspect="equal",
    )
    fig.update_layout(height=350)
    return fig


# ── Streamlit app ─────────────────────────────────────────────────────────────

def run_app():
    st.set_page_config(
        page_title="GeoRadar Dashboard",
        page_icon="🌊",
        layout="wide",
    )

    st.title("🌊 GeoRadar — Underground Water Detection")
    st.caption("Team Astrix · AESH 2026 · Green Radar Systems · SDG 6 · SDG 9 · SDG 13")

    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    nx = st.sidebar.slider("Grid size X", 10, 50, 30)
    ny = st.sidebar.slider("Grid size Y", 10, 50, 30)
    snr = st.sidebar.slider("SNR (dB)", 5, 40, 20)
    depth_max = st.sidebar.slider("Max depth (m)", 5.0, 15.0, 10.0)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)

    grid, conf = make_demo_scan_grid(nx, ny, seed=int(seed))
    volumes = make_3d_volume(grid, depths=[2.0, depth_max / 2, depth_max], seed=int(seed))

    # KPI row
    water_pct = (grid == 1).sum() / grid.size * 100
    rock_pct  = (grid == 2).sum() / grid.size * 100
    soil_pct  = 100 - water_pct - rock_pct
    avg_conf  = conf.mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Water-bearing zones", f"{water_pct:.1f}%")
    c2.metric("CNN avg confidence", f"{avg_conf:.1f}%")
    c3.metric("Power reduction", "~70%", delta="vs conventional")
    c4.metric("Duty-cycle activation", "~30%", delta="active scans only")

    st.divider()

    # 3D map
    st.plotly_chart(build_3d_scatter(volumes), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(build_power_chart(), use_container_width=True)
    with col_b:
        st.plotly_chart(build_activation_map(grid, conf), use_container_width=True)

    # Summary table
    st.subheader("Subsurface Classification Summary")
    summary = pd.DataFrame({
        "Class": ["Dry soil", "Water-bearing", "Rock"],
        "Grid cells": [
            int((grid == 0).sum()),
            int((grid == 1).sum()),
            int((grid == 2).sum()),
        ],
        "Coverage (%)": [
            f"{soil_pct:.1f}",
            f"{water_pct:.1f}",
            f"{rock_pct:.1f}",
        ],
        "Avg CNN confidence": [
            f"{conf[grid == 0].mean()*100:.1f}%",
            f"{conf[grid == 1].mean()*100:.1f}%",
            f"{conf[grid == 2].mean()*100:.1f}%",
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "⚠️ All results are simulated. Dielectric profiles derived from NASA/ESA open datasets. "
        "CNN trained on synthetic data; real-world validation requires field deployment. "
        "Power values are modelled estimates, not hardware measurements."
    )


if __name__ == "__main__":
    if not DEPS_OK:
        print("Install dependencies first:  pip install streamlit plotly pandas")
    else:
        run_app()
