
# 🌊 GeoRadar — AI-Powered Green Radar System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-AI-orange?style=for-the-badge\&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge\&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-3D_Visualization-3f4f75?style=for-the-badge\&logo=plotly)
![Status](https://img.shields.io/badge/Status-Prototype-success?style=for-the-badge)

### Team Astrix · AESH 2026 · Sustainability Hackathon

AI-powered underground water detection using Green Radar Systems, adaptive sensing, and 3D subsurface visualization.

</div>

---

# 📌 Overview

GeoRadar is an intelligent groundwater exploration platform that combines:

* 🧠 AI-based subsurface classification
* 📡 Adaptive Green Radar concepts
* 🌍 3D underground mapping
* 🖼️ Soil-image screening
* ⚡ Low-power duty-cycle operation

The system aims to support smarter and more sustainable groundwater surveying while reducing unnecessary radar power consumption.

---

# 🚨 Problem Statement

Traditional groundwater exploration systems:

* Consume high power continuously
* Require expensive large-scale surveys
* Lack intelligent adaptive scanning
* Provide limited visualization for quick decision-making

GeoRadar addresses these limitations using AI-assisted radar analysis and adaptive activation strategies.

---

# 💡 Proposed Solution

GeoRadar combines AI, radar simulation, and visualization into one integrated platform.

## Core Features

✅ CNN subsurface classification
✅ Water-bearing zone prediction
✅ 3D underground visualization
✅ Soil-image validation
✅ Invalid-image rejection (logos/non-soil inputs)
✅ Adaptive radar duty-cycle simulation
✅ Interactive Streamlit dashboard

---

# 🧠 System Architecture

```text
          Soil Image / Radar Signal
                       │
                       ▼
           ┌────────────────────┐
           │ Signal Processing  │
           │ & Image Validation │
           └────────────────────┘
                       │
                       ▼
           ┌────────────────────┐
           │   CNN AI Model     │
           │  Classification    │
           └────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌──────────────────┐      ┌──────────────────┐
│ Water Prediction │      │ Power Optimization│
└──────────────────┘      └──────────────────┘
         │                           │
         └─────────────┬─────────────┘
                       ▼
           ┌────────────────────┐
           │ 3D Visualization   │
           │ Interactive UI     │
           └────────────────────┘
```

---

# 🛰️ AI Model

GeoRadar uses a 1D CNN architecture for subsurface classification.

### Classification Targets

| Class | Description         |
| ----- | ------------------- |
| 0     | Dry Soil            |
| 1     | Water-Bearing Layer |
| 2     | Rock Formation      |

### Model Pipeline

```text
Radar A-Scan
     ↓
Envelope Extraction
     ↓
1D CNN Processing
     ↓
Softmax Classification
     ↓
Confidence Estimation
```

---

# 📊 Dashboard Features

The interactive dashboard includes:

* 🌍 3D underground classification map
* 📈 CNN confidence heatmap
* ⚡ Power-consumption comparison
* 🖼️ Soil-image analysis panel
* 📡 Adaptive radar activation simulation

---

# ⚡ Green Radar Concept

GeoRadar reduces unnecessary radar operation using adaptive duty-cycle activation.

| System           | Average Power |
| ---------------- | ------------- |
| Conventional GPR | 2.0 W         |
| GeoRadar         | 0.62 W        |

### Estimated Reduction

# ≈ 70% Lower Power Consumption

---

# 🌱 UN Sustainable Development Goals

| SDG    | Contribution                    |
| ------ | ------------------------------- |
| SDG 6  | Smart groundwater exploration   |
| SDG 9  | AI & Green Radar innovation     |
| SDG 12 | Efficient energy usage          |
| SDG 13 | Lower environmental impact      |
| SDG 17 | Interdisciplinary collaboration |

---

# 🧪 Technologies Used

## Software

* Python
* TensorFlow
* Streamlit
* Plotly
* NumPy
* Pandas
* Scikit-learn

## AI / Visualization

* CNN Classification
* 3D Plotly Visualization
* Interactive Dashboard UI

---

# 📂 Repository Structure

```text
GeoRadar/
│
├── README.md
├── requirements.txt
│
├── docs/
├── src/
├── simulation/
├── results/
├── assets/
├── hardware/
└── data/
```

---

# ▶️ How to Run

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Dashboard

```bash
streamlit run src/dashboard.py
```

## Train CNN Model

```bash
python src/cnn_model.py --train --epochs 50
```

---

# 📈 Prototype Results

### Demonstrated Capabilities

* AI-based groundwater prediction
* Soil-image validation
* 3D underground mapping
* Radar power optimization simulation
* CNN confidence visualization

---

# ⚠️ Current Limitations

* Current radar data is partially synthetic
* Real-world field deployment still in progress
* Groundwater predictions require physical validation

---

# 🚀 Future Work

* Real GPR hardware integration
* Drone-assisted surveying
* GIS & satellite-data fusion
* Edge AI deployment
* Real-world field validation

---

# 👨‍💻 Team Astrix

AESH 2026 Sustainability Hackathon Project

Focused on AI, sustainability, and intelligent Green Radar systems.

---

# 📜 AI Usage Disclosure

AI tools were used to assist with:

* Code generation
* UI design
* Documentation drafting
* Visualization support

All outputs were reviewed, modified, and validated by the team.
validated, and modified by team members. The CNN architecture, signal simulation logic, duty-cycle algorithm, and all numerical results were developed and verified by the team.
