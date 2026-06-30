#  AI-Based Battery Health Prediction & Charging Optimization

An interactive **Streamlit** application that predicts **Remaining Useful Life (RUL)** and **End-of-Life (EOL)** of lithium-ion batteries while recommending an optimal charging protocol using machine learning and Bayesian Optimization.

This project was developed as part of an **Engineering Chemistry** course at **BITS Pilani** by a team of four members. My primary responsibility was the design and development of the complete Streamlit application.

---

##  Project Overview

Lithium-ion batteries gradually degrade with repeated charge-discharge cycles, reducing their capacity and efficiency. This project leverages data-driven modeling to estimate battery health from operational data and recommend charging parameters that maximize battery lifespan.

The application enables users to:

* Predict Remaining Useful Life (RUL)
* Estimate End-of-Life (EOL)
* Analyze battery degradation trends
* Extract electrochemical performance metrics
* Recommend an optimal charging C-rate and rest time
* Visualize battery performance interactively

---

##  Features

* Interactive Streamlit dashboard
* Upload battery discharge cycle data (CSV)
* Automatic feature extraction from battery sensor data
* Remaining Useful Life (RUL) prediction
* End-of-Life (EOL) estimation
* Battery degradation visualization
* XGBoost-based surrogate modeling
* Bayesian Optimization for charging protocol recommendation

---

##  Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* SciPy
* XGBoost
* Scikit-Optimize (Bayesian Optimization)

---

##  Dataset

* **NASA Lithium-Ion Battery Dataset**
* Battery Used: **B0047**

The dataset contains battery charge-discharge cycle information including voltage, current, time, and capacity measurements used for battery degradation analysis.

---

##  Methodology

1. Load historical battery capacity data.
2. Fit an exponential degradation model to estimate battery aging.
3. Predict Remaining Useful Life (RUL) and End-of-Life (EOL).
4. Upload a discharge cycle for analysis.
5. Extract key electrochemical features:

   * Capacity
   * Energy
   * Average Voltage
   * Minimum Voltage
   * Voltage Slope
   * Voltage Variance
   * Internal Resistance
   * Discharge Efficiency
6. Train an XGBoost surrogate model.
7. Apply Bayesian Optimization to determine the optimal charging protocol.

---

##  Results

The application performs:

* Battery health assessment
* Remaining Useful Life prediction
* End-of-Life estimation
* Automatic electrochemical feature extraction
* Charging protocol optimization
* Interactive visualization of battery degradation

---

##  Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Battery-Health-Prediction.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit application

```bash
streamlit run battery_dashboard.py
```

---

##  Repository Structure

```
Battery-Health-Prediction/
│
├── battery_dashboard.py
├── capacity_summary.csv
├── sample_discharge_cycle.csv
├── requirements.txt
├── README.md
├── ec_project.pptx
```

---

##  My Contribution

This project was completed by a **team of four**.

My responsibilities included:

* Developing the complete Streamlit application
* Integrating data upload and preprocessing
* Building the interactive dashboard
* Implementing visualization of battery degradation
* Integrating machine learning predictions
* Displaying optimization results within the application

---

##  Potential Applications

* Electric Vehicles (EVs)
* Battery Management Systems (BMS)
* Renewable Energy Storage
* Consumer Electronics
* Robotics and Drones
* Predictive Maintenance

---

##  License

This project is intended for educational and portfolio purposes.
