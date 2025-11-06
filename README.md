# Time Series Anomaly Detection for IoT Sensors — NASA Bearing Dataset

## 🧠 Overview
This project implements an **end-to-end anomaly detection pipeline** for industrial IoT sensor data.  
We use the **NASA Bearing Dataset**, which records vibration data from four bearings on a rotating shaft.  
The goal is to automatically detect **abnormal vibration patterns** that may indicate **bearing degradation or imminent failure**.

---

## 📊 Problem Statement
Equipment in manufacturing facilities is monitored using IoT sensors.  
Identifying **anomalies in sensor readings** helps in **predictive maintenance**, preventing costly downtime.  
We develop and compare two anomaly detection approaches:

1. **Isolation Forest** — a statistical/unsupervised model  
2. **LSTM Autoencoder** — a deep learning sequence model

---

## 🧩 Project Structure
``` bash
bearing_anomaly_project/
│
├── data/
│   ├── 1st_test/
│   │   ├── 2003.10.22.12.06.24  (text files from dataset)
│   │   └── ...
│
├── notebooks/
│   └── anomaly_detection.ipynb
│
├── models/
│   ├── isolation_forest.joblib
│   └── lstm_autoencoder/
│
├── outputs/
│   ├── feature_distributions.png
│   ├── anomaly_visualization.png
│   └── summary.txt
│
└── README.md
```


---

## ⚙️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow joblib
```

## Download Dataset
Download the NASA Bearing Dataset from Kaggle:
🔗 https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset
