#  Smart Surveillance System – Anomaly Detection

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)  
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-green)](https://github.com/ultralytics/ultralytics)  

An end-to-end **AI-powered Surveillance System** for detecting suspicious activities in video datasets (UCSD, Avenue, custom CCTV feeds).  
The system focuses on **loitering detection** and **abandoned bag detection**, with real-time tracking, alert generation, and an interactive **Streamlit Dashboard**.

---

## Dashboard
<img width="1919" height="1079" alt="Screenshot 2025-08-24 232010" src="https://github.com/user-attachments/assets/1ac3d9ee-ace9-4600-a85c-d1f5fb8d7c46" />

<img width="1919" height="1073" alt="Screenshot 2025-08-24 232028" src="https://github.com/user-attachments/assets/201263e4-69d5-4c0f-826f-fc4a5e3df913" />


## Working

<img width="1919" height="1012" alt="Screenshot 2025-08-24 231121" src="https://github.com/user-attachments/assets/fd4768fb-feac-4630-bad5-22a014bc85dc" />

<img width="1919" height="1006" alt="Screenshot 2025-08-24 231246" src="https://github.com/user-attachments/assets/0551b2c7-e6e9-42c8-8f75-efdbe71c6a2d" />

---

##  Features
-  **YOLOv8 Object Detection** – Detects people, bags, bicycles, and other objects.
-  **StrongSORT Tracking** – Assigns consistent IDs across frames for multi-object tracking.
-  **Behavioral Rule Engine**  
  - **Loitering Detection**: Person stays in an area for longer than allowed.  
  - **Abandoned Bag Detection**: Bag left unattended for a fixed duration.
-  **Alerts & Logging** – Snapshots, CSV logs, and annotated video outputs.
-  **Interactive Dashboard** – View alerts, filter by type/object, and inspect snapshots.

---

##  Project Structure
```
├── src/
│   ├── detect_anomalies.py     # Main detection script
|   ├── streamkit_app.py        # Streamlit dashboard
│   ├── rules/                  # Behavioral rules (loitering, abandonment)
│   ├── utils/                  # Helper functions (drawing, logging, tracking)
│   ├── anomaly_model.py        # IsolationForest (optional anomaly model)
│   ├── train_iso.py            # Training script for anomaly features
│
├── outputs/
│   ├── alerts/                 # CSV logs
|   ├── snaps/                  # Snapshots
│   ├── videos/                 # Processed video outputs
|   ├── models/                 # Trained models
│
├── data/                       # Datasets (UCSD, Avenue, etc.)
├── requirements.txt
└── README.md
```

---

##  Installation

```bash
# Clone repository
git clone https://github.com/zaidtausif56/Smart-Surveillance-System.git
cd smart-surveillance-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

##  Usage

### **1. Run Detection**
```bash
# Single video
python -m src.detect_anomalies --video data/test.avi --show --save

# Folder of .avi videos or UCSD/Avenue TestXXX .tif folders
python -m src.detect_anomalies --folder data/UCSDped2/Test --save
```

### **2. Launch Dashboard**
```bash
streamlit run src/streamlit_app.py
```
 Explore alerts interactively, filter by type, and preview snapshots.

---

##  Algorithms

<details>
<summary><b>Loitering Detection</b></summary>
Person is flagged if displacement < **40 px** over a window of **12 seconds**.  
Implemented using centroid history and displacement calculation.
</details>

<details>
<summary><b>Abandoned Bag Detection</b></summary>
Bag is flagged if:  
- Stationary (<20 px movement)  
- Unattended (no person within 120 px)  
- Duration > 10–12 seconds
</details>

<details>
<summary><b>Tracking</b></summary>
Uses StrongSORT/ByteTrack to minimize ID switches and maintain object identity.
</details>

---

##  Future Work
-  Integrate **IsolationForest** anomaly scoring  
-  Use **GANs for synthetic anomaly generation** (e.g., rare abandoned bag events)  
-  Extend for **real-time CCTV integration**

---


##  Author
Developed by [Md Zaid Tausif](https://github.com/zaidtausif56).
