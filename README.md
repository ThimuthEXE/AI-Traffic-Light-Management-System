# AI-Based Traffic Light Management System

**Course:** IT 3182 - Essentials of Artificial Intelligence  
**Institution:** General Sir John Kotelawala Defence University (KDU), Faculty of Computing  
**Group:** 06  

---

## 🚦 Project Overview
An intelligent, computer-vision and machine-learning-driven traffic light management system designed to replace static fixed-time signals with dynamic, traffic-responsive control.

### Core Modules:
1. **i_engine/**: 
   - **Computer Vision:** YOLOv8 / OpenCV for multi-lane vehicle detection and classification (car, bus, truck, motorcycle, emergency).
   - **Density & PCU Calculation:** Passenger Car Unit weighted traffic load estimation.
   - **Fuzzy Decision Engine:** Mamdani fuzzy logic controller calculating optimal green-light intervals.
   - **Congestion Predictor:** ML forecasting model (Scikit-Learn) for historical traffic pattern analysis.
2. **ackend/**: Node.js & Express REST API / WebSockets for real-time intersection state streaming and MongoDB data persistence.
3. **rontend/**: React.js interactive web dashboard for real-time traffic monitoring, lane visualizers, and analytics reports.
4. **data/**: Sample traffic footage, camera feeds, and training datasets.
5. **docs/**: Project proposal, architecture diagrams, and academic reports.

---

## 🛠️ Tech Stack
- **AI / Computer Vision:** Python, Ultralytics YOLO, OpenCV, Scikit-learn, Scikit-fuzzy
- **Backend & Communication:** Node.js, Express.js, WebSockets / Socket.io
- **Database:** MongoDB
- **Frontend Dashboard:** React.js, TailwindCSS / Material-UI, Chart.js
