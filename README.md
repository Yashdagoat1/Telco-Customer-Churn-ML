# Telco Churn – End-to-End ML Project

## 🚀 Live Demo

🔗 **Try the app here:** https://telco-customer-churn-ml-qlia.onrender.com/ui/

⚠️ Note: App may take ~30 seconds to load due to cold start (free hosting).

## 📌 Purpose
Build and deploy a complete machine learning solution to predict customer churn in a telecom setting—from data preprocessing and model training to a production-ready API and web interface.

---

## 🚀 Problem Solved & Benefits

- **Faster decision-making:** Predicts customers likely to churn so retention strategies can be applied early  
- **Operational ML:** Model is accessible via a REST API and UI (no notebooks required)  
- **Reproducibility:** Fixed dependencies + version control ensure consistent builds  
- **Automation:** CI/CD pipeline validates and deploys updates automatically  

---

## 🏗️ What I Built

### 🔹 Data & Modeling
- Feature engineering on Telco dataset  
- Trained classification models (XGBoost / Random Forest)  
- Selected best-performing model  

### 🔹 Model Artifacts
- Saved trained model (`model.pkl`)  
- Stored feature columns (`feature_columns.json`) for consistent inference  

### 🔹 Inference Service
- Built using FastAPI  
- Endpoints:
  - `POST /predict` → returns churn prediction + probability  
  - `GET /health` → service health check  

### 🔹 Web UI
- Built using Gradio  
- Available at `/ui` for easy testing  

### 🔹 Deployment
- Deployed on Render (managed cloud platform)  
- Handles environment, dependencies, and hosting  

### 🔹 CI/CD Pipeline
- Implemented using GitHub Actions  
- Automatically:
  - installs dependencies  
  - checks model files  
  - validates app startup  
- Render auto-deploys on every push  

---

## 🔄 Deployment Flow

1. Push code to GitHub  
2. CI pipeline runs (build + validation)  
3. Render automatically deploys updated code  
4. API becomes live  
5. Users:
   - Send requests to `/predict`  
   - Access UI via `/ui`  

---

## ⚙️ ML Pipeline & Workflow

1. Data ingestion  
2. Data cleaning & preprocessing  
3. Feature engineering  
4. Model training  
5. Model evaluation  
6. Model serialization (`model.pkl`)  
7. Inference pipeline (same transformations applied)  
8. API deployment  

---

## ⚠️ Challenges & Solutions

### 1. Python version mismatch
- **Issue:** Some libraries failed on newer Python versions  
- **Solution:** Fixed Python version for compatibility  

### 2. Missing model artifacts
- **Issue:** Model file not found during deployment  
- **Solution:** Ensured artifacts are included and correctly loaded  

### 3. API validation errors
- **Issue:** Missing required input fields  
- **Solution:** Matched API schema with training features  

### 4. Feature mismatch
- **Issue:** Training vs inference feature inconsistency  
- **Solution:** Used saved feature columns and reindexed inputs  

### 5. Environment differences
- **Issue:** Local vs deployed behavior mismatch  
- **Solution:** Used virtual environments + CI checks  

---

## 🛠️ Tech Stack

- Python  
- FastAPI  
- Gradio  
- Scikit-learn  
- XGBoost  
- GitHub Actions (CI/CD)  
- Render (Deployment)  

---

## ▶️ How to Run Locally

```bash
# create virtual environment
python -m venv .venv

# activate (Windows)
.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run app
uvicorn src.app.main:app --reload
