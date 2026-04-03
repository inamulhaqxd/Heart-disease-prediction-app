# 🫀 Heart Disease Prediction App

A machine learning web application built with **Streamlit** that predicts whether a patient is at risk of heart disease using the **UCI Heart Disease Dataset** and a **Decision Tree Classifier**.

---

## 🚀 Features

* 🤖 Decision Tree Classifier with tuned hyperparameters
* 📥 Automatic dataset download using `ucimlrepo` (no Kaggle required)
* 🧾 Interactive patient input form (13 medical features)
* 📊 Exploratory Data Analysis (EDA) dashboard:

  * Class Distribution & Age KDE
  * Correlation Heatmap
  * ROC Curve & Confusion Matrix
  * Feature Importance
* 🩺 Risk prediction with confidence score
* 📈 Model performance metrics (Accuracy & ROC-AUC)

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── train_model.py
├── app.py
├── model.pkl
├── scaler.pkl
├── feature_names.pkl
├── eda_plot.png
├── correlation_heatmap.png
├── evaluation_plot.png
├── feature_importance.png
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

* **Name:** UCI Heart Disease Dataset
* **Source:** UCI Machine Learning Repository (via `ucimlrepo`)
* **Samples:** 303
* **Features:** 13
* **Target:**

  * `0` → No Disease
  * `1` → Disease

---

## 🧾 Features Description

| Feature  | Description                       |
| -------- | --------------------------------- |
| age      | Age in years                      |
| sex      | Sex (1 = Male, 0 = Female)        |
| cp       | Chest pain type (0–3)             |
| trestbps | Resting blood pressure (mm Hg)    |
| chol     | Serum cholesterol (mg/dl)         |
| fbs      | Fasting blood sugar > 120 mg/dl   |
| restecg  | Resting ECG results (0–2)         |
| thalach  | Maximum heart rate achieved       |
| exang    | Exercise induced angina           |
| oldpeak  | ST depression induced by exercise |
| slope    | Slope of peak exercise ST segment |
| ca       | Number of major vessels (0–3)     |
| thal     | Thalassemia type                  |

---

## 🤖 Model Details

* **Algorithm:** Decision Tree Classifier
* **Max Depth:** 5
* **Min Samples Split:** 10
* **Preprocessing:** StandardScaler
* **Train/Test Split:** 80/20 (stratified)

### Evaluation Metrics:

* Accuracy
* ROC-AUC Score
* Confusion Matrix

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Train Model

```
python train_model.py
```

### Expected Output:

```
Dataset downloaded successfully
Model trained successfully

Accuracy : ~82%
ROC-AUC  : ~0.88

Model and scaler saved
```

### 4. Run Application

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🛠️ Workflow

```
Data Collection (UCI API)
        ↓
Data Cleaning
        ↓
EDA & Visualization
        ↓
Feature Scaling
        ↓
Model Training (Decision Tree)
        ↓
Evaluation (ROC, Accuracy, CM)
        ↓
Deployment (Streamlit)
```

---

## 📦 Requirements

```
numpy
pandas
scikit-learn
streamlit
matplotlib
seaborn
ucimlrepo
```
