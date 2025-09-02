# 🌱 Crop Recommendation System (Streamlit + KNN)

## 📌 Overview  
This project is a **Streamlit web app** that recommends the most suitable crop based on soil nutrients, weather conditions, and district.  
It uses the **K-Nearest Neighbors (KNN)** algorithm from **scikit-learn** to classify crops.  

---

## ⚡ Features  
- Directly loads `crop.csv` (no file upload required)  
- Interactive **Streamlit web interface**  
- Adjustable **KNN neighbors (`k`)** from sidebar  
- Model evaluation with:  
  - Train & Test Accuracy  
  - Precision, Recall, F1 Score  
  - Confusion Matrix Heatmap  
- User input form for custom crop prediction  

---

## 🛠️ Tech Stack  
- Python 3.9+  
- Streamlit – UI framework  
- Scikit-learn – ML model & preprocessing  
- Pandas & Numpy – Data handling  
- Seaborn & Matplotlib – Visualization  

---

## 📂 Dataset  
The dataset used is **`crop.csv`**.  
It contains the following features:  

- **N** – Nitrogen content in soil  
- **P** – Phosphorus content in soil  
- **K** – Potassium content in soil  
- **temperature** – Temperature (°C)  
- **humidity** – Humidity (%)  
- **ph** – Soil pH value  
- **rainfall** – Rainfall (mm)  
- **district** – District name  
- **label** – Recommended crop (target variable)
## 📊 Example Output

Model scores (Accuracy, Precision, Recall, F1 Score)

Confusion matrix heatmap

Recommended crop for given soil & weather conditions
