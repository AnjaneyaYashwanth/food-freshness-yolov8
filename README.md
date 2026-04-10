# 🍌 Food Freshness Detection using Deep Learning

## 📌 Overview
This project presents an intelligent system for detecting fruit freshness using deep learning techniques. The system classifies fruits into **Fresh**, **Ripe**, and **Overripe** categories by analyzing visual features such as color, texture, and structure. Additionally, it provides a **freshness score (0–10)** for better interpretation.

---

## 🎯 Objectives
- Classify fruit freshness into:
  - Fresh
  - Ripe
  - Overripe
- Provide a quantitative freshness score (0–10)
- Compare different deep learning models
- Develop a user-friendly web application for real-time prediction

---

## 🧠 Models Implemented

### 🔹 1. Dual-Stream Model (Proposed Model)
- Separate feature extraction pipelines:
  - **RGB Stream** → captures color and ripeness
  - **Texture Stream (Edge + LBP)** → captures surface structure
- Feature fusion improves classification performance
- Achieved highest accuracy in experiments

---

### 🔹 2. 5-Channel Model
- Combines:
  - RGB (3 channels)
  - Edge (1 channel)
  - LBP (1 channel)
- Single-stream architecture
- Used for performance comparison

---

## 🏗️ System Workflow

1. Input image
2. Preprocessing:
   - Resize to 640×640
   - Edge detection (Canny)
   - LBP feature extraction
3. Model inference
4. Output:
   - Predicted class
   - Confidence score
   - Freshness score (0–10)

---

## ⚙️ Technologies Used

- Python
- PyTorch
- OpenCV
- Streamlit
- NumPy
- Scikit-learn
- Matplotlib

---

## 📊 Evaluation Metrics

The model performance is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## 📈 Results

- ✅ Dual-Stream Model Accuracy: **~98.2%**
- ✅ Strong generalization on test dataset
- ✅ Effective use of texture + color fusion
- ✅ Improved detection of overripe fruits using texture features

---

## 🌐 Web Application

A Streamlit-based web interface is developed for real-time usage.

### Features:
- Upload image
- Select model:
  - Dual Stream
  - 5-Channel
- View predictions instantly
- Displays:
  - Class label
  - Confidence
  - Freshness score
  - Class probabilities

---

### ▶️ Run the Web App

```bash
streamlit run app.py

## Updates
- Fixed minor documentation issues
- No changes made to code functionality
- Made relatable changes