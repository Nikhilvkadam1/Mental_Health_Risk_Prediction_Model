# 🧠 Mental Health Risk Prediction Model

This project predicts an individual's **mental health risk level** — *Low*, *Medium*, or *High* — based on lifestyle and well-being factors such as age, sleep hours, work hours, stress level, and mood rating.

## 📌 Key Features

- Interactive console input for personal risk level prediction
- Data preprocessing: standard scaling and label encoding
- Trains and compares 4 machine learning models:
  - Decision Tree (selected as best model)
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- Accuracy evaluation using 5-fold cross-validation
- Visual comparison of model performance using a seaborn bar plot
- Summary of average lifestyle factors by risk level

## 📊 Input Features

- `Age`
- `Sleep Hours` (avg per day)
- `Work Hours` (avg per day)
- `Stress Level` (1–10)
- `Mood Rating` (1–10)

## 🧠 Output

- Predicted **Mental Health Risk Level**
- Model performance metrics (accuracy, CV score)
- Bar plot comparing accuracy of models
- Mean behavior trends for each risk group

## 📦 Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
