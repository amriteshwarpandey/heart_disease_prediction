# heart_disease_prediction
# ❤️ Heart Disease Prediction using Logistic Regression
by AMRITESHWAR PANDEY ,VIT BHOPAL UNIVERSITY

This project uses **Logistic Regression**, a popular machine learning algorithm, to **predict the presence of heart disease** in patients based on medical attributes. The model is trained using the **Cleveland Heart Disease dataset** from the UCI Machine Learning Repository.

---

## 📊 Dataset: Processed Cleveland Data

- 🔗 [Dataset Source](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- Contains 13 features such as age, sex, chest pain type, blood pressure, cholesterol, etc.
- The target value is transformed into:
  - `0` → No heart disease
  - `1` → Presence of heart disease

---

## 🧰 Tools & Libraries Used

- Python 3.x
- Pandas
- NumPy
- scikit-learn

---

## 🚀 How it Works

1. **Load Dataset**  
   The `processed.cleveland.data` file is read using pandas.

2. **Preprocess Data**  
   - Missing values (`?`) replaced with `NaN`
   - Converted to numeric
   - Imputed using mean

3. **Transform Target**  
   Target values `>0` are considered as `1` (disease present), and `0` as no disease.

4. **Train/Test Split**  
   The data is split into 80% training and 20% testing sets.

5. **Standardize Features**  
   All features are scaled using `StandardScaler` for better model performance.

6. **Train Model**  
   Logistic Regression is trained on the dataset.

7. **Evaluate Model**  
   Accuracy, confusion matrix, classification report, and ROC-AUC score are calculated.

8. **Predict New Patient**  
   A new patient's data is scaled and passed through the model to predict disease presence.

---

## 🧪 Example Output

Accuracy: 85.25%
Confusion Matrix:
[[22 5]
[ 3 31]]
Classification Report:
precision recall f1-score support

markdown
Copy
Edit
       0       0.88      0.81      0.84        27
       1       0.86      0.91      0.88        34

accuracy                           0.85        61
macro avg 0.87 0.86 0.86 61
weighted avg 0.85 0.85 0.85 61

ROC-AUC Score: 0.86
Prediction: Heart Disease Detected
