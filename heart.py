# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Define column names
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'is', 'restecg', 'thalach',
           'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Load the dataset
file_path = r"C:\\Users\\pande\\Desktop\\c++\\.vscode\\processed.cleveland.data"
data = pd.read_csv(file_path, header=None, names=columns)

# Data Preprocessing
# Replace missing values (denoted by '?') with NaN
data = data.replace('?', np.nan)

# Convert data to numeric and handle missing values (e.g., using mean for simplicity)
data = data.apply(pd.to_numeric)
data.fillna(data.mean(), inplace=True)

# Feature and target separation
X = data.drop('target', axis=1)  # Features
y = data['target']  # Target

# Binarize the target: 1 if disease present, 0 if no disease (target > 0 is considered disease presence)
y = y.apply(lambda x: 1 if x > 0 else 0)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting the test set results
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Output results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# New patient data for prediction
new_patient = pd.DataFrame([[67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0

]],
                           columns=X.columns)

# Scale the new patient data
new_patient_scaled = scaler.transform(new_patient)

# Predict using the trained model
prediction = model.predict(new_patient_scaled)

# Output the prediction
if prediction[0] == 1:
    print("Prediction: Heart Disease Detected")
else:
    print("Prediction: No Heart Disease")
