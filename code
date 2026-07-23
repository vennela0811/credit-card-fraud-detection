# credit-card-fraud-detection
An end-to-end machine learning framework designed to detect fraudulent credit card transactions. Implements advanced data preprocessing and handles extreme class imbalance (SMOTE) to identify anomalies and protect financial transactions with high precision and recall.

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset (Kaggle Credit Card Fraud dataset) [cite: 107]
df = pd.read_csv("creditcard.csv")

# Step 4.2: Data Preprocessing
# Remove the 'Time' column as it is not needed [cite: 261]
df = df.drop(columns=['Time'])

# Scale the 'Amount' column to normalize values [cite: 260]
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 4.5: Model Training [cite: 336]
# Split data into 80% training and 20% testing [cite: 338]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier [cite: 341, 344]
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Predict on the test data [cite: 346]
y_pred = rf_model.predict(X_test)

# Display results [cite: 355]
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred)) # Shows True Positives, False Positives, etc. [cite: 358]
print("\nDetailed Report:")
print(classification_report(y_test, y_pred)) # Precision, Recall, and F1-score [cite: 361]

# Extract feature importance [cite: 272]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(5))

import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing the imbalance in the original dataset (Section 4.1)
plt.figure(figsize=(8, 5))
sns.countplot(x='Class', data=df) # 'df' is the original dataframe from Part 1
plt.title('Class Distribution ($0$: Normal, $1$: Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Model Evaluation: Confusion Matrix (Section 4.6)
cm = confusion_matrix(y_test, y_pred) # 'y_test' and 'y_pred' from Part 4

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1']) 
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Insights: Feature Importance Analysis (Section 4.3)
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df) 
plt.title('Top 10 Important Features for Fraud Detection')
plt.show()
