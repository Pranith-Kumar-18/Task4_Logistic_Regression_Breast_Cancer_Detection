import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Step 1: Load the dataset
df = pd.read_csv('data.csv')
print("✅ Dataset loaded!")

# Step 2: Drop useless column
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

# Step 3: Encode 'diagnosis' (B = 0, M = 1)
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Step 4: Split data
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Step 5: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Step 9: Evaluate
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"🎯 ROC-AUC Score: {roc_auc:.2f}")

# Step 10: Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('🚦 ROC Curve')
plt.legend()
plt.show()

input("✅ Press Enter to exit...")
