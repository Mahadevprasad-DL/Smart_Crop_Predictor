import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report

# Load Dataset
df = pd.read_csv("Crop_recommendation.csv")

# Encode labels
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# Prepare data
X = df.drop(columns=["label", "label_encoded"])
y = df["label_encoded"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test)) * 100

# Train KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test)) * 100
class_report=classification_report(y_test,knn_model.predict(X_test))

# Save Models, Label Encoder, and Scaler
pickle.dump(rf_model, open("crop_model_rf.pkl", "wb"))
pickle.dump(knn_model, open("crop_model_knn.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))
pickle.dump(scaler, open("feature_scaler.pkl", "wb"))

print(f"Random Forest Model Accuracy: {rf_accuracy:.2f}%")
print(f"KNN Model Accuracy: {knn_accuracy:.2f}%")
print(f"Classification Report :")
print(f"{class_report}")
print("Models saved successfully!")
