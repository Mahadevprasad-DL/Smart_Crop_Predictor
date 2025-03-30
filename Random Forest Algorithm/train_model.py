import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("Crop_recommendation.csv")

# Encode labels
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# Prepare data
X = df.drop(columns=["label", "label_encoded"])
y = df["label_encoded"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41, stratify=y)

# Train Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=41)
rf_model.fit(X_train, y_train)
-
# Save Model and Label Encoder
pickle.dump(rf_model, open("crop_model.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

print("Model saved successfully!")

