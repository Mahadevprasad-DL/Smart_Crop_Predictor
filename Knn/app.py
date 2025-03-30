from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved models, label encoder, and scaler
rf_model = pickle.load(open("crop_model_rf.pkl", "rb"))
knn_model = pickle.load(open("crop_model_knn.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
scaler = pickle.load(open("feature_scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])
        selected_model = request.form["model"]

        # Prepare input data
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        # Make prediction based on selected model
        if selected_model == "rf":
            prediction = rf_model.predict(input_scaled)[0]
            model_name = "Random Forest"
            model_accuracy = round(rf_model.score(scaler.transform(input_data), [prediction]) * 100, 2)
        else:
            prediction = knn_model.predict(input_scaled)[0]
            model_name = "K-Nearest Neighbors (KNN)"
            model_accuracy = round(knn_model.score(scaler.transform(input_data), [prediction]) * 100, 2)

        # Get crop name
        crop = label_encoder.inverse_transform([prediction])[0]

        return render_template("index.html", 
                               prediction_text=f"Recommended Crop: {crop}", 
                               model_text=f"Model Used: {model_name}", 
                               accuracy_text=f"Model Accuracy: {model_accuracy}%")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
ṭ