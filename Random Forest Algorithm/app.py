from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("crop_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("crop.html")

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

        # Make prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)[0]
        crop = label_encoder.inverse_transform([prediction])[0]

        return render_template("crop.html", prediction_text=f"Recommended Crop: {crop}")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

