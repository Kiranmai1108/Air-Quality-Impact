from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
try:
    model = joblib.load("classification_model.pkl")
except FileNotFoundError:
    model = None
    print("⚠️ classification_model.pkl not found in root folder")

# Label map
label_map = {
    0: ("Very High", "very-high"),
    1: ("High", "high"),
    2: ("Moderate", "moderate"),
    3: ("Low", "low"),
    4: ("Very Low", "very-low")
}

features_order = ["AQI", "PM10", "PM2_5", "NO2", "SO2", "O3", "Temperature", "humidity", "windspeed"]

@app.route("/")
def home():
    return render_template("home.html", title="Home")

@app.route("/about")
def about():
    return render_template("about.html", title="About")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    predicted_label = None
    badge_class = None
    error = None

    if request.method == "POST":
        if model is None:
            error = "Model file not found. Place classification_model.pkl in root directory."
        else:
            try:
                # Get inputs in the correct order
                values = [float(request.form[f]) for f in features_order]
                arr = np.array(values).reshape(1, -1)
                pred = int(model.predict(arr)[0])
                predicted_label, badge_class = label_map.get(pred, ("Unknown", "unknown"))
            except Exception as e:
                error = f"Invalid input: {e}"

    return render_template("predict.html",
                           title="Predict",
                           predicted_label=predicted_label,
                           badge_class=badge_class,
                           error=error)

@app.route("/insights")
def insights():
    return render_template("insights.html", title="Insights")

@app.route("/tips")
def tips():
    return render_template("tips.html", title="Health Tips")

if __name__ == "__main__":
    app.run(debug=True)
