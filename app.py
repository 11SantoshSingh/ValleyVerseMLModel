from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.preprocessing import LabelEncoder
import logging

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # Adjust origin as needed

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the saved pipeline dictionary
loaded_pipeline = joblib.load("tourism_recommender_pipeline2.pkl")

def safe_transform(le: LabelEncoder, value: str, label_name: str):
    if value not in le.classes_:
        raise ValueError(f"❌ '{value}' is not a known {label_name}. Valid values are: {list(le.classes_)}")
    return le.transform([value])[0]

@app.route('/')
def home():
    return "✅ Flask API is running."

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json(force=True)
        app.logger.info(f"Received data: {data}")

        interest = data.get("interest")
        age_group = data.get("age_group")
        weather = data.get("weather")
        duration = data.get("duration")

        if not all([interest, age_group, weather, duration]):
            return jsonify({"error": "Missing one or more input fields"}), 400

        X_input = [[
            safe_transform(loaded_pipeline["le_interest"], interest, "interest"),
            safe_transform(loaded_pipeline["le_age"], age_group, "age_group"),
            safe_transform(loaded_pipeline["le_weather"], weather, "weather"),
            safe_transform(loaded_pipeline["le_duration"], duration, "duration")
        ]]

        model = loaded_pipeline["model"]
        y_encoded = model.predict(X_input)[0]
        destination = loaded_pipeline["le_destination"].inverse_transform([y_encoded])[0]

        app.logger.info(f"Predicted destination: {destination}")

        return jsonify({"destination": destination})

    except ValueError as ve:
        app.logger.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Exception: {e}")
        return jsonify({"error": "Something went wrong: " + str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
