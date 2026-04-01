import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load model and scaler
regmodel = pickle.load(open('regmodel (2).pkl', 'rb'))
scaler = pickle.load(open('scaling (1).pkl', 'rb'))

# Home route (HTML page)
@app.route('/')
def home():
    return render_template('home.html')


# Function to convert stress score → mental health
def get_mental_health(score):
    if score <= 2:
        return "Good"
    elif score <= 4:
        return "Moderate"
    else:
        return "Poor"


# API route
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']

        # Correct feature order (VERY IMPORTANT)
        feature_order = [
            'Aca_stage',
            'Peer_press',
            'Aca_press_home',
            'Study_environment',
            'Coping_strategy',
            'bad_habits',
            'Aca_comp_rate'
        ]

        # Convert input to array in correct order
        input_data = [data[i] for i in feature_order]
        input_data = np.array(input_data).reshape(1, -1)

        # Scale input
        new_data = scaler.transform(input_data)

        # Predict
        output = regmodel.predict(new_data)
        score = float(output[0])

        # Convert to mental health label
        mental_health = get_mental_health(score)

        # Return JSON
        return jsonify({
            "stress_score": score,
            "mental_health": mental_health
        })

    except Exception as e:
        # If any error occurs, return JSON instead of HTML
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)