import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import os
import json

# Klasa NumpyEncoder do obsługi konwersji obiektów NumPy do formatu JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

# Ustaw katalog roboczy na folder, gdzie znajduje się skrypt
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Użyj względnej ścieżki do pliku modelu
model_filename = 'model_wpp.model'

def load_xgb_model(filename: str):
    booster = xgb.Booster()
    booster.load_model(filename)
    return booster

print("Bieżący katalog:", os.getcwd())

# Load a pre-trained model from a file (assuming 'model_mpp.pkl' is the file)
xgb_model = load_xgb_model(model_filename)

app = Flask('water_quality_probability')

# Manually define feature names
# You can customize these names based on your use case.
feature_names = [
    'ph',
    'hardness',
    'solids',
    'chloramines',
    'sulfate',
    'conductivity',
    'organic_carbon',
    'trihalomethanes',
    'turbidity'
]
# Assuming you have a trained XGBoost model (xgb_model) and water quality labels
water_potability_labels = ["undrinkable", "drinkable"]

def predict_water_potability(model, input_data, class_labels=None):
    # Convert input_data to DataFrame if it's not already
    if not isinstance(input_data, pd.DataFrame):
        if isinstance(input_data, list):
            input_data = [input_data]
        input_df = pd.DataFrame(input_data, columns=feature_names, index=[0])
    else:
        input_df = input_data

    # Create DMatrix for XGBoost
    dmatrix = xgb.DMatrix(input_df)

    # Make prediction
    prediction = model.predict(dmatrix, output_margin=True)
    probability = 1.0 / (1.0 + np.exp(-prediction))
    print(f"Probability for class 1: {probability[0] * 100:.2f}%")

    # Map prediction to class label
    predicted_label = water_potability_labels[int(prediction[0])]

    return predicted_label, probability[0]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        water_data = request.json
        print("Received client data:", water_data)

        # Make prediction
        predicted_label, probability = predict_water_potability(xgb_model, water_data, class_labels=water_potability_labels)

        # Konwersja wyników do formatu JSON
        result = {'water_quality': predicted_label, 'probability': probability}
        return json.dumps(result, cls=NumpyEncoder)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'Error occurred during prediction'})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
