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

# Load a pre-trained model from a file
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

def predict_water_potability(model, input_data, true_labels=None, class_labels=None):
    # Check if input_data is a list, dict, DataFrame, or numpy array
    if isinstance(input_data, list):
        # Assuming each element in the list corresponds to a feature in the order defined by feature_names
        input_data_dict = dict(zip(feature_names, input_data))
        input_array = pd.DataFrame([input_data_dict], columns=feature_names).values
    elif isinstance(input_data, dict):
        # Convert the dictionary to a DataFrame and then to a 2D NumPy array
        input_array = pd.DataFrame([input_data], columns=feature_names).values
    elif isinstance(input_data, pd.DataFrame):
        # Convert the DataFrame to a 2D NumPy array
        input_array = input_data.values
    elif isinstance(input_data, np.ndarray):
        # Check if the array is 1D, and if so, reshape it to 2D
        input_array = input_data.reshape(1, -1) if len(input_data.shape) == 1 else input_data
    else:
        # If format not supported, raise an error
        raise ValueError("Unsupported input data format. Supported formats: list, dict, DataFrame, numpy array.")

    # Make prediction
    prediction = model.predict(xgb.DMatrix(input_array), output_margin=True)
    probability = 1.0 / (1.0 + np.exp(-prediction))
    print(f"Probability for class 1: {probability[0] * 100:.2f}%")

    # Map prediction to class label
    predicted_label = water_potability_labels[int(prediction[0] > 0.5)]

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
