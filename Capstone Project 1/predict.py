import pickle
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import os


# Ustaw katalog roboczy na folder, gdzie znajduje się skrypt
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Użyj względnej ścieżki do pliku modelu
filename = 'model_wpp.pkl'

def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)
    
print("Bieżący katalog:", os.getcwd())

# Load a pre-trained model from a file (assuming 'model_mpp.pkl' is the file)
model = load('model_wpp.pkl')

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

    
    # Make predictions
    predictions = model.predict(input_array)
    probability = model.predict_proba(input_array)[:, 1]
    print(f"Probability for class 1: {probability[0] * 100:.2f}%")


    # Print predictions with class labels if provided
    if class_labels:
        for prediction in predictions:
            if 0 <= prediction < len(class_labels):
                print(f"Predicted water quality: '{class_labels[int(prediction)]}'")
            else:
                print("Invalid water quality prediction")

    # Print accuracy if true_labels are provided
    if true_labels is not None:
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Model accuracy: {accuracy * 100:.2f}%")

    return predictions


@app.route('/predict', methods=['POST'])
def predict():
    try:
        water_data = request.json
        print("Received client data:", water_data)

        # Assuming `true_labels` is available
        true_labels = [1]  # Replace with your true labels

        predictions = predict_water_potability(model, water_data, true_labels=true_labels, class_labels=water_potability_labels)

        result = {'water_quality': water_potability_labels[int(predictions[0])]}
        return jsonify(result)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'Error occurred during prediction'})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

