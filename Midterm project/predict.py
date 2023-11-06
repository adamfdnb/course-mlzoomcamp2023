import pickle
from flask import Flask, request, jsonify
from sklearn.feature_extraction import DictVectorizer

def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)

# Initialize a DictVectorizer object (dv)
dv = DictVectorizer()

# Load a pre-trained DictVectorizer from a file (assuming 'dv.bin' is the file)
dv = load('dv.pkl')

# Load a pre-trained model from a file (assuming 'model_mqp.pkl' is the file)
model = load('model_mqp.pkl')

app = Flask('milk_quality_probability')

# Manually define feature names
# You can customize these names based on your use case.
feature_names = [
    'ph',
    'temperature',
    'taste',
    'odor',
    'fat',
    'turbidity',
    'colour'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        milk_data = request.json
        print("Received client data:", milk_data)
        
        X_milk = dv.transform([milk_data])
        
        print("Transformed data:", X_milk)
        probs = model.predict(X_milk)
        probability = probs[0]
        print("Probability for class 1:", probability)

        # Define milk quality labels
        milk_quality_labels = ["Bad", "Moderate", "Good"]

        # Determine milk quality prediction based on probability
        if 0 <= probability < len(milk_quality_labels):
            milk_quality_prediction = milk_quality_labels[int(probability)]
            print(f"The milk quality is '{milk_quality_prediction}'")
            result = {'milk_quality': milk_quality_prediction}
            return jsonify(result)
        else:
            print("Invalid milk quality prediction")
            return jsonify({'error': 'Invalid milk quality prediction'})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
