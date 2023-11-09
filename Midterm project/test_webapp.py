import requests

web = "milk-quality-prediction-uqjr.onrender.com"

url = f'http://{web}/predict'

milk_data = {
    "ph": 3.8,
    "temperature": 38,
    "taste": 0,
    "odor": 1,
    "fat": 0,
    "turbidity": 1,
    "colour": 254,
}

response = requests.post(url, json=milk_data).json()

print(milk_data)
print(response)
