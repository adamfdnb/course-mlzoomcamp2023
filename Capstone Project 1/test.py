import requests

url = "http://localhost:9696/predict"

water_data = {
  'ph': 7.442023,
  'hardness': 1.9476,
  'solids': 3.4565,
  'chloramines': 8.493347,
  'sulfate': 2.9483,
  'conductivity': 3.50085,
  'organic_carbon': 1.89620,
  'trihalomethanes': 7.9958,
  'turbidity': 2.894651
}

response = requests.post(url, json=water_data).json()

print(water_data)
print(response)