import requests

web = "water-potability-2oqj.onrender.com"

url = f'http://{web}/predict'

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

# water_data = [7.481397771916631,186.07476549927424,13781.780283955726,4.649253929836745,377.3443718640398,323.7728426252204,11.687123837165814,66.2916498377711,4.039976289796705]

response = requests.post(url, json=water_data).json()

print(water_data)
print(response)
