import requests

url = "http://localhost:9696/predict"

client = {"job": "retired", "duration": 445, "poutcome": "success"}
response = requests.post(url, json=client).json()

try:
    response = requests.post(url, json=client)
    response.raise_for_status()  
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Request Exception: {e}")
except Exception as e:
    print(f"Other Exception: {e}")