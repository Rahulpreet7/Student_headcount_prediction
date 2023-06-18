import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Day':2, 'Month':9, 'Year':6, 'Time': 18.5})

print(r.json())