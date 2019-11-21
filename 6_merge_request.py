import requests
import json
import time



url = 'http://localhost:9873/v1/models/test_model:predict' 

input_x = [5]
input_y = [5]

data = {
    "signature_name":"predictions",
    "inputs":{
        "input_x":input_x,
        "input_y":input_y
        }
    }
start = time.time()
response = requests.post(url=url,data=json.dumps(data))
end = time.time()
time_cost = end - start
outputs = json.loads(response.text)
#outputs = response.text
print(outputs)
print("time cost:",time_cost)
