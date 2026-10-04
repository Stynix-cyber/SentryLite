import json

data = '{"name": "test"}'
result = json.loads(data)

print(result["name"])
