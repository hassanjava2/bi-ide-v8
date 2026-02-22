# Test Council Discussion
import requests
import json

topic = "Market Expansion Strategy"
print("=== Council Discussion ===\n")

r = requests.post(
    'http://localhost:8000/api/v1/council/discuss',
    json={'topic': topic},
    timeout=30
)

data = r.json()
print(f"Topic: {data.get('topic')}")
print(f"Participants: {data.get('participants')}\n")

for item in data.get('discussion', []):
    print(f"{item['wise_man']} ({item['role']}):")
    print(f"  {item['response']}\n")
