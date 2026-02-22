# Test Group Discussion API
import requests
import json

topic = "Market Expansion Strategy"
print("=== Testing Group Discussion ===")
print(f"Topic: {topic}\n")

r = requests.post(
    'http://localhost:8000/api/v1/council/discuss',
    json={'topic': topic},
    timeout=30
)

print(f"Status: {r.status_code}")
data = r.json()
print(f"Participants: {data.get('participants')}\n")

for i, item in enumerate(data.get('discussion', []), 1):
    print(f"{i}. {item['wise_man']} ({item['role']}):")
    print(f"   {item['response'][:60]}...\n")
