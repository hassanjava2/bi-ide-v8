"""اختبار سريع للـ API"""
import requests
import json

# Test health
print("Testing health endpoint...")
try:
    r = requests.get("http://localhost:8000/health", timeout=5)
    print(f"Health: {r.status_code}")
    print(r.text[:200])
except Exception as e:
    print(f"Error: {e}")

# Test system status
print("\nTesting system status endpoint...")
try:
    r = requests.get("http://localhost:8000/api/v1/status", timeout=5)
    print(f"Status: {r.status_code}")
    print(r.text[:200])
except Exception as e:
    print(f"Error: {e}")

# Test RTX4090 status
print("\nTesting RTX4090 status endpoint...")
try:
    r = requests.get("http://localhost:8000/api/v1/rtx4090/status", timeout=5)
    print(f"RTX4090: {r.status_code}")
    print(r.text[:200])
except Exception as e:
    print(f"Error: {e}")

# Test council message
print("\nTesting council message...")
try:
    r = requests.post(
        "http://localhost:8000/api/v1/council/message",
        json={"message": "مرحبا", "user_id": "president"},
        timeout=10
    )
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text}")
except Exception as e:
    print(f"Error: {e}")
