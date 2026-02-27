"""
Root conftest.py - pytest configuration
"""
import sys
import os

# Set environment for tests
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTEST_RUNNING"] = "1"

# Mark that we're running from test
sys._called_from_test = True
