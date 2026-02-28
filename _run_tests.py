"""Quick test runner script — cross-platform (macOS/Linux/Windows)"""
import subprocess, sys, os
from pathlib import Path

# Dynamic path — works on any OS
PROJECT_ROOT = str(Path(__file__).resolve().parent)
os.chdir(PROJECT_ROOT)

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "--tb=short", "-q", "--no-header", "-x"],
    capture_output=True, text=True, timeout=120, cwd=PROJECT_ROOT,
    env={**os.environ, "PYTEST_RUNNING": "1", "PYTHONIOENCODING": "utf-8"}
)
print("=== STDOUT ===")
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
print("=== STDERR ===")
print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print(f"=== EXIT CODE: {result.returncode} ===")
