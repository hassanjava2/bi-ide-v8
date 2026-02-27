"""Quick test runner script"""
import subprocess, sys, os
os.chdir(r"d:\bi-ide-v8")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "--tb=short", "-q", "--no-header", "-x", "--timeout=30"],
    capture_output=True, text=True, timeout=120, cwd=r"d:\bi-ide-v8",
    env={**os.environ, "PYTEST_RUNNING": "1", "PYTHONIOENCODING": "utf-8"}
)
print("=== STDOUT ===")
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
print("=== STDERR ===")
print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print(f"=== EXIT CODE: {result.returncode} ===")
