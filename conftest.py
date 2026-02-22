"""
Root conftest.py - Runs BEFORE any other conftest or test collection.
Sets up environment to prevent encoding_fix from breaking pytest.
"""
import sys
import os
import types

# ═══════════════════════════════════════════════════
# CRITICAL: Must run before ANY module imports encoding_fix
# ═══════════════════════════════════════════════════
os.environ["PYTEST_RUNNING"] = "1"
sys._called_from_test = True
sys._encoding_fix_applied = True

# Pre-load a dummy encoding_fix into sys.modules
# so that any `import encoding_fix` gets this instead of the real file
encoding_fix_module = types.ModuleType("encoding_fix")
encoding_fix_module.safe_print = print  # type: ignore
encoding_fix_module._PYTEST_RUNNING = True
sys.modules["encoding_fix"] = encoding_fix_module
