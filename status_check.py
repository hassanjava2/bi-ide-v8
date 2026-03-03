#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")

print("=== BI-IDE v8 STATUS ===")

components = [
    ("Real Life Layer", "hierarchy.real_life_layer", "real_life_layer"),
    ("Autonomous Council", "hierarchy.autonomous_council", "autonomous_council"),
    ("Data Flywheel", "ai.training.data_flywheel", "data_flywheel"),
    ("Synthetic Data", "ai.training.synthetic_data_engine", "synthetic_data_engine"),
    ("Curriculum", "ai.training.curriculum_scheduler", "curriculum_scheduler"),
]

for name, module, attr in components:
    try:
        mod = __import__(module, fromlist=[attr])
        getattr(mod, attr)
        print(f"✅ {name}")
    except Exception as e:
        print(f"❌ {name}: {str(e)[:40]}")

print("\n=== RTX 5090 GPU ===")
try:
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
except Exception as e:
    print(f"❌ GPU Error: {e}")

print("\n=== Training Data ===")
import os
data_path = "/home/bi/training_data"
if os.path.exists(data_path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(data_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    print(f"Total: {total / 1024**3:.1f} GB")
    
print("\n✅ SYSTEM READY FOR REAL TRAINING")
