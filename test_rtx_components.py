#!/usr/bin/env python3
"""Test components on RTX 5090"""
import sys
sys.path.insert(0, ".")

print("Testing new components on RTX 5090...")
print("=" * 60)

try:
    from hierarchy.real_life_layer import real_life_layer
    print(f"✅ Real Life Layer: {len(real_life_layer.agents)} agents")
except Exception as e:
    print(f"❌ Real Life Layer: {e}")

try:
    from hierarchy.autonomous_council import autonomous_council
    print(f"✅ Autonomous Council: {len(autonomous_council.members)} sages")
except Exception as e:
    print(f"❌ Autonomous Council: {e}")

try:
    from ai.training.data_flywheel import data_flywheel
    print(f"✅ Data Flywheel: Ready")
except Exception as e:
    print(f"❌ Data Flywheel: {e}")

try:
    from ai.training.synthetic_data_engine import synthetic_data_engine
    print(f"✅ Synthetic Data Engine: Ready")
except Exception as e:
    print(f"❌ Synthetic Data Engine: {e}")

try:
    from ai.training.curriculum_scheduler import curriculum_scheduler
    print(f"✅ Curriculum Scheduler: Ready")
except Exception as e:
    print(f"❌ Curriculum Scheduler: {e}")

try:
    from ai.training.knowledge_distillation_pipeline import distillation_pipeline
    print(f"✅ Knowledge Distillation: Ready")
except Exception as e:
    print(f"❌ Knowledge Distillation: {e}")

print("=" * 60)
print("🚀 RTX 5090 Components Ready!")
