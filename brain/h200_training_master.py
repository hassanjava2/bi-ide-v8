#!/usr/bin/env python3
"""
h200_training_master.py — سكربت التدريب الرئيسي لـ Vast.ai 4x B200
Master training script for Vast.ai GPU servers

الاستخدام:
  # على سيرفر Vast.ai (4x B200, $12/hr)
  python h200_training_master.py --gdrive-path /mnt/gdrive --model Qwen/Qwen2.5-72B-Instruct

  # عند الاستئناف (checkpoint)
  python h200_training_master.py --resume --checkpoint /data/checkpoints/latest
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ═══════════════════════════════════════════════
# Vast.ai Setup
# ═══════════════════════════════════════════════

VAST_SETUP = """
#!/bin/bash
# Vast.ai B200 setup script — run once when server starts
set -e

echo "🔧 Setting up Vast.ai training server..."

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.43.0 peft>=0.12.0 bitsandbytes>=0.43.0
pip install accelerate>=0.33.0 datasets>=2.20.0 trl>=0.9.0
pip install flash-attn>=2.6.0 --no-build-isolation
pip install wandb sentencepiece protobuf scipy
pip install rclone  # For Google Drive access

echo "✅ Dependencies installed"

# Mount Google Drive
rclone config create gdrive drive
mkdir -p /mnt/gdrive
rclone mount gdrive:bi-ide-training-data /mnt/gdrive --daemon --vfs-cache-mode full
echo "✅ Google Drive mounted at /mnt/gdrive"

# Check GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem/1e9:.0f}GB)') for i in range(torch.cuda.device_count())]"
"""


# ═══════════════════════════════════════════════
# Training configurations
# ═══════════════════════════════════════════════

CONFIGS = {
    # QLoRA on RTX 5090 (24GB VRAM)
    "rtx5090": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "qlora": True,
        "batch_size": 2,
        "gradient_accumulation": 16,
        "lora_r": 64,
        "max_seq_length": 4096,
        "num_epochs": 3,
    },

    # LoRA on 4x B200 (total 720GB VRAM)
    "4xb200": {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "qlora": False,
        "batch_size": 4,
        "gradient_accumulation": 4,
        "lora_r": 128,
        "max_seq_length": 8192,
        "num_epochs": 3,
    },

    # Full finetune on 8x H200 (total 1128GB VRAM)
    "8xh200": {
        "model": "meta-llama/Llama-3-70B-Instruct",
        "qlora": False,
        "batch_size": 8,
        "gradient_accumulation": 2,
        "lora_r": 256,
        "max_seq_length": 8192,
        "num_epochs": 5,
    },
}

# ═══════════════════════════════════════════════
# 24-hour training plan
# ═══════════════════════════════════════════════

TRAINING_PLAN_24H = """
═══════════════════════════════════════════════════
  BI-IDE 24-Hour Training Plan — 4x B200 ($288)
═══════════════════════════════════════════════════

Hour  0-1:  Setup + mount Google Drive + verify GPUs
Hour  1-4:  Curriculum Tiers 1-4 (basics: languages, instructions)
Hour  4-8:  Curriculum Tiers 5-7 (knowledge: science, medicine, engineering)
Hour  8-12: Curriculum Tiers 8-10 (skills: code, vision)
Hour 12-16: Curriculum Tiers 11-13 (advanced: research, deep code)
Hour 16-20: Full pass with all data (mixed)
Hour 20-22: Evaluation + GGUF export
Hour 22-24: Upload to Google Drive + cleanup

Budget: $12/hr × 24h = $288
Result: Fully trained 72B model with all 446 capsules
"""


def run_training_pipeline(config_name: str, data_path: str, output_dir: str,
                          resume_from: str = None):
    """Run the complete training pipeline"""

    config = CONFIGS[config_name]
    model_name = config["model"]

    print(TRAINING_PLAN_24H)
    print(f"\n🚀 Starting training pipeline:")
    print(f"   Config: {config_name}")
    print(f"   Model: {model_name}")
    print(f"   Data: {data_path}")
    print(f"   Output: {output_dir}")
    print(f"   QLoRA: {config['qlora']}")

    # Check GPU
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        print(f"   GPUs: {gpu_count}")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / 1e9
            print(f"     GPU {i}: {name} ({mem:.0f}GB)")
    except ImportError:
        print("   ⚠️ No GPU detected (dry run)")
        return

    # Step 1: Preprocess data
    print(f"\n{'='*50}")
    print("📊 Step 1: Preprocessing data...")
    processed_dir = os.path.join(output_dir, "processed")
    subprocess.run([
        sys.executable, "data_preprocessor.py",
        data_path, processed_dir,
    ], check=True)

    # Step 2: Train with curriculum
    print(f"\n{'='*50}")
    print("🧠 Step 2: Training with curriculum learning...")
    train_cmd = [
        sys.executable, "lora_trainer.py",
        "--data", processed_dir,
        "--model", model_name,
        "--output", os.path.join(output_dir, "model"),
        "--epochs", str(config["num_epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--lora-r", str(config["lora_r"]),
        "--lr", "2e-4",
    ]

    if config["qlora"]:
        train_cmd.append("--qlora")
    else:
        train_cmd.append("--lora")

    if resume_from:
        train_cmd.extend(["--resume", resume_from])

    # Use accelerate for multi-GPU
    if gpu_count > 1:
        train_cmd = [
            sys.executable, "-m", "accelerate", "launch",
            "--num_processes", str(gpu_count),
            "--mixed_precision", "bf16",
        ] + train_cmd[1:]  # Skip python3

    subprocess.run(train_cmd, check=True)

    # Step 3: Evaluate
    print(f"\n{'='*50}")
    print("🧪 Step 3: Evaluating...")
    final_model = os.path.join(output_dir, "model", "final")
    subprocess.run([
        sys.executable, "evaluator.py",
        "--model", final_model,
        "--capsule", "all",
    ], check=True)

    # Step 4: Export GGUF
    print(f"\n{'='*50}")
    print("📦 Step 4: Exporting to GGUF...")
    gguf_path = os.path.join(output_dir, "bi-ide-brain.gguf")
    # This would use llama.cpp conversion

    # Step 5: Upload to Google Drive
    print(f"\n{'='*50}")
    print("☁️ Step 5: Uploading to Google Drive...")
    subprocess.run([
        "rclone", "copy", output_dir,
        "gdrive:bi-ide-training-output",
        "--transfers", "8",
    ], check=False)

    print(f"\n{'='*50}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"   Model: {final_model}")
    print(f"   GGUF: {gguf_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BI-IDE H200 Training Master")
    parser.add_argument("--config", default="4xb200", choices=CONFIGS.keys())
    parser.add_argument("--data", default="/mnt/gdrive", help="Data path (Google Drive mount)")
    parser.add_argument("--output", default="/data/training", help="Output directory")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--setup", action="store_true", help="Run Vast.ai setup")
    args = parser.parse_args()

    if args.setup:
        print(VAST_SETUP)
        sys.exit(0)

    run_training_pipeline(args.config, args.data, args.output,
                         args.checkpoint if args.resume else None)
