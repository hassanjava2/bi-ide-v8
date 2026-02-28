#!/usr/bin/env python3
"""
GPU Training Wrapper for BI-IDE Worker
Uses the native AutoLearningSystem (auto_learning_system.py) for each layer.

Usage:
    python gpu_trainer.py --layer council --epochs 50 --server https://bi-iq.com --job-id abc123
"""

import os, sys, json, time, argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path so we can import hierarchy modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_GPU = torch.cuda.is_available()

# Layer name → specialization mapping (from AutoLearningSystem)
LAYER_SPECS = {
    "president": "📊 Strategic Decision Making from Global News",
    "seventh_dimension": "🔮 Long-term Future Planning & Trends",
    "high_council": "🧠 Collective Wisdom & Governance",
    "shadow_light": "⚖️ Risk Assessment & Opportunity Analysis",
    "scouts": "🔍 Intelligence Gathering & Research",
    "meta_team": "⚙️ System Optimization & Performance",
    "domain_experts": "🎓 Multi-Domain Expert Knowledge",
    "execution": "🚀 Task Execution & Project Management",
    "meta_architect": "🏗️ System Architecture & Design Patterns",
    "builder_council": "🔨 Software Development & Engineering",
    "executive_controller": "🎮 Executive Control & Command",
    "guardian": "🛡️ Cybersecurity & Threat Detection",
    "cosmic_bridge": "🌌 External API Integration & Data",
    "eternity": "💾 Knowledge Preservation & Memory",
    "learning_core": "🧬 Continuous Self-Improvement",
    # Aliases from training_coordinator.py
    "council": "🧠 Collective Wisdom & Governance",
    "erp_accounting": "📊 Financial Accounting & Reports",
    "erp_inventory": "📦 Inventory Management & Tracking",
    "code_generation": "💻 Software Development & Engineering",
    "copilot": "🤖 AI Code Assistant & Completion",
    "balance": "⚖️ Risk Assessment & Opportunity Analysis",
    "security": "🛡️ Cybersecurity & Threat Detection",
    "compliance": "📋 Regulatory Compliance & Auditing",
}


def report_progress(server, token, job_id, metrics):
    """Report training progress to orchestrator."""
    try:
        import requests
        requests.post(
            f"{server}/api/v1/orchestrator/jobs/{job_id}/log",
            params={"line": json.dumps(metrics, ensure_ascii=False)},
            headers={"X-Orchestrator-Token": token},
            timeout=5,
        )
    except Exception:
        pass


def report_completion(server, token, job_id, worker_id, final_metrics):
    """Report job completion to orchestrator."""
    try:
        import requests
        requests.post(
            f"{server}/api/v1/orchestrator/jobs/{job_id}/complete",
            params={"worker_id": worker_id or "unknown"},
            json=final_metrics,
            headers={"X-Orchestrator-Token": token},
            timeout=10,
        )
    except Exception:
        pass


def update_council(server, token, layer_name, metrics):
    """Push training results to the council."""
    try:
        import requests
        requests.post(
            f"{server}/api/v1/council/training-update",
            json={
                "layer": layer_name,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            },
            headers={"X-Orchestrator-Token": token} if token else {},
            timeout=10,
        )
        print(f"🧠 Council updated with {layer_name} results")
    except Exception:
        pass

    # Also save results locally
    results_dir = Path("./training_results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"{layer_name}_results.json"
    try:
        existing = json.loads(results_file.read_text()) if results_file.exists() else []
        existing.append({**metrics, "timestamp": datetime.now().isoformat()})
        results_file.write_text(json.dumps(existing[-50:], ensure_ascii=False, indent=2))
    except Exception:
        pass


def save_checkpoint(model, optimizer, epoch, loss, layer_name, stats):
    """Save training checkpoint."""
    checkpoint_dir = Path(f"./checkpoints/{layer_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'stats': stats,
        'layer_name': layer_name,
        'timestamp': datetime.now().isoformat(),
    }
    torch.save(state, checkpoint_dir / "latest.pt")
    print(f"💾 Checkpoint saved: {layer_name} (epoch {epoch}, loss={loss:.4f})")


def load_checkpoint(model, optimizer, layer_name):
    """Try to load existing checkpoint."""
    checkpoint_dir = Path(f"./checkpoints/{layer_name}")
    ckpt_path = checkpoint_dir / "latest.pt"
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print(f"📂 Resumed {layer_name} from epoch {ckpt.get('epoch', 0)}")
            return ckpt.get('epoch', 0)
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="BI-IDE GPU Training Wrapper")
    parser.add_argument("--layer", type=str, required=True, help="Layer name to train")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--server", type=str, default=None, help="Orchestrator server URL")
    parser.add_argument("--token", type=str, default=None, help="Orchestrator token")
    parser.add_argument("--job-id", type=str, default=None, help="Job ID for reporting")
    parser.add_argument("--worker-id", type=str, default=None, help="Worker ID")
    args = parser.parse_args()

    layer_name = args.layer
    specialization = LAYER_SPECS.get(layer_name, f"🎯 {layer_name} Training")

    # ─── Print header ───
    print(f"\n{'='*60}")
    print(f"🧠 BI-IDE trainer — {layer_name}")
    print(f"   📋 {specialization}")
    print(f"   Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if IS_GPU else ""))
    if IS_GPU:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   VRAM:   {vram:.1f} GB")
    print(f"   Epochs: {args.epochs}")
    print(f"{'='*60}")

    # ─── Use the native SmartTrainer ───
    # Import only the class, NOT the module-level singleton
    import importlib.util
    trainer_module_path = Path(__file__).parent / "auto_learning_system.py"
    if not trainer_module_path.exists():
        # Try alternate paths
        for p in [
            Path("./hierarchy/auto_learning_system.py"),
            Path("./auto_learning_system.py"),
            Path(__file__).parent.parent / "hierarchy" / "auto_learning_system.py",
        ]:
            if p.exists():
                trainer_module_path = p
                break

    # Load module without executing the singleton at the bottom
    import types
    spec = importlib.util.spec_from_file_location("auto_learning_system_isolated", str(trainer_module_path))
    mod = importlib.util.module_from_spec(spec)
    # Prevent singleton creation by monkey-patching __name__
    mod.__name__ = "auto_learning_system_isolated"
    
    # We'll load manually and extract SmartTrainer
    source_code = trainer_module_path.read_text()
    # Remove the singleton line at the bottom
    clean_source = source_code.replace(
        "auto_learning_system = AutoLearningSystem()",
        "auto_learning_system = None  # Disabled in worker mode"
    )
    exec(compile(clean_source, str(trainer_module_path), 'exec'), mod.__dict__)
    SmartTrainer = mod.SmartTrainer
    print(f"✅ Loaded SmartTrainer from {trainer_module_path.name}")

    trainer = SmartTrainer(layer_name, specialization)

    # Try to load existing checkpoint
    start_epoch = load_checkpoint(trainer.model, trainer.optimizer, layer_name)

    # ─── Scale batch size for GPU VRAM (but cap to dataset size) ───
    dataset_size = len(trainer.dataset)
    if IS_GPU:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 20:    # RTX 5090
            target_batch = min(128, max(1, dataset_size // 2))
        elif vram_gb >= 6:   # RTX 4050
            target_batch = min(64, max(1, dataset_size // 2))
        else:
            target_batch = min(32, max(1, dataset_size // 2))
    else:
        target_batch = min(16, max(1, dataset_size // 2))
    
    trainer.dataloader = torch.utils.data.DataLoader(
        trainer.dataset, batch_size=target_batch, shuffle=True,
        num_workers=0, pin_memory=IS_GPU,
    )
    print(f"   Batch:  {target_batch} (dataset: {dataset_size} samples)")

    param_count = sum(p.numel() for p in trainer.model.parameters())
    print(f"   Params: {param_count:,} ({param_count/1e6:.1f}M)")
    print(f"{'='*60}\n")

    # ─── Training loop ───
    total_start = time.time()
    best_loss = float('inf')

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()
        trainer.train_epoch()
        epoch_time = time.time() - epoch_start

        loss = trainer.stats["loss"]
        accuracy = trainer.stats["accuracy"]

        # Save checkpoint if best
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(trainer.model, trainer.optimizer, epoch, loss, layer_name, trainer.stats)

        # Print progress
        bar = f"{'█' * int(30 * (epoch - start_epoch + 1) / args.epochs)}{'░' * (30 - int(30 * (epoch - start_epoch + 1) / args.epochs))}"
        print(f"📊 [{bar}] Epoch {epoch+1}/{start_epoch + args.epochs} | "
              f"Loss={loss:.4f} | Acc={accuracy:.1f}% | {epoch_time:.1f}s")

        # Report to orchestrator every 5 epochs
        if args.server and args.job_id and (epoch + 1) % 5 == 0:
            report_progress(args.server, args.token, args.job_id, {
                "epoch": epoch + 1,
                "loss": round(loss, 6),
                "accuracy": round(accuracy, 2),
                "epoch_time": round(epoch_time, 1),
                "layer": layer_name,
            })

    total_time = time.time() - total_start

    # ─── Final report ───
    final_metrics = {
        "layer": layer_name,
        "specialization": specialization,
        "total_epochs": args.epochs,
        "best_loss": round(best_loss, 6),
        "final_accuracy": round(accuracy, 2),
        "total_time_sec": round(total_time, 1),
        "param_count": param_count,
        "device": str(DEVICE),
        "samples_trained": trainer.stats["samples"],
        "data_fetches": trainer.stats["data_fetches"],
    }

    print(f"\n{'='*60}")
    print(f"✅ Training complete: {layer_name}")
    print(f"   Best Loss:  {best_loss:.4f}")
    print(f"   Accuracy:   {accuracy:.1f}%")
    print(f"   Samples:    {trainer.stats['samples']:,}")
    print(f"   Total Time: {total_time/60:.1f} min")
    print(f"{'='*60}")

    # Update council with results
    if args.server:
        update_council(args.server, args.token, layer_name, final_metrics)
        report_completion(args.server, args.token, args.job_id, args.worker_id, final_metrics)

    sys.exit(0)


if __name__ == "__main__":
    main()
