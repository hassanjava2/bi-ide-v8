#!/usr/bin/env python3
"""
brain_daemon.py — المايسترو 🎭

الحلقة الرئيسية التي لا تتوقف أبداً:
  1. الكشافة تجيب بيانات جديدة لكل الكبسولات
  2. يدرّب كل كبسولة عندها بيانات جديدة كافية
  3. المصنع يولّد كبسولات أطفال من الأقوياء
  4. دولاب البيانات — حذف البيانات المتدرّب عليها لتفريغ مساحة
  5. يكرر من 1 ← ∞

التشغيل:
  python3 brain/brain_daemon.py
  أو:
  nohup python3 brain/brain_daemon.py > /tmp/brain_daemon.log 2>&1 &
"""

import sys
import os
import shutil
import json
import time
import gc
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from brain.brain_factory import BrainFactory
from brain.knowledge_scout import KnowledgeScout

LOG_PATH = Path("/tmp/brain_daemon.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_PATH)),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("daemon")

# ═══════════════════════════════════════════════════════════
# الإعدادات
# ═══════════════════════════════════════════════════════════

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
CAPSULES_DIR = PROJECT_ROOT / "capsules"
MIN_SAMPLES_TO_TRAIN = 50  # أقل عدد عينات للبدء بالتدريب
SCOUT_SAMPLES_PER_CYCLE = 10  # عينات كل دورة لكل كبسولة
EVOLVE_EVERY_N_CYCLES = 3  # كل 3 دورات يطوّر
PAUSE_BETWEEN_CYCLES = 30  # ثواني بين الدورات
CLEANUP_AFTER_TRAIN = True  # دولاب البيانات — حذف البيانات بعد التدريب


class BrainDaemon:
    """المايسترو — يدير الكل 24/7"""

    def __init__(self):
        self.factory = BrainFactory(CAPSULES_DIR)
        self.scout = KnowledgeScout(CAPSULES_DIR)
        self.cycle = 0
        self.total_trained = 0
        self.total_evolved = 0
        self.start_time = datetime.now()

    def _train_capsule(self, capsule_dir: Path, capsule_id: str) -> dict:
        """تدريب كبسولة واحدة"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from datasets import load_dataset

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        data_dir = capsule_dir / "data"
        model_dir = capsule_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # دمج كل JSONL
        merged = data_dir / "merged_train.jsonl"
        count = 0
        with open(merged, "w") as out:
            for f in data_dir.glob("*.jsonl"):
                if f.name == "merged_train.jsonl":
                    continue
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                            sample = {
                                "input_text": d.get("input_text", d.get("instruction", "")),
                                "output_text": d.get("output_text", d.get("output", "")),
                            }
                            if sample["input_text"] and sample["output_text"]:
                                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                                count += 1
                        except:
                            pass

        if count < MIN_SAMPLES_TO_TRAIN:
            return {"status": "skip", "reason": f"only {count} samples"}

        has_gpu = torch.cuda.is_available()
        max_steps = min(count * 2, 2000)

        logger.info(f"🏋️ Training {capsule_id}: {count} samples, {max_steps} steps")

        # الموديل الابتدائي: إذا الكبسولة ورثت موديل، نستخدمه
        model_source = BASE_MODEL
        if (model_dir / "config.json").exists():
            model_source = str(model_dir)
            logger.info(f"  → Continuing from inherited model")

        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16 if has_gpu else torch.float32,
            device_map="auto" if has_gpu else "cpu",
            trust_remote_code=True,
        )
        if has_gpu:
            model.gradient_checkpointing_enable()

        dataset = load_dataset("json", data_files=str(merged), split="train")

        def tokenize_fn(examples):
            texts = []
            for i in range(len(examples["input_text"])):
                inp = str(examples["input_text"][i])
                out = str(examples["output_text"][i])
                text = "<|im_start|>user\n" + inp + "<|im_end|>\n<|im_start|>assistant\n" + out + "<|im_end|>"
                texts.append(text)
            result = tokenizer(texts, truncation=True, max_length=256, padding="max_length")
            result["labels"] = [ids[:] for ids in result["input_ids"]]
            return result

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

        args = TrainingArguments(
            output_dir=str(model_dir),
            max_steps=max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16 if has_gpu else 4,
            learning_rate=5e-5,
            bf16=has_gpu,
            fp16=False,
            save_strategy="steps",
            save_steps=500,
            logging_steps=100,
            save_total_limit=1,
            remove_unused_columns=False,
            report_to="none",
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=model, args=args,
            train_dataset=tokenized,
            processing_class=tokenizer,
        )

        t0 = time.time()
        result = trainer.train()
        elapsed = time.time() - t0
        loss = result.metrics.get("train_loss", None)

        trainer.save_model(str(model_dir))
        try:
            tokenizer.save_pretrained(str(model_dir))
        except:
            pass

        logger.info(f"✅ {capsule_id}: {elapsed/60:.1f}min, loss={loss}")

        info = {
            "capsule": capsule_id, "status": "completed",
            "minutes": round(elapsed / 60, 1), "samples": count,
            "steps": max_steps, "loss": loss,
            "timestamp": datetime.now().isoformat(),
            "cycle": self.cycle,
        }
        (capsule_dir / "result.json").write_text(json.dumps(info, indent=2, default=str))

        del model, trainer, tokenized, dataset
        gc.collect()
        if has_gpu:
            import torch
            torch.cuda.empty_cache()

        self.total_trained += 1

        # دولاب البيانات — حذف البيانات الخام بعد التدريب
        if CLEANUP_AFTER_TRAIN:
            self._cleanup_data(capsule_dir, capsule_id)

        return info

    def _cleanup_data(self, capsule_dir: Path, capsule_id: str):
        """
        دولاب البيانات — حذف البيانات المتدرّب عليها
        المعرفة صارت داخل الموديل → البيانات الخام ما نحتاجها
        المساحة تتفرّغ للكشافة تجيب بيانات جديدة
        """
        data_dir = capsule_dir / "data"
        if not data_dir.exists():
            return

        deleted_size = 0
        deleted_count = 0
        for f in data_dir.glob("*.jsonl"):
            try:
                deleted_size += f.stat().st_size
                f.unlink()
                deleted_count += 1
            except:
                pass

        mb = deleted_size / (1024 * 1024)
        logger.info(f"🗑️ [{capsule_id}] Data wheel: deleted {deleted_count} files "
                    f"({mb:.1f}MB freed) — knowledge is in the model now")

    def _scout_for_capsule(self, capsule_id: str, capsule_dir: Path,
                            samples: int = 50) -> int:
        """كشافة لكبسولة واحدة فقط — FIFO"""
        from brain.knowledge_scout import InternalScout

        total = 0

        # 1. كشافة داخلية (ملفات حقيقية)
        internal = InternalScout(self.capsules_dir)
        results = internal.scan_all()
        total += results.get(capsule_id, 0)

        # 2. كشافة خارجية (Ollama)
        model = self.scout._get_model()
        if model:
            for _ in range(samples):
                if self.scout.scout_one(capsule_id, model):
                    total += 1

        return total

    def _count_data(self, capsule_dir: Path) -> int:
        """عد عينات كبسولة"""
        data_dir = capsule_dir / "data"
        if not data_dir.exists():
            return 0
        total = 0
        for f in data_dir.glob("*.jsonl"):
            if f.name == "merged_train.jsonl":
                continue
            try:
                total += sum(1 for _ in open(f))
            except:
                pass
        return total

    def run_cycle(self):
        """
        دورة FIFO — لكل كبسولة:
          📡 كشافة → 🏋️ تدريب → 🗑️ حذف → التالية
        الهارد ما يمتلي أبداً!
        """
        self.cycle += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 CYCLE #{self.cycle} — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"{'='*60}")

        # جمع كل الكبسولات النشطة
        capsules = []
        for d in sorted(CAPSULES_DIR.iterdir()):
            if not d.is_dir():
                continue
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    if meta.get("archived"):
                        continue
                except:
                    pass
            capsules.append((d.name, d))

        logger.info(f"🎯 {len(capsules)} active capsules")

        trained_this_cycle = 0
        for cid, cdir in capsules:
            # ═══ 1. كشافة لهاي الكبسولة بس ═══
            existing = self._count_data(cdir)
            new_data = self._scout_for_capsule(cid, cdir, SCOUT_SAMPLES_PER_CYCLE)
            total_data = existing + new_data

            if total_data < MIN_SAMPLES_TO_TRAIN:
                continue

            # شوف إذا يحتاج تدريب
            result_path = cdir / "result.json"
            needs_train = True
            if result_path.exists():
                try:
                    last = json.loads(result_path.read_text())
                    if total_data <= last.get("samples", 0) * 1.2:
                        needs_train = False  # ما زادت البيانات بما يكفي
                except:
                    pass

            if not needs_train:
                continue

            # ═══ 2. تدريب فوري ═══
            logger.info(f"\n🔄 FIFO [{cid}]: scout={new_data} + existing={existing} → train")
            try:
                self._train_capsule(cdir, cid)
                trained_this_cycle += 1
                # ═══ 3. الحذف صار أوتوماتيكي داخل _train_capsule ═══
            except Exception as e:
                logger.error(f"❌ {cid}: {e}")

        # ═══ تطور كل N دورات ═══
        if self.cycle % EVOLVE_EVERY_N_CYCLES == 0:
            logger.info(f"\n🧬 Evolve...")
            result = self.factory.evolve()
            self.total_evolved += len(result.get("created", []))
            logger.info(f"🧬 +{len(result.get('created', []))} children, "
                        f"-{len(result.get('archived', []))} archived")

        # ═══ ملخص ═══
        disk_info = self._disk_status()
        status = self.factory.get_status()
        logger.info(f"\n📊 Cycle #{self.cycle}: "
                    f"trained={trained_this_cycle}, "
                    f"total_capsules={status['total_capsules']}, "
                    f"L{status['max_layer']} deep")
        logger.info(f"💾 Disk: {disk_info}")

    def _disk_status(self) -> str:
        """حالة مساحة القرص"""
        try:
            stat = os.statvfs(str(CAPSULES_DIR))
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
            used_pct = ((total_gb - free_gb) / total_gb) * 100
            return f"{free_gb:.1f}GB free / {total_gb:.1f}GB ({used_pct:.0f}% used)"
        except:
            return "unknown"

    def run_forever(self):
        """المايسترو — لا يتوقف أبداً"""
        logger.info("=" * 60)
        logger.info("🎭 BRAIN DAEMON — بسم الله")
        logger.info(f"   Capsules: {CAPSULES_DIR}")
        logger.info(f"   Model: {BASE_MODEL}")
        logger.info(f"   Scout: {SCOUT_SAMPLES_PER_CYCLE} samples/capsule/cycle")
        logger.info(f"   Evolve: every {EVOLVE_EVERY_N_CYCLES} cycles")
        logger.info("=" * 60)

        while True:
            try:
                self.run_cycle()

                uptime = (datetime.now() - self.start_time).total_seconds() / 3600
                logger.info(f"\n💤 Pause {PAUSE_BETWEEN_CYCLES}s... "
                            f"(uptime: {uptime:.1f}h, "
                            f"trained: {self.total_trained}, "
                            f"evolved: {self.total_evolved})")
                time.sleep(PAUSE_BETWEEN_CYCLES)

            except KeyboardInterrupt:
                logger.info("🛑 Daemon stopped by user")
                break
            except Exception as e:
                logger.error(f"💥 Daemon error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)  # انتظر دقيقة وحاول مرة ثانية


def main():
    daemon = BrainDaemon()

    if "--once" in sys.argv:
        daemon.run_cycle()
    else:
        daemon.run_forever()


if __name__ == "__main__":
    main()
