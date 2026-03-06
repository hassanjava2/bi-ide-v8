#!/usr/bin/env python3
"""
brain_daemon.py — المايسترو 🎭

الحلقة الرئيسية التي لا تتوقف أبداً:
  1. الكشافة تجيب بيانات جديدة لكل الكبسولات
  2. يدرّب كل كبسولة عندها بيانات جديدة كافية
  3. المصنع يولّد كبسولات أطفال من الأقوياء
  4. يكرر من 1 ← ∞

التشغيل:
  python3 brain/brain_daemon.py
  أو:
  nohup python3 brain/brain_daemon.py > /tmp/brain_daemon.log 2>&1 &
"""

import sys
import os
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
        return info

    def _find_capsules_needing_training(self) -> list[tuple]:
        """كبسولات تحتاج تدريب (بيانات جديدة من الكشافة أو ما اتدربت)"""
        needs_training = []
        for d in sorted(CAPSULES_DIR.iterdir()):
            if not d.is_dir():
                continue
            cid = d.name

            # تخطي الأرشيف
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    if meta.get("archived"):
                        continue
                except:
                    pass

            # عد العينات
            data_dir = d / "data"
            if not data_dir.exists():
                continue
            total = 0
            for f in data_dir.glob("*.jsonl"):
                if f.name == "merged_train.jsonl":
                    continue
                try:
                    total += sum(1 for _ in open(f))
                except:
                    pass

            if total < MIN_SAMPLES_TO_TRAIN:
                continue

            # شوف آخر تدريب
            result_path = d / "result.json"
            if result_path.exists():
                try:
                    result = json.loads(result_path.read_text())
                    last_samples = result.get("samples", 0)
                    # إعادة تدريب إذا البيانات زادت 20%+
                    if total > last_samples * 1.2:
                        needs_training.append((cid, d, total))
                except:
                    needs_training.append((cid, d, total))
            else:
                needs_training.append((cid, d, total))

        return needs_training

    def run_cycle(self):
        """دورة واحدة: كشافة → تدريب → تطور"""
        self.cycle += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 CYCLE #{self.cycle} — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"{'='*60}")

        # ═══ الخطوة 1: الكشافة ═══
        logger.info(f"\n📡 Step 1: Scout — gathering data...")
        scout_results = self.scout.scout_cycle(SCOUT_SAMPLES_PER_CYCLE)
        total_new = sum(scout_results.values()) if isinstance(scout_results, dict) else 0
        logger.info(f"📡 Scout: +{total_new} new samples")

        # ═══ الخطوة 2: تدريب ═══
        logger.info(f"\n🏋️ Step 2: Train — checking capsules...")
        needs = self._find_capsules_needing_training()
        if needs:
            logger.info(f"🏋️ {len(needs)} capsules need training")
            for cid, cdir, samples in needs:
                try:
                    self._train_capsule(cdir, cid)
                except Exception as e:
                    logger.error(f"❌ Train {cid}: {e}")
        else:
            logger.info("🏋️ No capsules need training this cycle")

        # ═══ الخطوة 3: تطور ═══
        if self.cycle % EVOLVE_EVERY_N_CYCLES == 0:
            logger.info(f"\n🧬 Step 3: Evolve...")
            result = self.factory.evolve()
            self.total_evolved += len(result.get("created", []))
            logger.info(f"🧬 Created: {result.get('created', [])}")
            logger.info(f"🧬 Archived: {result.get('archived', [])}")

        # ═══ ملخص ═══
        status = self.factory.get_status()
        logger.info(f"\n📊 Status: {status['total_capsules']} capsules, "
                    f"{status['trained']} trained, "
                    f"L{status['max_layer']} deep, "
                    f"cycle #{self.cycle}")

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
