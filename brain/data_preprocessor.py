#!/usr/bin/env python3
"""
data_preprocessor.py — تحويل كل الداتاسيتات لصيغة تدريب موحّدة
Converts all downloaded datasets into unified JSONL training format

الصيغة الموحّدة:
{
    "instruction": "...",
    "input": "...",       # اختياري
    "output": "...",
    "capsule": "...",     # أي كبسولة ينتمي
    "source": "...",      # مصدر البيانات
    "lang": "ar|en",
    "difficulty": 1-13    # مستوى المنهج
}
"""
import json
import os
import glob
import hashlib
from pathlib import Path
from typing import Generator, Dict, Any, Optional


# ═══════════════════════════════════════════════
# Capsule keyword matcher
# ═══════════════════════════════════════════════
CAPSULE_KEYWORDS = {}

def load_capsule_keywords():
    """Load keywords from capsule_500.py"""
    global CAPSULE_KEYWORDS
    try:
        from brain.capsule_500 import CAPSULE_REGISTRY
        CAPSULE_KEYWORDS = {
            cid: keywords
            for cid, (_, keywords) in CAPSULE_REGISTRY.items()
        }
    except ImportError:
        pass

def match_capsule(text: str) -> str:
    """Match text to most relevant capsule"""
    if not CAPSULE_KEYWORDS:
        load_capsule_keywords()

    text_lower = text.lower()
    best_capsule = "general"
    best_score = 0

    for cid, keywords in CAPSULE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > best_score:
            best_score = score
            best_capsule = cid

    return best_capsule


# ═══════════════════════════════════════════════
# Curriculum difficulty assignment
# ═══════════════════════════════════════════════
DIFFICULTY_MAP = {
    # Tier 1-3: Basics
    "dolly": 1, "alpaca": 1, "no_robots": 1,
    "oasst2": 2, "capybara": 2,
    # Tier 4-6: Knowledge
    "sciq": 3, "mmlu": 4, "metamath": 5,
    "wikipedia": 4, "medqa": 5,
    # Tier 7-9: Skills
    "slimorca": 6, "ultrachat": 6,
    "code_alpaca": 7, "evol_code": 8, "magicoder": 9,
    # Tier 10-11: Vision
    "llava_instruct": 10, "sharegpt4v": 10, "safety_vision": 10,
    # Tier 12-13: Advanced
    "pile": 11, "arxiv": 12, "starcoderdata": 13,
}


# ═══════════════════════════════════════════════
# Dataset parsers
# ═══════════════════════════════════════════════

def parse_alpaca(path: str) -> Generator[Dict, None, None]:
    """Parse Alpaca/Dolly format: instruction + input + output"""
    for f in glob.glob(os.path.join(path, "*.json")) + glob.glob(os.path.join(path, "*.jsonl")):
        with open(f, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
                if isinstance(data, list):
                    for item in data:
                        yield {
                            "instruction": item.get("instruction", ""),
                            "input": item.get("input", item.get("context", "")),
                            "output": item.get("output", item.get("response", "")),
                        }
            except json.JSONDecodeError:
                fh.seek(0)
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            yield {
                                "instruction": item.get("instruction", ""),
                                "input": item.get("input", item.get("context", "")),
                                "output": item.get("output", item.get("response", "")),
                            }
                        except json.JSONDecodeError:
                            continue


def parse_oasst(path: str) -> Generator[Dict, None, None]:
    """Parse OpenAssistant conversations"""
    for f in glob.glob(os.path.join(path, "*.parquet")):
        try:
            import pandas as pd
            df = pd.read_parquet(f)
            for _, row in df.iterrows():
                messages = row.get("messages", row.get("conversation", []))
                if isinstance(messages, list) and len(messages) >= 2:
                    yield {
                        "instruction": messages[0].get("content", "") if isinstance(messages[0], dict) else str(messages[0]),
                        "input": "",
                        "output": messages[1].get("content", "") if isinstance(messages[1], dict) else str(messages[1]),
                    }
        except ImportError:
            # Fall back to JSON
            pass

    for f in glob.glob(os.path.join(path, "*.json*")):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    item = json.loads(line.strip())
                    if "messages" in item:
                        msgs = item["messages"]
                        if len(msgs) >= 2:
                            yield {
                                "instruction": msgs[0].get("content", ""),
                                "input": "",
                                "output": msgs[1].get("content", ""),
                            }
                    elif "text" in item:
                        yield {
                            "instruction": item.get("text", "")[:200],
                            "input": "",
                            "output": item.get("text", ""),
                        }
                except json.JSONDecodeError:
                    continue


def parse_wikipedia(path: str) -> Generator[Dict, None, None]:
    """Parse Wikipedia dumps — extract articles as knowledge"""
    for f in glob.glob(os.path.join(path, "**/*.parquet"), recursive=True):
        try:
            import pandas as pd
            df = pd.read_parquet(f)
            for _, row in df.iterrows():
                title = row.get("title", "")
                text = row.get("text", "")
                if len(text) > 100:
                    # Create Q&A from article
                    yield {
                        "instruction": f"اشرح لي عن: {title}" if any(
                            ord(c) > 0x600 for c in title
                        ) else f"Explain: {title}",
                        "input": "",
                        "output": text[:4000],
                    }
        except ImportError:
            pass


def parse_code(path: str) -> Generator[Dict, None, None]:
    """Parse code instruction datasets"""
    for f in glob.glob(os.path.join(path, "*.json*")):
        with open(f, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
                items = data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                fh.seek(0)
                items = []
                for line in fh:
                    try:
                        items.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue

            for item in items:
                yield {
                    "instruction": item.get("instruction", item.get("prompt", "")),
                    "input": item.get("input", ""),
                    "output": item.get("output", item.get("response", item.get("completion", ""))),
                }


def parse_vision(path: str) -> Generator[Dict, None, None]:
    """Parse vision-language datasets (LLaVA format)"""
    for f in glob.glob(os.path.join(path, "*.json*")):
        with open(f, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
                items = data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                fh.seek(0)
                items = []
                for line in fh:
                    try:
                        items.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue

            for item in items:
                convos = item.get("conversations", [])
                if len(convos) >= 2:
                    yield {
                        "instruction": convos[0].get("value", ""),
                        "input": item.get("image", ""),
                        "output": convos[1].get("value", ""),
                    }


def parse_generic(path: str) -> Generator[Dict, None, None]:
    """Generic parser — tries all formats"""
    for f in sorted(glob.glob(os.path.join(path, "**/*"), recursive=True)):
        if not os.path.isfile(f):
            continue
        ext = os.path.splitext(f)[1].lower()

        if ext == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(f)
                for _, row in df.iterrows():
                    d = row.to_dict()
                    yield {
                        "instruction": str(d.get("instruction", d.get("question", d.get("prompt", "")))),
                        "input": str(d.get("input", d.get("context", ""))),
                        "output": str(d.get("output", d.get("answer", d.get("response", d.get("text", ""))))),
                    }
            except (ImportError, Exception):
                continue

        elif ext in (".json", ".jsonl"):
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                try:
                    data = json.load(fh)
                    items = data if isinstance(data, list) else [data]
                except json.JSONDecodeError:
                    fh.seek(0)
                    items = []
                    for line in fh:
                        try:
                            items.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue

                for item in items:
                    if isinstance(item, dict):
                        yield {
                            "instruction": str(item.get("instruction", item.get("question", item.get("prompt", "")))),
                            "input": str(item.get("input", item.get("context", ""))),
                            "output": str(item.get("output", item.get("answer", item.get("response", item.get("text", ""))))),
                        }


# ═══════════════════════════════════════════════
# Dataset → Parser mapping
# ═══════════════════════════════════════════════
PARSERS = {
    "alpaca": parse_alpaca,
    "dolly": parse_alpaca,
    "code_alpaca": parse_code,
    "evol_code": parse_code,
    "magicoder": parse_code,
    "oasst2": parse_oasst,
    "slimorca": parse_oasst,
    "ultrachat": parse_oasst,
    "capybara": parse_oasst,
    "no_robots": parse_oasst,
    "llava_instruct": parse_vision,
    "sharegpt4v": parse_vision,
    "wikipedia": parse_wikipedia,
}


# ═══════════════════════════════════════════════
# Main preprocessor
# ═══════════════════════════════════════════════

class DataPreprocessor:
    """تحويل كل الداتاسيتات لصيغة تدريب موحّدة"""

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seen_hashes = set()
        self.stats = {}

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def process_dataset(self, name: str, path: str) -> int:
        """Process one dataset and write unified JSONL"""
        parser = PARSERS.get(name, parse_generic)
        difficulty = DIFFICULTY_MAP.get(name, 5)
        output_file = self.output_dir / f"{name}.jsonl"
        count = 0

        with open(output_file, "w", encoding="utf-8") as out:
            for item in parser(path):
                # Skip empty
                if not item.get("output") and not item.get("instruction"):
                    continue

                # Deduplicate
                h = self._hash(item.get("instruction", "") + item.get("output", ""))
                if h in self.seen_hashes:
                    continue
                self.seen_hashes.add(h)

                # Match capsule
                text = f"{item.get('instruction', '')} {item.get('output', '')}"
                capsule = match_capsule(text)

                # Detect language
                lang = "ar" if any(ord(c) > 0x600 and ord(c) < 0x700 for c in text[:200]) else "en"

                # Write unified format
                unified = {
                    "instruction": item["instruction"],
                    "input": item.get("input", ""),
                    "output": item["output"],
                    "capsule": capsule,
                    "source": name,
                    "lang": lang,
                    "difficulty": difficulty,
                }
                out.write(json.dumps(unified, ensure_ascii=False) + "\n")
                count += 1

        self.stats[name] = count
        return count

    def process_all(self):
        """Process all datasets in data_dir"""
        print(f"🔄 Processing datasets from: {self.data_dir}")
        total = 0

        for entry in sorted(self.data_dir.iterdir()):
            if entry.is_dir() and entry.name != "models":
                name = entry.name
                print(f"\n📚 Processing: {name}...")
                count = self.process_dataset(name, str(entry))
                total += count
                print(f"   → {count:,} samples")

        # Merge all into one file ordered by difficulty (curriculum)
        merged = self.output_dir / "all_training_data.jsonl"
        all_samples = []

        for f in sorted(self.output_dir.glob("*.jsonl")):
            if f.name == "all_training_data.jsonl":
                continue
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        all_samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Sort by difficulty (curriculum learning)
        all_samples.sort(key=lambda x: x.get("difficulty", 5))

        with open(merged, "w", encoding="utf-8") as out:
            for s in all_samples:
                out.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(f"\n{'='*50}")
        print(f"✅ Total: {total:,} training samples")
        print(f"📁 Merged: {merged}")
        print(f"📊 By dataset:")
        for name, count in sorted(self.stats.items(), key=lambda x: -x[1]):
            print(f"   {name:25s} {count:>8,}")
        print(f"\n📊 By difficulty (curriculum):")
        diff_counts = {}
        for s in all_samples:
            d = s.get("difficulty", 0)
            diff_counts[d] = diff_counts.get(d, 0) + 1
        for d in sorted(diff_counts):
            print(f"   Tier {d:2d}: {diff_counts[d]:>8,}")

        return total


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/data_staging"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/root/training_data"

    processor = DataPreprocessor(data_dir, output_dir)
    processor.process_all()
