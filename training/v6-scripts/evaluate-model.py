#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bi IDE - تقييم النموذج
- قياس perplexity على مجموعة validation
- مقارنة مع النموذج السابق (إن وُجد)
- تقرير أداء

الاستخدام:
    python training/evaluate-model.py
    python training/evaluate-model.py --model path/to/model
"""

import json
import sys
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR.parent / "models"
VALIDATION_FILE = OUTPUT_DIR / "validated_training_data.json"
REPORT_FILE = OUTPUT_DIR / "evaluation_report.json"


def load_validation_samples(limit=200):
    if not VALIDATION_FILE.exists():
        data_path = OUTPUT_DIR / "all_training_data.json"
        if not data_path.exists():
            return []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        samples = data if isinstance(data, list) else data.get('samples', data.get('examples', []))
    else:
        with open(VALIDATION_FILE, 'r', encoding='utf-8') as f:
            samples = json.load(f)
    return (samples or [])[:limit]


def get_text_from_sample(s):
    if isinstance(s, dict):
        inp = s.get('input', s.get('prompt', s.get('question', ''))) or ''
        out = s.get('output', s.get('response', s.get('completion', s.get('answer', '')))) or ''
        return inp + '\n' + out
    return str(s)


def compute_perplexity_approx(model_path, samples):
    """تقدير تقريبي لـ perplexity (يتطلب تحميل النموذج)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return None, "transformers/torch not installed"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.eval()
        total_loss = 0.0
        count = 0
        for s in samples[:50]:
            text = get_text_from_sample(s)
            if len(text) < 20:
                continue
            inputs = tokenizer(text[:512], return_tensors='pt', truncation=True)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item()
                count += 1
        if count == 0:
            return None, "No valid samples"
        avg_loss = total_loss / count
        ppl = float(torch.exp(torch.tensor(avg_loss)).item())
        return ppl, None
    except Exception as e:
        return None, str(e)


def main():
    model_path = MODELS_DIR / "merged"
    for i, arg in enumerate(sys.argv):
        if arg == '--model' and i + 1 < len(sys.argv):
            model_path = Path(sys.argv[i + 1])
            break
    if not model_path.exists():
        model_path = MODELS_DIR / "finetuned-extended"
    if not model_path.exists():
        model_path = MODELS_DIR / "finetuned"

    samples = load_validation_samples()
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model_path),
        'validation_samples': len(samples),
        'perplexity': None,
        'error': None,
        'previous_report': None
    }

    prev_report_path = OUTPUT_DIR / "evaluation_report_previous.json"
    if prev_report_path.exists():
        try:
            with open(prev_report_path, 'r', encoding='utf-8') as f:
                report['previous_report'] = json.load(f)
        except Exception:
            pass

    if not samples:
        report['error'] = 'No validation samples (run validate-data.py first or ensure output/all_training_data.json exists)'
        print("⚠️", report['error'])
    elif not model_path.exists():
        report['error'] = f'Model path not found: {model_path}'
        print("⚠️", report['error'])
    else:
        print("Evaluating model (perplexity on validation set)...")
        ppl, err = compute_perplexity_approx(str(model_path), samples)
        report['perplexity'] = ppl
        report['error'] = err
        if ppl is not None:
            print(f"  Perplexity: {ppl:.2f}")
            if report.get('previous_report', {}).get('perplexity') is not None:
                prev_ppl = report['previous_report']['perplexity']
                diff = ppl - prev_ppl
                print(f"  Previous: {prev_ppl:.2f} (diff: {diff:+.2f})")
        elif err:
            print("  Error:", err)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if REPORT_FILE.exists():
        import shutil
        shutil.copy(REPORT_FILE, prev_report_path)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
