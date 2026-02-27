#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bi IDE - Validate training data quality.
- Remove duplicates
- Check min/max length
- Language check
- Quality report

Usage:
    python training/validate-data.py
    python training/validate-data.py --fix
"""

import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR.parent / "data"
MIN_LENGTH = 10
MAX_LENGTH = 8000
LANG_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')


def count_arabic(text):
    return len(LANG_PATTERN.findall(text)) if text else 0


def count_english(text):
    return len(re.findall(r'[a-zA-Z]+', text or ''))


def detect_lang(text):
    a, e = count_arabic(text), count_english(text)
    if a > e:
        return 'ar'
    if e > a:
        return 'en'
    return 'mixed'


def get_sample_signature(sample):
    if isinstance(sample, dict):
        inp = sample.get('input', sample.get('prompt', sample.get('question', '')))
        out = sample.get('output', sample.get('response', sample.get('completion', sample.get('answer', ''))))
        key = f"{inp}|{out}"
    else:
        key = str(sample)
    return hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]


def load_all_training_data():
    samples = []
    for base in (OUTPUT_DIR, DATA_DIR, DATA_DIR / "learning"):
        if not base.exists():
            continue
        for path in base.rglob("*.json"):
            if path.name.startswith('.'):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        samples.append({'source': str(path), 'index': i, 'data': item})
                elif isinstance(data, dict):
                    if 'samples' in data:
                        for i, item in enumerate(data['samples']):
                            samples.append({'source': str(path), 'index': i, 'data': item})
                    elif 'examples' in data:
                        for i, item in enumerate(data['examples']):
                            samples.append({'source': str(path), 'index': i, 'data': item})
                    else:
                        samples.append({'source': str(path), 'index': 0, 'data': data})
            except Exception as e:
                print("Skip", path, e)
    return samples


def text_length(sample):
    d = sample.get('data', sample) if isinstance(sample, dict) and 'data' in sample else sample
    if not isinstance(d, dict):
        return len(str(d))
    inp = d.get('input', d.get('prompt', d.get('question', ''))) or ''
    out = d.get('output', d.get('response', d.get('completion', d.get('answer', '')))) or ''
    return len(inp) + len(out)


def validate(samples, fix=False):
    report = {'total': len(samples), 'duplicates_removed': 0, 'too_short': 0, 'too_long': 0,
              'by_language': defaultdict(int), 'valid': 0, 'invalid': []}
    seen = set()
    valid_list = []
    for s in samples:
        d = s['data'] if isinstance(s, dict) and 'data' in s else s
        sig = get_sample_signature(d)
        if sig in seen:
            report['duplicates_removed'] += 1
            continue
        seen.add(sig)
        length = text_length(s)
        if length < MIN_LENGTH:
            report['too_short'] += 1
            report['invalid'].append({'reason': 'too_short', 'length': length, 'source': s.get('source', '')})
            continue
        if length > MAX_LENGTH:
            report['too_long'] += 1
            report['invalid'].append({'reason': 'too_long', 'length': length, 'source': s.get('source', '')})
            continue
        inp = (d.get('input') or d.get('prompt') or d.get('question') or '') if isinstance(d, dict) else ''
        report['by_language'][detect_lang(inp)] += 1
        report['valid'] += 1
        valid_list.append(d)
    report['by_language'] = dict(report['by_language'])
    return report, valid_list


def main():
    import sys
    fix = '--fix' in sys.argv
    print("Bi IDE - Validate Training Data")
    print("=" * 50)
    samples = load_all_training_data()
    print("Loaded", len(samples), "raw samples")
    report, valid_list = validate(samples, fix=fix)
    print("Total:", report['total'])
    print("Duplicates removed:", report['duplicates_removed'])
    print("Too short:", report['too_short'])
    print("Too long:", report['too_long'])
    print("Valid:", report['valid'])
    print("By language:", report['by_language'])
    if fix and valid_list:
        out_path = OUTPUT_DIR / "validated_training_data.json"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(valid_list, f, ensure_ascii=False, indent=2)
        print("Saved", len(valid_list), "to", out_path)
    report_path = OUTPUT_DIR / "validation_report.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({**report, 'invalid': report['invalid'][:100]}, f, ensure_ascii=False, indent=2)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
