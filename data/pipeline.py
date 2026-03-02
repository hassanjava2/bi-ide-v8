"""
Data Pipeline Module — DataCleaner & DataValidator
Used by auto_learning_system.py for training data preprocessing.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class DataCleaner:
    """Cleans and preprocesses raw training data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_length = self.config.get("min_length", 10)
        self.max_length = self.config.get("max_length", 100000)
        self.stats = {"total": 0, "cleaned": 0, "rejected": 0}

    def clean_text(self, text: str) -> str:
        """Clean a single text sample."""
        if not text or not isinstance(text, str):
            return ""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text

    def clean_batch(self, samples: List[str]) -> List[str]:
        """Clean a batch of text samples."""
        cleaned = []
        for sample in samples:
            self.stats["total"] += 1
            text = self.clean_text(sample)
            if self.min_length <= len(text) <= self.max_length:
                cleaned.append(text)
                self.stats["cleaned"] += 1
            else:
                self.stats["rejected"] += 1
        return cleaned

    def clean_dataset(self, data: List[Dict[str, Any]], text_field: str = "text") -> List[Dict[str, Any]]:
        """Clean a dataset of dictionaries."""
        cleaned = []
        for item in data:
            self.stats["total"] += 1
            if text_field in item:
                text = self.clean_text(item[text_field])
                if self.min_length <= len(text) <= self.max_length:
                    item[text_field] = text
                    cleaned.append(item)
                    self.stats["cleaned"] += 1
                else:
                    self.stats["rejected"] += 1
        return cleaned

    def get_stats(self) -> Dict[str, int]:
        return self.stats.copy()


class DataValidator:
    """Validates training data quality and integrity."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.checks_passed = 0
        self.checks_failed = 0
        self.errors: List[str] = []

    def validate_sample(self, sample: str) -> bool:
        """Validate a single text sample."""
        if not sample or not isinstance(sample, str):
            self.checks_failed += 1
            self.errors.append("Empty or non-string sample")
            return False
        if len(sample.strip()) < 5:
            self.checks_failed += 1
            self.errors.append(f"Sample too short: {len(sample)} chars")
            return False
        self.checks_passed += 1
        return True

    def validate_batch(self, samples: List[str]) -> Dict[str, Any]:
        """Validate a batch of samples and return report."""
        valid = []
        invalid = []
        for s in samples:
            if self.validate_sample(s):
                valid.append(s)
            else:
                invalid.append(s)
        return {
            "total": len(samples),
            "valid": len(valid),
            "invalid": len(invalid),
            "valid_ratio": len(valid) / max(len(samples), 1),
            "valid_samples": valid,
        }

    def validate_dataset(self, data: List[Dict[str, Any]], required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate dataset structure and content."""
        required_fields = required_fields or ["text"]
        missing_fields = []
        valid_count = 0

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                self.checks_failed += 1
                continue
            missing = [f for f in required_fields if f not in item]
            if missing:
                missing_fields.append({"index": i, "missing": missing})
                self.checks_failed += 1
            else:
                valid_count += 1
                self.checks_passed += 1

        return {
            "total": len(data),
            "valid": valid_count,
            "missing_fields": missing_fields[:10],  # first 10
            "passed": len(missing_fields) == 0,
        }

    def validate_file(self, filepath: str) -> Dict[str, Any]:
        """Validate a data file (JSON/JSONL)."""
        path = Path(filepath)
        if not path.exists():
            return {"error": f"File not found: {filepath}", "passed": False}

        try:
            if path.suffix == ".jsonl":
                data = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data = [data]

            return {
                "file": str(path),
                "records": len(data),
                "validation": self.validate_dataset(data),
                "passed": True,
            }
        except Exception as e:
            return {"error": str(e), "passed": False}

    def get_report(self) -> Dict[str, Any]:
        return {
            "passed": self.checks_passed,
            "failed": self.checks_failed,
            "errors": self.errors[:20],
            "status": "healthy" if self.checks_failed == 0 else "issues_found",
        }
