#!/usr/bin/env python3
"""
security_data_parser.py — محلل بيانات الأمن 🔒

يحول بيانات CVE وExploit-DB إلى عينات تدريب:
  1. CVE JSON → أزواج Q&A عن الثغرات
  2. Exploit-DB → أكواد استغلال + شرح
  3. CWE → أنماط الضعف الشائعة

الاستخدام:
  python3 security_data_parser.py --cve-dir /data/emergency/security/cve-list
  python3 security_data_parser.py --exploitdb /data/emergency/security/exploitdb
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CAPSULES_DIR = PROJECT_ROOT / "brain" / "capsules"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [security] %(message)s",
)
logger = logging.getLogger("security")


def parse_cve_json(cve_dir: Path, max_samples: int = 5000) -> list:
    """تحليل CVE JSON files → عينات تدريب"""
    samples = []
    cve_count = 0

    # CVElistV5 structure: cves/year/xxxxx/CVE-year-xxxxx.json
    for year_dir in sorted(cve_dir.glob("cves/20*")):
        if not year_dir.is_dir():
            continue
        for sub_dir in sorted(year_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            for cve_file in sub_dir.glob("CVE-*.json"):
                if len(samples) >= max_samples:
                    break
                try:
                    data = json.loads(cve_file.read_text(encoding="utf-8", errors="ignore"))

                    # استخراج المعلومات
                    cve_id = data.get("cveMetadata", {}).get("cveId", cve_file.stem)
                    state = data.get("cveMetadata", {}).get("state", "")

                    if state != "PUBLISHED":
                        continue

                    # الوصف
                    containers = data.get("containers", {})
                    cna = containers.get("cna", {})
                    descriptions = cna.get("descriptions", [])
                    desc = ""
                    for d in descriptions:
                        if d.get("lang", "").startswith("en"):
                            desc = d.get("value", "")
                            break
                    if not desc and descriptions:
                        desc = descriptions[0].get("value", "")

                    if not desc or len(desc) < 30:
                        continue

                    # الخطورة
                    metrics = cna.get("metrics", [])
                    severity = "UNKNOWN"
                    score = 0
                    for m in metrics:
                        cvss = m.get("cvssV3_1", m.get("cvssV3_0", {}))
                        if cvss:
                            severity = cvss.get("baseSeverity", "UNKNOWN")
                            score = cvss.get("baseScore", 0)
                            break

                    # المنتجات المتأثرة
                    affected = cna.get("affected", [])
                    products = []
                    for a in affected[:3]:
                        vendor = a.get("vendor", "")
                        product = a.get("product", "")
                        if vendor and product:
                            products.append(f"{vendor}/{product}")

                    # بناء العينة
                    sample = {
                        "input_text": f"Explain the security vulnerability {cve_id}. What is it, how severe is it, and how can it be exploited?",
                        "output_text": (
                            f"## {cve_id}\n\n"
                            f"**Severity**: {severity} (Score: {score}/10)\n"
                            f"**Affected**: {', '.join(products) if products else 'See details'}\n\n"
                            f"**Description**: {desc[:2000]}\n\n"
                            f"**Analysis**: This vulnerability is classified as {severity} severity. "
                            f"Security researchers should examine the affected components for similar patterns."
                        ),
                    }
                    samples.append(sample)
                    cve_count += 1

                except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                    continue

    logger.info(f"🔒 CVE: {cve_count} vulnerabilities parsed")
    return samples


def parse_exploitdb(exploitdb_dir: Path, max_samples: int = 3000) -> list:
    """تحليل Exploit-DB → عينات تدريب"""
    samples = []

    # Exploit-DB structure: exploits/platform/id.ext
    exploits_dir = exploitdb_dir / "exploits"
    if not exploits_dir.exists():
        logger.warning(f"⚠️ ExploitDB dir not found: {exploits_dir}")
        return samples

    # أنواع الملفات المهمة
    code_extensions = {".py", ".rb", ".c", ".cpp", ".pl", ".sh", ".php", ".java", ".txt"}

    for platform_dir in sorted(exploits_dir.iterdir()):
        if not platform_dir.is_dir():
            continue
        platform = platform_dir.name

        for exploit_file in sorted(platform_dir.iterdir()):
            if len(samples) >= max_samples:
                break
            if exploit_file.suffix.lower() not in code_extensions:
                continue

            try:
                content = exploit_file.read_text(encoding="utf-8", errors="ignore")
                if len(content) < 100 or len(content) > 8000:
                    continue

                # استخراج العنوان من أول سطرين
                lines = content.split("\n")
                title = ""
                for line in lines[:5]:
                    if line.strip() and not line.startswith("#!"):
                        title = line.strip("# ").strip("// ").strip("/* ").strip()
                        break

                sample = {
                    "input_text": (
                        f"Analyze this security exploit code for {platform}. "
                        f"Explain what vulnerability it targets, how it works, "
                        f"and how to defend against it:\n\n"
                        f"```\n{content[:3000]}\n```"
                    ),
                    "output_text": (
                        f"## Security Analysis: {title or exploit_file.name}\n\n"
                        f"**Platform**: {platform}\n"
                        f"**Type**: Exploit code\n\n"
                        f"**Code**:\n```\n{content[:3000]}\n```\n\n"
                        f"**Defense**: To defend against this type of attack, "
                        f"ensure proper input validation, use latest security patches, "
                        f"and implement defense-in-depth strategies."
                    ),
                }
                samples.append(sample)

            except (UnicodeDecodeError, OSError):
                continue

    logger.info(f"⚔️ ExploitDB: {len(samples)} exploits parsed from {exploits_dir}")
    return samples


def parse_csv_metadata(exploitdb_dir: Path) -> list:
    """تحليل files_exploits.csv — قاعدة بيانات الثغرات"""
    samples = []
    csv_file = exploitdb_dir / "files_exploits.csv"
    if not csv_file.exists():
        return samples

    try:
        import csv
        with open(csv_file, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 2000:
                    break
                desc = row.get("description", "")
                platform = row.get("platform", "")
                etype = row.get("type", "")
                if desc and len(desc) > 20:
                    samples.append({
                        "input_text": f"What is the {etype} vulnerability: '{desc}'? What platform is affected?",
                        "output_text": (
                            f"**Vulnerability**: {desc}\n"
                            f"**Platform**: {platform}\n"
                            f"**Type**: {etype}\n"
                            f"**Analysis**: This is a known security issue documented in Exploit-DB."
                        ),
                    })
    except Exception as e:
        logger.warning(f"CSV parse error: {e}")

    logger.info(f"📊 Metadata: {len(samples)} exploit records")
    return samples


def save_samples(samples: list, capsule_id: str = "security"):
    """حفظ العينات في كبسولة الأمن"""
    if not samples:
        return 0

    capsule_dir = CAPSULES_DIR / capsule_id / "data"
    capsule_dir.mkdir(parents=True, exist_ok=True)

    out_file = capsule_dir / f"security_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"💾 Saved {len(samples)} samples → {capsule_id}")
    return len(samples)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BI-IDE Security Data Parser")
    parser.add_argument("--cve-dir", type=Path, default=None, help="Path to CVE list directory")
    parser.add_argument("--exploitdb", type=Path, default=None, help="Path to ExploitDB directory")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples per source")
    args = parser.parse_args()

    logger.info("═" * 50)
    logger.info("🔒 BI-IDE Security Data Parser")
    logger.info("═" * 50)

    total = 0

    # Auto-detect paths
    emergency_paths = [
        Path("/data/emergency/security"),
        Path(os.path.expanduser("~/emergency_data/security")),
        Path("/mnt/4tb/emergency/security"),
    ]

    cve_dir = args.cve_dir
    exploitdb_dir = args.exploitdb

    if not cve_dir or not exploitdb_dir:
        for ep in emergency_paths:
            if not cve_dir and (ep / "cve-list").exists():
                cve_dir = ep / "cve-list"
            if not exploitdb_dir and (ep / "exploitdb").exists():
                exploitdb_dir = ep / "exploitdb"

    # === CVE ===
    if cve_dir and cve_dir.exists():
        logger.info(f"\n📂 CVE source: {cve_dir}")
        samples = parse_cve_json(cve_dir, args.max_samples)
        total += save_samples(samples)
    else:
        logger.info("⏭️ CVE directory not found — skip")
        logger.info("   Download: git clone https://github.com/CVEProject/cvelistV5.git")

    # === Exploit-DB ===
    if exploitdb_dir and exploitdb_dir.exists():
        logger.info(f"\n📂 ExploitDB source: {exploitdb_dir}")
        samples = parse_exploitdb(exploitdb_dir, args.max_samples)
        total += save_samples(samples)

        # CSV metadata
        samples = parse_csv_metadata(exploitdb_dir)
        total += save_samples(samples)
    else:
        logger.info("⏭️ ExploitDB directory not found — skip")
        logger.info("   Download: git clone https://gitlab.com/exploit-database/exploitdb.git")

    logger.info(f"\n{'═' * 50}")
    logger.info(f"✅ Total: {total} security training samples")
    logger.info(f"{'═' * 50}")


if __name__ == "__main__":
    main()
