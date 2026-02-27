"""
Bi IDE - Artifact Packager
تجهيز مخرجات التدريب (finetuned / bi-ai-onnx / training output) كأرشيفات مقسمة
حتى نقدر نسحبها بدون ما نوقف التدريب.

الهدف الأساسي:
- إنشاء ملف tar
- تقسيمه إلى أجزاء بحجم ثابت (مثل 500MB)
- عدم تكرار التغليف إذا ماكو تغيير (state file)
"""

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def compute_fingerprint(paths: list[Path]) -> str:
    # بصمة خفيفة مبنية على (اسم + حجم + mtime) لكل الملفات
    items = []
    for root in paths:
        if not root.exists():
            continue
        if root.is_file():
            st = root.stat()
            items.append(f"F|{root.as_posix()}|{st.st_size}|{int(st.st_mtime)}")
        else:
            for p in root.rglob('*'):
                if p.is_file():
                    st = p.stat()
                    items.append(f"F|{p.as_posix()}|{st.st_size}|{int(st.st_mtime)}")
    items.sort()
    h = hashlib.sha256('\n'.join(items).encode('utf-8'))
    return h.hexdigest()


def load_state(state_path: Path) -> dict:
    if state_path and state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def split_file(src: Path, part_bytes: int, out_prefix: Path) -> list[Path]:
    parts = []
    idx = 0
    with src.open('rb') as f:
        while True:
            chunk = f.read(part_bytes)
            if not chunk:
                break
            part = Path(f"{out_prefix}.part-{idx:03d}")
            part.write_bytes(chunk)
            parts.append(part)
            idx += 1
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True, help='مجلد الإخراج')
    ap.add_argument('--include', action='append', default=[], help='مسار/مجلد لإدراجه (يمكن تكرارها)')
    ap.add_argument('--part-mb', type=int, default=500, help='حجم الجزء بالميغابايت')
    ap.add_argument('--max-output-gb', type=float, default=50.0, help='حد أقصى لحجم الإخراج (لمنع انفجار الحجم)')
    ap.add_argument('--state', default='', help='مسار ملف state لتجنب إعادة التغليف')
    args = ap.parse_args()

    base = Path('.').resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    include_paths = [ (base / p).resolve() for p in args.include ]
    state_path = Path(args.state).resolve() if args.state else None
    state = load_state(state_path) if state_path else {}

    fingerprint = compute_fingerprint(include_paths)
    if state.get('fingerprint') == fingerprint:
        print('No changes detected; skip packaging.')
        return

    tar_path = out_dir / 'artifacts.tar'
    print('Creating tar:', tar_path)

    with tarfile.open(tar_path, 'w') as tar:
        for p in include_paths:
            if not p.exists():
                continue
            arcname = p.relative_to(base).as_posix()
            tar.add(p, arcname=arcname)

    total_bytes = tar_path.stat().st_size
    if total_bytes > args.max_output_gb * (1024 ** 3):
        raise SystemExit(f"Artifacts tar too large: {total_bytes} bytes (limit {args.max_output_gb} GB)")

    part_bytes = args.part_mb * 1024 * 1024
    parts = split_file(tar_path, part_bytes, out_dir / 'artifacts.tar')

    manifest = {
        'fingerprint': fingerprint,
        'tar': tar_path.name,
        'tar_sha256': sha256_file(tar_path),
        'parts': [p.name for p in parts],
        'part_mb': args.part_mb,
        'total_bytes': total_bytes,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    # نقدر نحذف tar الأصلي لتوفير مساحة بعد التقسيم
    try:
        tar_path.unlink()
    except Exception:
        pass

    if state_path:
        state.update({
            'fingerprint': fingerprint,
            'last_output': str(out_dir),
            'last_total_bytes': total_bytes,
        })
        save_state(state_path, state)

    print('Done. Parts:', len(parts))


if __name__ == '__main__':
    main()
