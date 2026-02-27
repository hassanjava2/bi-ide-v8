#!/usr/bin/env bash
set -euo pipefail

# Bi IDE - Hostinger Pulse Training Runner
# تشغيل التدريب بنبضات لتفادي خفض أداء المعالج على Hostinger.
# الفكرة: 150 دقيقة “نشاط منخفض/مقيّد” + 45 دقيقة تهدئة، مع منع التكرار بعد اكتمال التدريب.

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

# Python interpreter (prefer project venv)
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$BASE_DIR/venv/bin/python3" ]]; then
    PYTHON="$BASE_DIR/venv/bin/python3"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  else
    echo "❌ Python غير موجود (لا python3 ولا venv)." >&2
    exit 1
  fi
fi

# === إعدادات النبضات ===
PULSE_TRAIN_MINUTES="${PULSE_TRAIN_MINUTES:-150}"
PULSE_REST_MINUTES="${PULSE_REST_MINUTES:-45}"

# === تقييد الضغط (threads + nice) ===
CPU_THREADS="${CPU_THREADS:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NICE_LEVEL="${NICE_LEVEL:-10}"

# === تقسيم ملفات النتائج إلى أجزاء (للسحب بدون توقف) ===
ARTIFACT_PART_MB="${ARTIFACT_PART_MB:-500}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$BASE_DIR/training/artifacts}"

# === ملفات حالة ===
STATE_FILE="${STATE_FILE:-$BASE_DIR/training/output/hostinger-pulse-state.json}"
LOCK_FILE="${LOCK_FILE:-$BASE_DIR/training/output/hostinger-pulse.lock}"

mkdir -p "$(dirname "$STATE_FILE")" "$ARTIFACTS_DIR"

log() {
  echo "[$(date '+%F %T')] $*"
}

acquire_lock() {
  if (set -o noclobber; echo "$$" > "$LOCK_FILE") 2>/dev/null; then
    trap 'rm -f "$LOCK_FILE"' EXIT
    return 0
  fi
  log "⚠️ يوجد تشغيل آخر (lock موجود): $LOCK_FILE"
  exit 0
}

read_progress_json() {
  # يطبع JSON بسيط عن التدريب إذا وجد trainer_state.json
  "$PYTHON" - <<'PY'
import json
from pathlib import Path

base = Path('.')
root = base / 'models' / 'finetuned'

def newest_trainer_state():
    if not root.exists():
        return None
    states = list(root.glob('checkpoint-*/trainer_state.json'))
    if not states:
        return None
    states.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return states[0]

p = newest_trainer_state()
if not p:
    print(json.dumps({"found": False}))
    raise SystemExit(0)

try:
    data = json.loads(p.read_text(encoding='utf-8'))
except Exception:
    print(json.dumps({"found": True, "path": str(p), "parse": False}))
    raise SystemExit(0)

out = {
  "found": True,
  "path": str(p),
  "global_step": data.get('global_step'),
  "max_steps": data.get('max_steps'),
  "epoch": data.get('epoch'),
  "best_global_step": data.get('best_global_step'),
  "best_metric": data.get('best_metric'),
}
print(json.dumps(out, ensure_ascii=False))
PY
}

should_stop_forever() {
  # يوقف نهائياً إذا التدريب مكتمل حسب global_step/max_steps
  local progress
  progress="$(read_progress_json)"

  "$PYTHON" - <<PY
import json
data = json.loads('''$progress''')
gs = data.get('global_step')
ms = data.get('max_steps')
if isinstance(gs, int) and isinstance(ms, int) and ms > 0 and gs >= ms:
    print('YES')
else:
    print('NO')
PY
}

package_artifacts() {
  # يعبّي artifacts إلى أجزاء 500MB لتسهيل السحب بدون توقف.
  # ما يغيّر أي شيء بالموديلات؛ فقط نسخ/أرشفة.
  local stamp out_dir
  stamp="$(date '+%F_%H%M%S')"
  out_dir="$ARTIFACTS_DIR/$stamp"
  mkdir -p "$out_dir"

  log "📦 Packaging artifacts (parts=${ARTIFACT_PART_MB}MB) → $out_dir"
  "$PYTHON" "training/package-artifacts.py" \
    --output "$out_dir" \
    --part-mb "$ARTIFACT_PART_MB" \
    --include "models/finetuned" \
    --include "models/bi-ai-onnx" \
    --include "models/model-registry.json" \
    --include "training/output" \
    --max-output-gb "50" \
    --state "$STATE_FILE" \
    || true
}

run_one_pulse() {
  log "🚀 Pulse start: train=${PULSE_TRAIN_MINUTES}min, rest=${PULSE_REST_MINUTES}min"

  export CPU_THREADS NUM_WORKERS
  export OMP_NUM_THREADS="$CPU_THREADS"
  export MKL_NUM_THREADS="$CPU_THREADS"
  export OPENBLAS_NUM_THREADS="$CPU_THREADS"
  export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
  export NUMEXPR_NUM_THREADS="$CPU_THREADS"

  # إيقاف تدريجي داخل finetune.py بعد مدة النبضة (يحفظ checkpoint)
  export PULSE_MAX_MINUTES="$PULSE_TRAIN_MINUTES"

  # تشغيل دورة واحدة من pipeline (جمع + تحقق + finetune + onnx)
  # نخليها بnice حتى يقل الضغط.
  nice -n "$NICE_LEVEL" "$PYTHON" training/continuous-train.py || true

  package_artifacts

  log "🧊 Rest/Throttle: ${PULSE_REST_MINUTES}min"
  sleep "$((PULSE_REST_MINUTES * 60))"
}

main() {
  acquire_lock
  log "✅ Hostinger pulse runner started (PID=$$)"

  while true; do
    if [[ "$(should_stop_forever)" == "YES" ]]; then
      log "✅ Training complete (global_step >= max_steps). Stopping service."
      package_artifacts
      exit 0
    fi
    run_one_pulse
  done
}

main "$@"
