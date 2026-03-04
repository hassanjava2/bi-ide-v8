"""
System Monitor — Real-time CPU/GPU temps, usage, training status
Serves a web dashboard at http://0.0.0.0:9090

Works on Linux (RTX 5090), Windows, and VPS.
"""
import os
import sys
import json
import time
import platform
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"


def get_gpu_info():
    """Get GPU temp, memory, utilization via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw,fan.speed",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {
                "name": parts[0],
                "temp_c": int(parts[1]) if parts[1] != "[N/A]" else 0,
                "memory_used_mb": int(parts[2]),
                "memory_total_mb": int(parts[3]),
                "utilization_pct": int(parts[4]),
                "power_w": float(parts[5]) if parts[5] != "[N/A]" else 0,
                "fan_pct": parts[6] if len(parts) > 6 else "N/A",
            }
    except Exception:
        pass
    return None


def get_cpu_info():
    """Get per-core CPU usage and temp."""
    cores = os.cpu_count() or 1
    usage_per_core = []
    avg_usage = 0
    temp_c = 0

    if IS_LINUX:
        try:
            # Per-core usage
            result = subprocess.run(
                ["mpstat", "-P", "ALL", "1", "1"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    line = line.strip()
                    if line and line[0].isdigit() and "Average" not in line:
                        parts = line.split()
                        if len(parts) > 3 and parts[2].replace(".", "").isdigit():
                            pass  # Skip header-like
                    if "Average:" in line and "all" in line:
                        parts = line.split()
                        idle = float(parts[-1])
                        avg_usage = round(100 - idle, 1)
                # Per-core from /proc/stat
                with open("/proc/stat") as f:
                    for line in f:
                        if line.startswith("cpu") and line[3] != " ":
                            parts = line.split()
                            total = sum(int(x) for x in parts[1:])
                            idle = int(parts[4])
                            usage = round((1 - idle/max(total, 1)) * 100, 1)
                            usage_per_core.append(usage)
        except Exception:
            pass

        # CPU temp
        try:
            for zone in sorted(Path("/sys/class/thermal/").glob("thermal_zone*")):
                temp_file = zone / "temp"
                if temp_file.exists():
                    t = int(temp_file.read_text().strip()) / 1000
                    if t > temp_c:
                        temp_c = t
        except Exception:
            pass

    return {
        "cores": cores,
        "usage_pct": avg_usage,
        "usage_per_core": usage_per_core[:cores],
        "temp_c": round(temp_c, 1),
    }


def get_training_status():
    """Read training log tail."""
    logs = []
    for log_path in ["/tmp/auto_training.log", "C:/Users/BI/training.log"]:
        p = Path(log_path)
        if p.exists():
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
                logs = [l for l in lines[-20:] if l.strip()]
            except Exception:
                pass
    return logs


def get_disk_info():
    """Get disk usage."""
    import shutil
    disks = {}
    for name, path in [("training", "/home/bi/training_data"), ("data", "/data"),
                        ("win_training", "C:/Users/BI/training_data")]:
        p = Path(path)
        if p.exists():
            try:
                usage = shutil.disk_usage(str(p))
                disks[name] = {
                    "total_gb": round(usage.total / 1e9, 1),
                    "used_gb": round(usage.used / 1e9, 1),
                    "free_gb": round(usage.free / 1e9, 1),
                }
            except Exception:
                pass
    return disks


# Cache for data
_cache = {"data": None, "updated": 0}

def refresh_data():
    while True:
        _cache["data"] = {
            "gpu": get_gpu_info(),
            "cpu": get_cpu_info(),
            "training_log": get_training_status(),
            "disks": get_disk_info(),
            "hostname": platform.node(),
            "os": platform.system(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _cache["updated"] = time.time()
        time.sleep(2)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BI-IDE Monitor</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0a0a1a; color:#e0e0e0; font-family:'Segoe UI',Tahoma,sans-serif; padding:15px; }
h1 { text-align:center; color:#00ff88; font-size:1.6em; margin-bottom:15px; text-shadow:0 0 20px rgba(0,255,136,0.3); }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:15px; max-width:1200px; margin:0 auto; }
.card { background:rgba(20,20,40,0.9); border:1px solid rgba(0,255,136,0.15); border-radius:12px; padding:18px; }
.card h2 { color:#00ff88; font-size:1.1em; margin-bottom:12px; display:flex; align-items:center; gap:8px; }
.metric { display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.05); }
.metric:last-child { border-bottom:none; }
.label { color:#888; font-size:0.9em; }
.value { font-size:1.2em; font-weight:bold; }
.hot { color:#ff4444; text-shadow:0 0 10px rgba(255,0,0,0.3); }
.warm { color:#ffaa00; }
.cool { color:#00ff88; }
.bar-container { width:100%; height:20px; background:#1a1a2e; border-radius:10px; overflow:hidden; margin:4px 0; }
.bar { height:100%; border-radius:10px; transition:width 0.5s ease; }
.bar.gpu { background:linear-gradient(90deg,#00ff88,#00cc66); }
.bar.cpu { background:linear-gradient(90deg,#4488ff,#2266dd); }
.bar.hot { background:linear-gradient(90deg,#ff4444,#ff0000); }
.bar.warm { background:linear-gradient(90deg,#ffaa00,#ff8800); }
.core-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:4px; margin-top:8px; }
.core { text-align:center; padding:4px; border-radius:4px; font-size:0.7em; }
.log { background:#0a0a15; border-radius:8px; padding:10px; max-height:200px; overflow-y:auto; font-family:monospace; font-size:0.75em; line-height:1.6; direction:ltr; text-align:left; }
.log .line { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.full-width { grid-column:1/-1; }
.temp-display { font-size:2em; font-weight:bold; text-align:center; padding:10px; }
#timestamp { text-align:center; color:#555; font-size:0.8em; margin-top:10px; }
</style>
</head>
<body>
<h1>🔥 BI-IDE System Monitor</h1>
<div class="grid">
  <div class="card">
    <h2>🎮 GPU</h2>
    <div id="gpu-info">Loading...</div>
  </div>
  <div class="card">
    <h2>🧠 CPU</h2>
    <div id="cpu-info">Loading...</div>
  </div>
  <div class="card">
    <h2>🌡️ Temperatures</h2>
    <div id="temp-info">Loading...</div>
  </div>
  <div class="card">
    <h2>💾 Disk</h2>
    <div id="disk-info">Loading...</div>
  </div>
  <div class="card full-width">
    <h2>📋 Training Log</h2>
    <div class="log" id="log">Loading...</div>
  </div>
</div>
<div id="timestamp"></div>
<script>
function tempClass(t) { return t > 80 ? 'hot' : t > 60 ? 'warm' : 'cool'; }
function barClass(pct) { return pct > 90 ? 'hot' : pct > 70 ? 'warm' : ''; }
function coreColor(pct) {
  if (pct > 80) return '#ff4444';
  if (pct > 50) return '#ffaa00';
  if (pct > 20) return '#4488ff';
  return '#1a2a3a';
}
async function refresh() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    // GPU
    if (d.gpu) {
      const g = d.gpu;
      document.getElementById('gpu-info').innerHTML = `
        <div class="metric"><span class="label">${g.name}</span><span class="value">${g.utilization_pct}%</span></div>
        <div class="bar-container"><div class="bar gpu ${barClass(g.utilization_pct)}" style="width:${g.utilization_pct}%"></div></div>
        <div class="metric"><span class="label">VRAM</span><span class="value">${(g.memory_used_mb/1024).toFixed(1)} / ${(g.memory_total_mb/1024).toFixed(1)} GB</span></div>
        <div class="bar-container"><div class="bar gpu" style="width:${(g.memory_used_mb/g.memory_total_mb*100).toFixed(0)}%"></div></div>
        <div class="metric"><span class="label">Power</span><span class="value">${g.power_w}W</span></div>
      `;
    } else {
      document.getElementById('gpu-info').innerHTML = '<div class="metric"><span class="label">No GPU</span></div>';
    }
    // CPU
    const c = d.cpu;
    let coreHtml = '';
    if (c.usage_per_core && c.usage_per_core.length) {
      coreHtml = '<div class="core-grid">' + c.usage_per_core.map((u,i) =>
        `<div class="core" style="background:${coreColor(u)}">${i+1}<br>${u.toFixed(0)}%</div>`
      ).join('') + '</div>';
    }
    document.getElementById('cpu-info').innerHTML = `
      <div class="metric"><span class="label">Cores</span><span class="value">${c.cores}</span></div>
      <div class="metric"><span class="label">Average</span><span class="value">${c.usage_pct}%</span></div>
      <div class="bar-container"><div class="bar cpu ${barClass(c.usage_pct)}" style="width:${c.usage_pct}%"></div></div>
      ${coreHtml}
    `;
    // Temps
    let tempHtml = '';
    if (d.gpu) tempHtml += `<div class="metric"><span class="label">🎮 GPU</span><span class="temp-display ${tempClass(d.gpu.temp_c)}">${d.gpu.temp_c}°C</span></div>`;
    tempHtml += `<div class="metric"><span class="label">🧠 CPU</span><span class="temp-display ${tempClass(c.temp_c)}">${c.temp_c}°C</span></div>`;
    document.getElementById('temp-info').innerHTML = tempHtml;
    // Disks
    let diskHtml = '';
    for (const [name, info] of Object.entries(d.disks || {})) {
      const pct = (info.used_gb / info.total_gb * 100).toFixed(0);
      diskHtml += `<div class="metric"><span class="label">${name}</span><span class="value">${info.used_gb}/${info.total_gb} GB</span></div>
        <div class="bar-container"><div class="bar gpu" style="width:${pct}%"></div></div>`;
    }
    document.getElementById('disk-info').innerHTML = diskHtml || 'No disks found';
    // Log
    if (d.training_log && d.training_log.length) {
      document.getElementById('log').innerHTML = d.training_log.map(l =>
        `<div class="line">${l.replace(/</g,'&lt;')}</div>`
      ).join('');
      const el = document.getElementById('log');
      el.scrollTop = el.scrollHeight;
    }
    document.getElementById('timestamp').textContent = `${d.hostname} | ${d.os} | ${d.timestamp}`;
  } catch(e) {
    console.error(e);
  }
}
setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>"""


class MonitorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = _cache.get("data") or {}
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Suppress logs


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9090
    
    # Start background data refresh
    t = threading.Thread(target=refresh_data, daemon=True)
    t.start()
    
    server = HTTPServer(("0.0.0.0", port), MonitorHandler)
    print(f"🖥️  Monitor running at http://0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
