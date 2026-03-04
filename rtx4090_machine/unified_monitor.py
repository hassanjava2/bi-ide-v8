"""
Unified Multi-Machine Monitor — Aggregates all BI-IDE machines into one dashboard.
Runs locally on Mac, fetches data from RTX 5090, Windows, and VPS monitors.
Serves at http://localhost:9091

Usage: python3 rtx4090_machine/unified_monitor.py
"""
import json
import time
import threading
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

MACHINES = {
    "RTX 5090": {"url": "http://192.168.1.164:9090/api/status", "icon": "🔥", "color": "#00ff88"},
    "Windows": {"url": "http://192.168.1.130:9090/api/status", "icon": "🖥️", "color": "#4488ff"},
    "VPS Hostinger": {"url": "http://bi-iq.com:9090/api/status", "icon": "☁️", "color": "#ff88ff"},
}

_data = {}

def fetch_machine(name, info):
    while True:
        try:
            req = urllib.request.Request(info["url"], headers={"User-Agent": "BI-Monitor"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                _data[name] = json.loads(resp.read())
                _data[name]["online"] = True
        except Exception:
            _data[name] = {"online": False, "hostname": name}
        time.sleep(3)


DASHBOARD = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BI-IDE — All Machines Monitor</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#08081a;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;padding:10px}
h1{text-align:center;font-size:1.3em;color:#00ff88;margin-bottom:12px;text-shadow:0 0 15px rgba(0,255,136,0.3)}
.machines{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;max-width:1400px;margin:0 auto}
@media(max-width:900px){.machines{grid-template-columns:1fr}}
.machine{background:rgba(15,15,30,0.95);border-radius:10px;padding:14px;border:1px solid rgba(255,255,255,0.08)}
.machine.offline{opacity:0.4}
.machine-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.08)}
.machine-name{font-size:1.1em;font-weight:bold}
.status-dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.status-dot.on{background:#00ff88;box-shadow:0 0 8px #00ff88}
.status-dot.off{background:#ff4444;box-shadow:0 0 8px #ff4444}
.row{display:flex;gap:8px;margin-bottom:8px}
.stat{flex:1;background:rgba(0,0,0,0.3);border-radius:8px;padding:10px;text-align:center}
.stat-label{font-size:0.7em;color:#888;margin-bottom:4px}
.stat-value{font-size:1.5em;font-weight:bold}
.hot{color:#ff4444} .warm{color:#ffaa00} .cool{color:#00ff88} .blue{color:#4488ff}
.bar{height:6px;background:#1a1a2e;border-radius:3px;margin-top:4px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;transition:width 0.5s}
.cores{display:grid;grid-template-columns:repeat(auto-fill,minmax(28px,1fr));gap:2px;margin-top:6px}
.core{text-align:center;padding:2px;border-radius:3px;font-size:0.6em;color:#fff}
.log-box{background:#0a0a15;border-radius:6px;padding:6px;max-height:120px;overflow-y:auto;font-family:monospace;font-size:0.65em;line-height:1.5;direction:ltr;text-align:left;margin-top:8px}
.log-line{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#888}
#ts{text-align:center;color:#444;font-size:0.7em;margin-top:8px}
</style>
</head>
<body>
<h1>🔥 BI-IDE — All Machines Monitor</h1>
<div class="machines" id="machines"></div>
<div id="ts"></div>
<script>
const machines = """ + json.dumps(MACHINES) + """;
function tc(t){return t>80?'hot':t>60?'warm':'cool'}
function cc(p){return p>80?'#ff4444':p>50?'#ffaa00':p>20?'#4488ff':'#1a2a3a'}
function bg(p){return p>90?'#ff4444':p>70?'#ffaa00':'#00ff88'}
async function refresh(){
  try{
    const r=await fetch('/api/all');
    const all=await r.json();
    let html='';
    for(const[name,cfg] of Object.entries(machines)){
      const d=all[name]||{online:false};
      const on=d.online;
      const gpu=d.gpu;
      const cpu=d.cpu||{cores:0,usage_pct:0,temp_c:0,usage_per_core:[]};
      html+=`<div class="machine ${on?'':'offline'}">
        <div class="machine-header">
          <span class="machine-name">${cfg.icon} ${name}</span>
          <span class="status-dot ${on?'on':'off'}"></span>
        </div>`;
      if(!on){html+=`<div style="text-align:center;color:#ff4444;padding:20px">OFFLINE</div></div>`;continue}
      // Stats row
      html+=`<div class="row">
        <div class="stat"><div class="stat-label">CPU</div><div class="stat-value ${tc(cpu.usage_pct>80?81:0)}">${cpu.usage_pct}%</div>
          <div class="bar"><div class="bar-fill" style="width:${cpu.usage_pct}%;background:${bg(cpu.usage_pct)}"></div></div></div>`;
      if(gpu){
        html+=`<div class="stat"><div class="stat-label">GPU</div><div class="stat-value cool">${gpu.utilization_pct}%</div>
          <div class="bar"><div class="bar-fill" style="width:${gpu.utilization_pct}%;background:${bg(gpu.utilization_pct)}"></div></div></div>`;
      }
      html+=`</div><div class="row">
        <div class="stat"><div class="stat-label">🌡 CPU</div><div class="stat-value ${tc(cpu.temp_c)}">${cpu.temp_c}°</div></div>`;
      if(gpu){
        html+=`<div class="stat"><div class="stat-label">🌡 GPU</div><div class="stat-value ${tc(gpu.temp_c)}">${gpu.temp_c}°</div></div>
          <div class="stat"><div class="stat-label">VRAM</div><div class="stat-value blue">${(gpu.memory_used_mb/1024).toFixed(1)}G</div></div>
          <div class="stat"><div class="stat-label">⚡</div><div class="stat-value">${gpu.power_w}W</div></div>`;
      }
      html+=`</div>`;
      // Cores
      if(cpu.usage_per_core&&cpu.usage_per_core.length){
        html+=`<div class="cores">`;
        cpu.usage_per_core.forEach((u,i)=>{
          html+=`<div class="core" style="background:${cc(u)}">${i+1}</div>`;
        });
        html+=`</div>`;
      }
      // Log
      const logs=d.training_log||[];
      if(logs.length){
        html+=`<div class="log-box">`;
        logs.slice(-6).forEach(l=>{html+=`<div class="log-line">${l.replace(/</g,'&lt;')}</div>`});
        html+=`</div>`;
      }
      html+=`</div>`;
    }
    document.getElementById('machines').innerHTML=html;
    document.getElementById('ts').textContent=new Date().toLocaleTimeString('ar-IQ');
  }catch(e){console.error(e)}
}
setInterval(refresh,3000);
refresh();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/all":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(_data, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD.encode("utf-8"))

    def log_message(self, format, *args):
        pass


def main():
    # Start fetch threads
    for name, info in MACHINES.items():
        t = threading.Thread(target=fetch_machine, args=(name, info), daemon=True)
        t.start()

    port = 9091
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"🖥️  Unified Monitor: http://localhost:{port}")
    print(f"   Watching: {', '.join(MACHINES.keys())}")
    server.serve_forever()


if __name__ == "__main__":
    main()
