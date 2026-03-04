"""
لوحة مراقبة متعددة الأجهزة — Multi-Machine Monitor
بيانات حقيقية: CPU, RAM, GPU, Disk, حرارة, شبكة, تدريب

🖥️ مثل Task Manager لكل الأجهزة — RTX 5090 + Windows + VPS
"""

import asyncio
import subprocess
import time
from typing import Dict, Any, List
from datetime import datetime


class MachineMonitor:
    """يجمع بيانات جهاز واحد عبر SSH أو محلياً"""
    
    def __init__(self, name: str, host: str, user: str, os_type: str = "linux", is_local: bool = False):
        self.name = name
        self.host = host
        self.user = user
        self.os_type = os_type
        self.is_local = is_local
    
    def _run(self, cmd: str, timeout: int = 8) -> str:
        """تنفيذ أمر — محلي أو عبر SSH"""
        try:
            if self.is_local:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            else:
                full = f'ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes {self.user}@{self.host} "{cmd}"'
                r = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip()
        except Exception:
            return ""
    
    async def get_stats(self) -> Dict[str, Any]:
        start = time.time()
        loop = asyncio.get_event_loop()
        
        if self.os_type == "windows":
            raw = await loop.run_in_executor(None, self._get_win_raw)
        else:
            raw = await loop.run_in_executor(None, self._get_linux_raw)
        
        raw["machine"] = self.name
        raw["host"] = self.host
        raw["os"] = self.os_type
        raw["response_ms"] = int((time.time() - start) * 1000)
        raw["timestamp"] = datetime.now().isoformat()
        raw["online"] = raw.get("cpu_percent", -1) >= 0
        return raw

    def _get_linux_raw(self) -> Dict:
        cmd = (
            "echo C=$(top -bn1 | grep Cpu | awk '{print 100-$8}') && "
            "echo T=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null) && "
            "echo M=$(free -b | awk '/Mem/{printf \"%s/%s\", $3, $2}') && "
            "echo D=$(df -B1 / | awk 'NR==2{printf \"%s/%s\", $3, $2}') && "
            "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,name "
            "--format=csv,noheader,nounits 2>/dev/null | awk '{print \"G=\"$0}' && "
            "echo S=$(systemctl is-active bi-training-daemon 2>/dev/null) && "
            "echo L=$(cat /proc/loadavg 2>/dev/null | cut -d' ' -f1-2)"
        )
        out = self._run(cmd)
        s = {}
        for line in out.split('\n'):
            line = line.strip()
            if line.startswith("C="):
                try: s["cpu_percent"] = round(float(line[2:]), 1)
                except: pass
            elif line.startswith("T="):
                try: s["cpu_temp_c"] = round(int(line[2:]) / 1000, 1)
                except: pass
            elif line.startswith("M="):
                parts = line[2:].split("/")
                if len(parts) == 2:
                    u, t = float(parts[0]), float(parts[1])
                    s["ram_used_gb"] = round(u/1e9, 1)
                    s["ram_total_gb"] = round(t/1e9, 1)
                    s["ram_percent"] = round(u/t*100, 1) if t else 0
            elif line.startswith("D="):
                parts = line[2:].split("/")
                if len(parts) == 2:
                    u, t = float(parts[0]), float(parts[1])
                    s["disk_used_gb"] = round(u/1e9, 1)
                    s["disk_total_gb"] = round(t/1e9, 1)
                    s["disk_percent"] = round(u/t*100, 1) if t else 0
            elif line.startswith("G="):
                p = [x.strip() for x in line[2:].split(",")]
                if len(p) >= 4:
                    s["gpu_temp_c"] = int(p[0]) if p[0].isdigit() else 0
                    s["gpu_util"] = int(p[1]) if p[1].isdigit() else 0
                    s["gpu_mem_used_mb"] = int(p[2]) if p[2].isdigit() else 0
                    s["gpu_mem_total_mb"] = int(p[3]) if p[3].isdigit() else 0
                    if len(p) >= 5: s["gpu_name"] = p[4].strip()
            elif line.startswith("S="):
                s["training_active"] = line[2:] == "active"
            elif line.startswith("L="):
                parts = line[2:].split()
                if parts: s["load_1m"] = float(parts[0])
        if "cpu_percent" not in s: s["cpu_percent"] = -1
        return s

    def _get_win_raw(self) -> Dict:
        cmd = (
            "powershell -Command \""
            "$c=(Get-CimInstance Win32_Processor).LoadPercentage;"
            "$o=Get-CimInstance Win32_OperatingSystem;"
            "$mt=$o.TotalVisibleMemorySize*1024;$mf=$o.FreePhysicalMemory*1024;"
            "$d=Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3'|Select -First 1;"
            "$g=nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,name "
            "--format=csv,noheader,nounits 2>$null;"
            "Write-Output \\\"C=$c|M=$($mt-$mf)/$mt|D=$($d.Size-$d.FreeSpace)/$($d.Size)|G=$g\\\""
            "\""
        )
        out = self._run(cmd, timeout=12)
        s = {}
        for part in out.split("|"):
            part = part.strip()
            if part.startswith("C="):
                try: s["cpu_percent"] = float(part[2:])
                except: pass
            elif part.startswith("M="):
                vals = part[2:].split("/")
                if len(vals) == 2:
                    u, t = float(vals[0]), float(vals[1])
                    s["ram_used_gb"] = round(u/1e9, 1)
                    s["ram_total_gb"] = round(t/1e9, 1)
                    s["ram_percent"] = round(u/t*100, 1) if t else 0
            elif part.startswith("D="):
                vals = part[2:].split("/")
                if len(vals) == 2:
                    u, t = float(vals[0]), float(vals[1])
                    s["disk_used_gb"] = round(u/1e9, 1)
                    s["disk_total_gb"] = round(t/1e9, 1)
                    s["disk_percent"] = round(u/t*100, 1) if t else 0
            elif part.startswith("G="):
                p = [x.strip() for x in part[2:].split(",")]
                if len(p) >= 4:
                    s["gpu_temp_c"] = int(p[0]) if p[0].isdigit() else 0
                    s["gpu_util"] = int(p[1]) if p[1].isdigit() else 0
                    s["gpu_mem_used_mb"] = int(p[2]) if p[2].isdigit() else 0
                    s["gpu_mem_total_mb"] = int(p[3]) if p[3].isdigit() else 0
                    if len(p) >= 5: s["gpu_name"] = p[4].strip()
        if "cpu_percent" not in s: s["cpu_percent"] = -1
        return s


# ──────── الأجهزة المسجلة ────────
ALL_MACHINES = [
    MachineMonitor("RTX 5090", "localhost", "bi", "linux", is_local=True),
    MachineMonitor("Windows RTX 4050", "192.168.1.130", "BI", "windows"),
    MachineMonitor("VPS bi-iq.com", "bi-iq.com", "root", "linux"),
]


async def collect_all() -> Dict[str, Any]:
    """جمع إحصائيات كل الأجهزة بالتوازي"""
    tasks = [m.get_stats() for m in ALL_MACHINES]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    machines = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            machines.append({"machine": ALL_MACHINES[i].name, "online": False, "error": str(r)})
        else:
            machines.append(r)
    return {
        "machines": machines,
        "total": len(ALL_MACHINES),
        "online": sum(1 for m in machines if m.get("online")),
        "timestamp": datetime.now().isoformat(),
    }


# ──────── HTML Dashboard ────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<title>مراقب النظام — BI-IDE</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Tahoma,sans-serif;background:#0a0e17;color:#e0e0e0;padding:16px}
.hdr{display:flex;align-items:center;gap:12px;margin-bottom:20px;padding:12px 16px;
  background:linear-gradient(135deg,#1a1f2e,#0d1117);border-radius:12px;border:1px solid #30363d}
.hdr h1{font-size:18px;background:linear-gradient(120deg,#58a6ff,#1f6feb);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.dot{width:10px;height:10px;border-radius:50%;background:#3fb950;
  box-shadow:0 0 8px #3fb950;animation:p 2s infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.4}}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.card{background:linear-gradient(145deg,#161b22,#0d1117);border:1px solid #30363d;
  border-radius:12px;padding:16px;transition:transform .2s,box-shadow .2s}
.card:hover{transform:translateY(-2px);box-shadow:0 4px 20px rgba(31,111,235,.15)}
.card.off{border-color:#f85149;opacity:.6}
.ch{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.ch h2{font-size:15px;color:#58a6ff}
.b{font-size:11px;padding:3px 8px;border-radius:12px;font-weight:600;margin-right:4px}
.b.on{background:#0d301a;color:#3fb950}
.b.off{background:#3d1117;color:#f85149}
.b.tr{background:#1c2333;color:#d29922;animation:p 1.5s infinite}
.ms{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.m{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:10px}
.m .l{font-size:11px;color:#8b949e;margin-bottom:4px}
.m .v{font-size:20px;font-weight:700}
.m .u{font-size:12px;color:#8b949e;margin-right:3px}
.bc{height:4px;background:#21262d;border-radius:2px;margin-top:6px;overflow:hidden}
.br{height:100%;border-radius:2px;transition:width .5s}
.br.c{background:linear-gradient(90deg,#3fb950,#f0883e)}
.br.r{background:linear-gradient(90deg,#58a6ff,#bc8cff)}
.br.g{background:linear-gradient(90deg,#1f6feb,#f85149)}
.br.d{background:linear-gradient(90deg,#8b949e,#d29922)}
.br.hot{background:linear-gradient(90deg,#f0883e,#f85149)!important}
.gs{grid-column:1/-1;display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.gs .m{padding:8px}.gs .m .v{font-size:16px}
.ri{text-align:center;margin-top:16px;font-size:11px;color:#484f58}
</style>
</head>
<body>
<div class="hdr">
  <div class="dot"></div>
  <h1>🖥️ مراقب النظام — BI-IDE v8</h1>
  <span style="margin-right:auto;font-size:12px;color:#484f58" id="lu"></span>
</div>
<div class="grid" id="g"><div class="card"><div style="text-align:center;padding:40px;color:#484f58">⏳ جاري التحميل...</div></div></div>
<div class="ri">يتحدث تلقائياً كل 5 ثوانٍ</div>
<script>
const API='/api/v1/monitor';
function rm(m){
  const on=m.online!==false;
  let bd=on?'<span class="b on">🟢 متصل</span>':'<span class="b off">🔴 غير متصل</span>';
  if(m.training_active)bd+='<span class="b tr">⚡ تدريب</span>';
  if(!on)return`<div class="card off"><div class="ch"><h2>${m.machine}</h2>${bd}</div><div style="text-align:center;padding:20px;color:#f85149">غير متصل — ${m.response_ms||'?'}ms</div></div>`;
  let gpu='';
  if(m.gpu_temp_c!==undefined){
    gpu=`<div class="gs">
      <div class="m"><div class="l">🌡️ GPU</div><div class="v">${m.gpu_temp_c}<span class="u">°C</span></div></div>
      <div class="m"><div class="l">⚡ GPU</div><div class="v">${m.gpu_util}<span class="u">%</span></div>
        <div class="bc"><div class="br g${m.gpu_util>75?' hot':''}" style="width:${m.gpu_util}%"></div></div></div>
      <div class="m"><div class="l">💾 VRAM</div><div class="v">${(m.gpu_mem_used_mb/1024).toFixed(1)}<span class="u">/${(m.gpu_mem_total_mb/1024).toFixed(0)}GB</span></div></div>
    </div>`;
  }
  return`<div class="card"><div class="ch"><h2>${m.machine}${m.gpu_name?' — '+m.gpu_name:''}</h2>${bd}</div>
    <div class="ms">
      <div class="m"><div class="l">🔧 CPU</div><div class="v">${m.cpu_percent>=0?m.cpu_percent:'—'}<span class="u">%</span></div>
        <div class="bc"><div class="br c${m.cpu_percent>85?' hot':''}" style="width:${Math.max(0,m.cpu_percent)}%"></div></div></div>
      <div class="m"><div class="l">🧠 RAM</div><div class="v">${m.ram_used_gb||'—'}<span class="u">/ ${m.ram_total_gb||'—'} GB</span></div>
        <div class="bc"><div class="br r" style="width:${m.ram_percent||0}%"></div></div></div>
      <div class="m"><div class="l">💿 Disk</div><div class="v">${m.disk_used_gb||'—'}<span class="u">/ ${m.disk_total_gb||'—'} GB</span></div>
        <div class="bc"><div class="br d" style="width:${m.disk_percent||0}%"></div></div></div>
      <div class="m"><div class="l">🌡️ CPU</div><div class="v">${m.cpu_temp_c||'—'}<span class="u">°C</span></div></div>
      ${gpu}
    </div></div>`;
}
async function refresh(){
  try{
    const r=await fetch(API);const d=await r.json();
    document.getElementById('g').innerHTML=d.machines.map(rm).join('');
    document.getElementById('lu').textContent='آخر تحديث: '+new Date().toLocaleTimeString('ar-IQ')+' | '+d.online+'/'+d.total+' متصل';
  }catch(e){console.error(e)}
}
refresh();setInterval(refresh,5000);
</script>
</body></html>"""
