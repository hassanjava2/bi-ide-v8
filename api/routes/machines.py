"""
نقطة استقبال Heartbeat — VPS Heartbeat Collector
يستقبل إحصائيات من كل الأجهزة ويحفظها في الذاكرة

🌐 الأجهزة ترسل → السيرفر يحفظ → الـ IDE يقرأ
"""

import time
from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["machines"])

# ──── تخزين مؤقت بالذاكرة ────
_machines: Dict[str, Dict[str, Any]] = {}
_STALE_SECONDS = 30  # إذا ما وصل heartbeat لمدة 30 ثانية = offline


@router.post("/api/v1/machines/heartbeat")
async def receive_heartbeat(request: Request):
    """استقبال heartbeat من جهاز"""
    try:
        data = await request.json()
        machine_id = data.get("machine_id", "unknown")
        data["_received_at"] = time.time()
        data["online"] = True
        _machines[machine_id] = data
        return {"status": "ok", "machine_id": machine_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/api/v1/machines")
async def get_all_machines():
    """كل الأجهزة المتصلة وحالتها"""
    now = time.time()
    result = []
    for mid, data in _machines.items():
        received = data.get("_received_at", 0)
        is_online = (now - received) < _STALE_SECONDS
        entry = {**data, "online": is_online, "machine": mid}
        if not is_online:
            entry["offline_since_sec"] = int(now - received)
        result.append(entry)

    return {
        "machines": result,
        "total": len(result),
        "online": sum(1 for m in result if m.get("online")),
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/api/v1/machines/{machine_id}")
async def get_machine(machine_id: str):
    """معلومات جهاز محدد"""
    if machine_id in _machines:
        data = _machines[machine_id]
        now = time.time()
        is_online = (now - data.get("_received_at", 0)) < _STALE_SECONDS
        return {**data, "online": is_online}
    return {"error": "Machine not found", "machine_id": machine_id}


@router.get("/monitor/dashboard", response_class=HTMLResponse)
async def monitor_dashboard_page():
    """لوحة المراقبة الذكية — VPS أولاً، LAN ثانياً"""
    return HTMLResponse(content=SMART_DASHBOARD_HTML)


# ──── Dashboard HTML مع VPS + LAN fallback ────
SMART_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<title>مراقب النظام — BI-IDE v8</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Tahoma,sans-serif;background:#0a0e17;color:#e0e0e0;padding:16px;min-height:100vh}
.hdr{display:flex;align-items:center;gap:12px;margin-bottom:20px;padding:12px 16px;
  background:linear-gradient(135deg,#1a1f2e,#0d1117);border-radius:12px;border:1px solid #30363d}
.hdr h1{font-size:18px;background:linear-gradient(120deg,#58a6ff,#1f6feb);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.dot{width:10px;height:10px;border-radius:50%;animation:p 2s infinite}
.dot.ok{background:#3fb950;box-shadow:0 0 8px #3fb950}
.dot.warn{background:#d29922;box-shadow:0 0 8px #d29922}
.dot.err{background:#f85149;box-shadow:0 0 8px #f85149}
@keyframes p{0%,100%{opacity:1}50%{opacity:.4}}
.conn{font-size:11px;padding:3px 10px;border-radius:10px;font-weight:600}
.conn.vps{background:#0d301a;color:#3fb950}
.conn.lan{background:#1c2333;color:#d29922}
.conn.none{background:#3d1117;color:#f85149}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.card{background:linear-gradient(145deg,#161b22,#0d1117);border:1px solid #30363d;
  border-radius:12px;padding:16px;transition:transform .2s,box-shadow .2s}
.card:hover{transform:translateY(-2px);box-shadow:0 4px 20px rgba(31,111,235,.15)}
.card.off{border-color:#f85149;opacity:.6}
.ch{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:6px}
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
  <div class="dot ok" id="dot"></div>
  <h1>🖥️ مراقب النظام — BI-IDE v8</h1>
  <span class="conn vps" id="conn">🌐 VPS</span>
  <span style="margin-right:auto;font-size:12px;color:#484f58" id="lu"></span>
</div>
<div class="grid" id="g"><div class="card"><div style="text-align:center;padding:40px;color:#484f58">⏳ جاري التحميل...</div></div></div>
<div class="ri">يتحدث تلقائياً كل 5 ثوانٍ — VPS أولاً، شبكة محلية إذا انقطع الإنترنت</div>

<script>
// مصادر البيانات — VPS أولاً، LAN ثانياً
const VPS_API = 'https://bi-iq.com/api/v1/machines';
const LAN_API = 'http://192.168.1.164:8090/api/v1/monitor';
let currentSource = 'vps';

function rm(m){
  const on=m.online!==false;
  let bd=on?'<span class="b on">🟢 متصل</span>':'<span class="b off">🔴 غير متصل</span>';
  if(m.training_active)bd+='<span class="b tr">⚡ تدريب</span>';
  if(!on){
    const sec=m.offline_since_sec?` (${m.offline_since_sec}s)`:'';
    return`<div class="card off"><div class="ch"><h2>${m.machine||m.machine_id||'?'}</h2>${bd}</div>
      <div style="text-align:center;padding:20px;color:#f85149">غير متصل${sec}</div></div>`;
  }
  let gpu='';
  if(m.gpu_temp_c!==undefined){
    gpu=`<div class="gs">
      <div class="m"><div class="l">🌡️ GPU</div><div class="v">${m.gpu_temp_c}<span class="u">°C</span></div></div>
      <div class="m"><div class="l">⚡ GPU</div><div class="v">${m.gpu_util||0}<span class="u">%</span></div>
        <div class="bc"><div class="br g${(m.gpu_util||0)>75?' hot':''}" style="width:${m.gpu_util||0}%"></div></div></div>
      <div class="m"><div class="l">💾 VRAM</div><div class="v">${((m.gpu_mem_used_mb||0)/1024).toFixed(1)}<span class="u">/${((m.gpu_mem_total_mb||0)/1024).toFixed(0)}GB</span></div></div>
    </div>`;
  }
  const name=m.machine||m.machine_id||'?';
  return`<div class="card"><div class="ch"><h2>${name}${m.gpu_name?' — '+m.gpu_name:''}</h2>${bd}</div>
    <div class="ms">
      <div class="m"><div class="l">🔧 CPU</div><div class="v">${m.cpu_percent>=0?m.cpu_percent:'—'}<span class="u">%</span></div>
        <div class="bc"><div class="br c${(m.cpu_percent||0)>85?' hot':''}" style="width:${Math.max(0,m.cpu_percent||0)}%"></div></div></div>
      <div class="m"><div class="l">🧠 RAM</div><div class="v">${m.ram_used_gb||'—'}<span class="u">/ ${m.ram_total_gb||'—'} GB</span></div>
        <div class="bc"><div class="br r" style="width:${m.ram_percent||0}%"></div></div></div>
      <div class="m"><div class="l">💿 Disk</div><div class="v">${m.disk_used_gb||'—'}<span class="u">/ ${m.disk_total_gb||'—'} GB</span></div>
        <div class="bc"><div class="br d" style="width:${m.disk_percent||0}%"></div></div></div>
      <div class="m"><div class="l">🌡️ CPU</div><div class="v">${m.cpu_temp_c||'—'}<span class="u">°C</span></div></div>
      ${gpu}
    </div></div>`;
}

function updateUI(data, source){
  const grid=document.getElementById('g');
  const conn=document.getElementById('conn');
  const dot=document.getElementById('dot');
  
  grid.innerHTML=(data.machines||[]).map(rm).join('')||'<div class="card"><div style="text-align:center;padding:40px;color:#484f58">لا أجهزة متصلة</div></div>';
  
  const srcText = source==='vps' ? '🌐 VPS' : '🏠 LAN';
  const srcClass = source==='vps' ? 'conn vps' : 'conn lan';
  conn.textContent = srcText;
  conn.className = srcClass;
  dot.className = source==='vps' ? 'dot ok' : 'dot warn';
  
  document.getElementById('lu').textContent=
    `آخر تحديث: ${new Date().toLocaleTimeString('ar-IQ')} | ${data.online||0}/${data.total||0} متصل | ${srcText}`;
}

async function tryFetch(url, timeout=4000){
  const ctrl=new AbortController();
  const t=setTimeout(()=>ctrl.abort(), timeout);
  try{
    const r=await fetch(url, {signal:ctrl.signal});
    clearTimeout(t);
    if(!r.ok) throw new Error(r.status);
    return await r.json();
  }catch(e){clearTimeout(t);throw e}
}

async function refresh(){
  // 1. جرّب VPS أولاً
  try{
    const data=await tryFetch(VPS_API, 4000);
    currentSource='vps';
    updateUI(data,'vps');
    return;
  }catch(e){}
  
  // 2. فشل VPS → جرّب LAN مباشرة
  try{
    const data=await tryFetch(LAN_API, 5000);
    currentSource='lan';
    updateUI(data,'lan');
    return;
  }catch(e){}
  
  // 3. لا VPS ولا LAN
  currentSource='none';
  document.getElementById('conn').textContent='❌ غير متصل';
  document.getElementById('conn').className='conn none';
  document.getElementById('dot').className='dot err';
}

refresh();
setInterval(refresh,5000);
</script>
</body></html>"""
