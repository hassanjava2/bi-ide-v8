import { useState, useEffect } from "react";
import { Layers, Eye, Dna, Radar, RefreshCw, Activity } from "lucide-react";
import { apiGet, apiPost } from "../lib/api-config";

export function BrainDashboardPanel() {
    const [layers, setLayers] = useState<any>(null);
    const [evolution, setEvolution] = useState<any>(null);
    const [scout, setScout] = useState<any>(null);
    const [vision, setVision] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    const fetchAll = async () => {
        setLoading(true);
        try {
            const [layersRes, evoRes, scoutRes, visionRes] = await Promise.all([
                apiGet("/api/layers").catch(() => null),
                apiGet("/api/evolution/status").catch(() => null),
                apiGet("/api/scout/status").catch(() => null),
                apiGet("/api/vision/status").catch(() => null),
            ]);
            setLayers(layersRes);
            setEvolution(evoRes);
            setScout(scoutRes);
            setVision(visionRes);
        } finally { setLoading(false); }
    };

    useEffect(() => { fetchAll(); }, []);

    const snapshot = async () => {
        setLoading(true);
        try {
            await apiPost("/api/evolution/snapshot", { reason: "Manual snapshot from IDE" });
            await fetchAll();
        } finally { setLoading(false); }
    };

    const scoutCycle = async () => {
        setLoading(true);
        try {
            await apiPost("/api/scout/cycle", {});
            await fetchAll();
        } finally { setLoading(false); }
    };

    return (
        <div className="p-3">
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-cyan-400" />
                    <span className="text-sm font-bold text-dark-200">الدماغ المتقدم 🧠</span>
                </div>
                <button onClick={fetchAll} disabled={loading} className="p-1 hover:bg-dark-800 rounded">
                    <RefreshCw className={`w-3 h-3 text-dark-400 ${loading ? "animate-spin" : ""}`} />
                </button>
            </div>

            {/* Layers */}
            <div className="mb-3">
                <div className="flex items-center gap-1 mb-1">
                    <Layers className="w-3 h-3 text-purple-400" />
                    <span className="text-xs font-medium text-dark-300">الطبقات</span>
                </div>
                {layers?.stats && (
                    <div className="bg-dark-800 rounded p-2">
                        <div className="flex justify-between text-[10px] text-dark-400">
                            <span>كلي: <span className="text-purple-400 font-bold">{layers.stats.total_layers}</span></span>
                            <span>نشط: <span className="text-green-400 font-bold">{layers.stats.active}</span></span>
                            <span>تلقائي: <span className="text-blue-400 font-bold">{layers.stats.auto_created}</span></span>
                        </div>
                    </div>
                )}
                {layers?.hierarchy && (
                    <pre className="bg-dark-850 rounded p-2 mt-1 text-[9px] text-dark-400 overflow-auto max-h-[150px] whitespace-pre-wrap">{layers.hierarchy}</pre>
                )}
            </div>

            {/* Evolution */}
            <div className="mb-3">
                <div className="flex items-center gap-1 mb-1">
                    <Dna className="w-3 h-3 text-green-400" />
                    <span className="text-xs font-medium text-dark-300">التطور الذاتي</span>
                </div>
                {evolution && (
                    <div className="bg-dark-800 rounded p-2">
                        <div className="flex justify-between text-[10px] text-dark-400 mb-1">
                            <span>نسخ: <span className="text-green-400 font-bold">{evolution.total_versions}</span></span>
                            <span>تحسينات: <span className="text-blue-400 font-bold">{evolution.improvements}</span></span>
                            <span>انتظار: <span className={evolution.pending ? "text-yellow-400" : "text-dark-500"}>{evolution.pending ? "نعم" : "لا"}</span></span>
                        </div>
                        <button onClick={snapshot} disabled={loading}
                            className="w-full mt-1 py-1 bg-green-700 hover:bg-green-600 rounded text-[10px] font-medium text-white disabled:opacity-50">
                            📸 Snapshot
                        </button>
                    </div>
                )}
            </div>

            {/* Scout */}
            <div className="mb-3">
                <div className="flex items-center gap-1 mb-1">
                    <Radar className="w-3 h-3 text-yellow-400" />
                    <span className="text-xs font-medium text-dark-300">الكشافة</span>
                </div>
                {scout && (
                    <div className="bg-dark-800 rounded p-2">
                        <div className="flex justify-between text-[10px] text-dark-400 mb-1">
                            <span>حالة: <span className={scout.online ? "text-green-400" : "text-red-400"}>{scout.online ? "Online" : "Offline"}</span></span>
                            <span>LAN: <span className="text-blue-400 font-bold">{scout.lan_devices}</span></span>
                            <span>ملفات: <span className="text-purple-400 font-bold">{scout.seen_files}</span></span>
                        </div>
                        <button onClick={scoutCycle} disabled={loading}
                            className="w-full mt-1 py-1 bg-yellow-700 hover:bg-yellow-600 rounded text-[10px] font-medium text-white disabled:opacity-50">
                            🔍 دورة كشافة
                        </button>
                    </div>
                )}
            </div>

            {/* Vision */}
            <div>
                <div className="flex items-center gap-1 mb-1">
                    <Eye className="w-3 h-3 text-cyan-400" />
                    <span className="text-xs font-medium text-dark-300">الرؤية</span>
                </div>
                {vision && (
                    <div className="bg-dark-800 rounded p-2">
                        <div className="flex justify-between text-[10px] text-dark-400">
                            <span>نموذج: <span className="text-cyan-400">{vision.image_model}</span></span>
                            <span>كاميرات: <span className="text-blue-400 font-bold">{vision.cameras}</span></span>
                            <span>تنبيهات: <span className={vision.alerts > 0 ? "text-red-400 font-bold" : "text-dark-500"}>{vision.alerts}</span></span>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
