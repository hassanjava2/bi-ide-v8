import { useState, useEffect, useCallback } from "react";
import {
    Layers, Cpu, Activity, Brain, RefreshCw,
    ChevronDown, ChevronRight, BarChart2,
    Zap, Shield, Eye, Crown, Code, Globe,
    Database, Lock, Users,
} from "lucide-react";
import { hierarchy, orchestrator, LayerStatus } from "../lib/tauri";

const LAYER_ICONS: Record<string, any> = {
    president: Crown, seventh_dimension: Eye, high_council: Users,
    shadow_light: Globe, scouts: Eye, meta_team: Cpu,
    domain_experts: Brain, execution: Zap, meta_architect: Layers,
    builder_council: Code, executive_controller: Activity,
    guardian: Shield, cosmic_bridge: Database, eternity: Lock,
    learning_core: Brain,
};

const LAYER_EMOJIS: Record<string, string> = {
    president: "📊", seventh_dimension: "🔮", high_council: "🧠",
    shadow_light: "⚖️", scouts: "🔍", meta_team: "⚙️",
    domain_experts: "🎓", execution: "🚀", meta_architect: "🏗️",
    builder_council: "🔨", executive_controller: "🎮",
    guardian: "🛡️", cosmic_bridge: "🌌", eternity: "💾",
    learning_core: "🧬",
};

export function HierarchyPanel() {
    const [hierarchyStatus, setHierarchyStatus] = useState<any>(null);
    const [orchHealth, setOrchHealth] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [expandedLayer, setExpandedLayer] = useState<string>("");

    const loadData = useCallback(async () => {
        setLoading(true);
        try {
            const [hs, oh] = await Promise.allSettled([
                hierarchy.getStatus(),
                orchestrator.getHealth(),
            ]);
            if (hs.status === "fulfilled") setHierarchyStatus(hs.value);
            if (oh.status === "fulfilled") setOrchHealth(oh.value);
        } catch (e) {
            console.error("Hierarchy load error:", e);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { loadData(); const t = setInterval(loadData, 15000); return () => clearInterval(t); }, [loadData]);

    const layers = hierarchyStatus?.layers
        ? Object.entries(hierarchyStatus.layers).map(([key, data]: [string, any]) => ({
            ...(data as LayerStatus),
            name: key,
        }))
        : [];

    const gpu = hierarchyStatus?.gpu;

    return (
        <div className="h-full flex flex-col bg-dark-900">
            {/* Header */}
            <div className="flex items-center justify-between px-3 py-2 border-b border-dark-700">
                <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-cyan-400" />
                    <span className="text-sm font-bold text-dark-100">النظام الهرمي</span>
                </div>
                <div className="flex items-center gap-1">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded ${hierarchyStatus?.is_training ? "bg-green-500/20 text-green-400" : "bg-dark-700 text-dark-400"}`}>
                        {hierarchyStatus?.is_training ? "🔥 تدريب" : "⏸️ متوقف"}
                    </span>
                    <button onClick={loadData} className="p-1 hover:bg-dark-800 rounded">
                        <RefreshCw className={`w-3 h-3 text-dark-400 ${loading ? "animate-spin" : ""}`} />
                    </button>
                </div>
            </div>

            {/* GPU Stats */}
            {gpu && (
                <div className="px-3 py-2 bg-dark-800/50 border-b border-dark-700/50">
                    <div className="flex items-center justify-between text-[10px] text-dark-300 mb-1">
                        <span className="flex items-center gap-1"><Cpu className="w-3 h-3 text-green-400" /> {gpu.name || "GPU"}</span>
                        <span className="text-green-400 font-bold">{gpu.utilization || 0}%</span>
                    </div>
                    <div className="h-1.5 bg-dark-700 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-green-500 to-cyan-500 transition-all duration-500"
                            style={{ width: `${gpu.utilization || 0}%` }}
                        />
                    </div>
                    <div className="flex justify-between text-[9px] text-dark-500 mt-0.5">
                        <span>VRAM: {(gpu.memory_used || 0).toFixed(1)} / {(gpu.memory_total || 0).toFixed(1)} GB</span>
                        <span>{hierarchyStatus?.device || ""}</span>
                    </div>
                </div>
            )}

            {/* Orchestrator Stats */}
            {orchHealth && (
                <div className="grid grid-cols-3 gap-1 px-3 py-1.5 border-b border-dark-700/50">
                    <div className="text-center">
                        <div className="text-[10px] text-dark-500">Workers</div>
                        <div className="text-xs font-bold text-green-400">{orchHealth.workers_online || 0}/{orchHealth.workers_total || 0}</div>
                    </div>
                    <div className="text-center">
                        <div className="text-[10px] text-dark-500">Jobs</div>
                        <div className="text-xs font-bold text-cyan-400">{orchHealth.jobs_total || 0}</div>
                    </div>
                    <div className="text-center">
                        <div className="text-[10px] text-dark-500">Running</div>
                        <div className="text-xs font-bold text-yellow-400">{orchHealth.jobs_running || 0}</div>
                    </div>
                </div>
            )}

            {/* Layers List */}
            <div className="flex-1 overflow-auto">
                <div className="px-3 py-1.5 text-[10px] font-semibold text-dark-400 uppercase flex items-center gap-1">
                    <BarChart2 className="w-3 h-3" /> {layers.length} طبقة
                </div>

                {layers.map(layer => {
                    const Icon = LAYER_ICONS[layer.name] || Brain;
                    const emoji = LAYER_EMOJIS[layer.name] || "🎯";
                    const isExpanded = expandedLayer === layer.name;
                    const accColor = (layer.accuracy || 0) > 80 ? "text-green-400" : (layer.accuracy || 0) > 50 ? "text-yellow-400" : "text-red-400";

                    return (
                        <div key={layer.name} className="border-b border-dark-800/50">
                            <button
                                onClick={() => setExpandedLayer(isExpanded ? "" : layer.name)}
                                className="w-full flex items-center gap-2 px-3 py-2 hover:bg-dark-800/30 transition-colors text-left"
                            >
                                {isExpanded ? <ChevronDown className="w-3 h-3 text-dark-500" /> : <ChevronRight className="w-3 h-3 text-dark-500" />}
                                <span className="text-xs">{emoji}</span>
                                <div className="flex-1 min-w-0">
                                    <div className="text-[11px] font-medium text-dark-200 truncate">{layer.name}</div>
                                    <div className="text-[9px] text-dark-500 truncate">{layer.specialization || ""}</div>
                                </div>
                                <div className="flex items-center gap-2 flex-shrink-0">
                                    <span className={`text-[10px] font-mono ${accColor}`}>{(layer.accuracy || 0).toFixed(0)}%</span>
                                    <span className="text-[10px] text-dark-500">E{layer.epoch || 0}</span>
                                </div>
                            </button>

                            {isExpanded && (
                                <div className="px-3 pb-2 ml-5">
                                    <div className="grid grid-cols-2 gap-1 text-[10px]">
                                        <div className="bg-dark-800/50 rounded p-1.5">
                                            <span className="text-dark-500">Loss</span>
                                            <span className="text-dark-200 float-right font-mono">{(layer.loss || 0).toFixed(4)}</span>
                                        </div>
                                        <div className="bg-dark-800/50 rounded p-1.5">
                                            <span className="text-dark-500">Accuracy</span>
                                            <span className={`float-right font-mono ${accColor}`}>{(layer.accuracy || 0).toFixed(1)}%</span>
                                        </div>
                                        <div className="bg-dark-800/50 rounded p-1.5">
                                            <span className="text-dark-500">Epoch</span>
                                            <span className="text-dark-200 float-right font-mono">{layer.epoch || 0}</span>
                                        </div>
                                        <div className="bg-dark-800/50 rounded p-1.5">
                                            <span className="text-dark-500">Samples</span>
                                            <span className="text-dark-200 float-right font-mono">{(layer.samples || 0).toLocaleString()}</span>
                                        </div>
                                        <div className="bg-dark-800/50 rounded p-1.5">
                                            <span className="text-dark-500">Fetches</span>
                                            <span className="text-dark-200 float-right font-mono">{layer.fetches || 0}</span>
                                        </div>
                                        <div className="bg-dark-800/50 rounded p-1.5">
                                            <span className="text-dark-500">VRAM</span>
                                            <span className="text-dark-200 float-right font-mono">{(layer.vram_gb || 0).toFixed(2)} GB</span>
                                        </div>
                                    </div>
                                    {/* Accuracy bar */}
                                    <div className="mt-1.5 h-1 bg-dark-700 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full transition-all duration-300 ${(layer.accuracy || 0) > 80
                                                ? "bg-gradient-to-r from-green-500 to-emerald-400"
                                                : (layer.accuracy || 0) > 50
                                                    ? "bg-gradient-to-r from-yellow-500 to-orange-400"
                                                    : "bg-gradient-to-r from-red-500 to-pink-400"
                                                }`}
                                            style={{ width: `${Math.min(100, layer.accuracy || 0)}%` }}
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    );
                })}

                {layers.length === 0 && !loading && (
                    <div className="text-center text-dark-500 text-xs py-8">
                        <Layers className="w-8 h-8 mx-auto mb-2 opacity-30" />
                        <p>لا يوجد بيانات طبقات</p>
                        <p className="text-[10px] mt-1">تأكد من اتصال الخادم</p>
                    </div>
                )}

                {loading && layers.length === 0 && (
                    <div className="text-center text-dark-500 text-xs py-8">
                        <RefreshCw className="w-6 h-6 mx-auto mb-2 animate-spin opacity-30" />
                        <p>جاري التحميل...</p>
                    </div>
                )}
            </div>
        </div>
    );
}
