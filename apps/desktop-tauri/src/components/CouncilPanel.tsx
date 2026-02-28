import { useState, useEffect, useCallback } from "react";
import {
    Crown, Users, MessageSquare, Brain, Shield, Eye,
    Cpu, Zap, RefreshCw, Send, ChevronDown, ChevronRight,
    Activity, Globe, Database, Code, Lock, Layers,
} from "lucide-react";
import { council, hierarchy, orchestrator, WiseMan } from "../lib/tauri";

// Wise men icons map
const WISE_ICONS: Record<string, any> = {
    president: Crown, high_council: Users, scouts: Eye,
    guardian: Shield, meta_team: Cpu, domain_experts: Brain,
    execution: Zap, meta_architect: Layers, builder_council: Code,
    executive_controller: Activity, shadow_light: Globe,
    cosmic_bridge: Database, eternity: Lock,
    seventh_dimension: Eye, learning_core: Brain,
};

const STATUS_COLORS: Record<string, string> = {
    active: "text-green-400", idle: "text-yellow-400",
    training: "text-cyan-400", offline: "text-red-400",
    online: "text-green-400",
};

interface CouncilPanelProps {
    onSwitchToHierarchy?: () => void;
}

export function CouncilPanel({ onSwitchToHierarchy }: CouncilPanelProps) {
    const [wiseMen, setWiseMen] = useState<WiseMan[]>([]);
    const [councilStatus, setCouncilStatus] = useState<any>(null);
    const [orchHealth, setOrchHealth] = useState<any>(null);
    const [chatInput, setChatInput] = useState("");
    const [chatMessages, setChatMessages] = useState<Array<{ role: string; content: string; source?: string }>>([
        { role: "assistant", content: "🏛️ مرحباً بك في مجلس الحكماء. كلمني شتريد.", source: "council" },
    ]);
    const [isSending, setIsSending] = useState(false);
    const [loading, setLoading] = useState(true);
    const [expandedSection, setExpandedSection] = useState<string>("wise-men");

    const loadData = useCallback(async () => {
        setLoading(true);
        try {
            const [wm, status, health] = await Promise.allSettled([
                council.getWiseMen(),
                council.getStatus(),
                orchestrator.getHealth(),
            ]);
            if (wm.status === "fulfilled") setWiseMen(Array.isArray(wm.value) ? wm.value : []);
            if (status.status === "fulfilled") setCouncilStatus(status.value);
            if (health.status === "fulfilled") setOrchHealth(health.value);
        } catch (e) {
            console.error("Council load error:", e);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { loadData(); const t = setInterval(loadData, 30000); return () => clearInterval(t); }, [loadData]);

    const handleSend = async () => {
        if (!chatInput.trim() || isSending) return;
        const msg = chatInput.trim();
        setChatInput("");
        setChatMessages(prev => [...prev, { role: "user", content: msg }]);
        setIsSending(true);
        try {
            const result = await council.sendMessage(msg);
            const reply = result?.response || result?.message || result?.text || JSON.stringify(result);
            setChatMessages(prev => [...prev, {
                role: "assistant",
                content: reply,
                source: result?.council_member || result?.source || "council",
            }]);
        } catch (e: any) {
            setChatMessages(prev => [...prev, { role: "assistant", content: `❌ خطأ: ${e.message}` }]);
        } finally {
            setIsSending(false);
        }
    };

    const Section = ({ id, title, icon: Icon, children }: any) => (
        <div className="border-b border-dark-700/50">
            <button
                onClick={() => setExpandedSection(expandedSection === id ? "" : id)}
                className="w-full flex items-center gap-2 px-3 py-2 text-xs font-semibold text-dark-300 hover:bg-dark-800/50 transition-colors"
            >
                {expandedSection === id ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                <Icon className="w-3.5 h-3.5 text-primary-400" />
                {title}
            </button>
            {expandedSection === id && <div className="px-2 pb-2">{children}</div>}
        </div>
    );

    return (
        <div className="h-full flex flex-col bg-dark-900">
            {/* Header */}
            <div className="flex items-center justify-between px-3 py-2 border-b border-dark-700">
                <div className="flex items-center gap-2">
                    <Crown className="w-4 h-4 text-yellow-400" />
                    <span className="text-sm font-bold text-dark-100">مجلس الحكماء</span>
                </div>
                <div className="flex items-center gap-1">
                    <span className={`w-2 h-2 rounded-full ${councilStatus?.connected !== false ? "bg-green-400 animate-pulse" : "bg-red-400"}`} />
                    <button onClick={loadData} className="p-1 hover:bg-dark-800 rounded"><RefreshCw className={`w-3 h-3 text-dark-400 ${loading ? "animate-spin" : ""}`} /></button>
                </div>
            </div>

            {/* Status Bar */}
            {orchHealth && (
                <div className="flex items-center gap-3 px-3 py-1.5 bg-dark-800/50 text-[10px] text-dark-400 border-b border-dark-700/50">
                    <span>👥 {orchHealth.workers_online || 0}/{orchHealth.workers_total || 0} workers</span>
                    <span>📋 {orchHealth.jobs_total || 0} jobs</span>
                    <span>🏃 {orchHealth.jobs_running || 0} running</span>
                </div>
            )}

            <div className="flex-1 overflow-auto">
                {/* Wise Men */}
                <Section id="wise-men" title={`الحكماء (${wiseMen.length})`} icon={Users}>
                    <div className="grid grid-cols-2 gap-1">
                        {wiseMen.map((wm, i) => {
                            const Icon = WISE_ICONS[wm.id] || Brain;
                            const statusColor = STATUS_COLORS[wm.status] || "text-dark-400";
                            return (
                                <div key={wm.id || i} className="flex items-center gap-1.5 p-1.5 rounded bg-dark-800/60 hover:bg-dark-700/60 transition-colors cursor-default group">
                                    <Icon className={`w-3.5 h-3.5 ${statusColor} flex-shrink-0`} />
                                    <div className="min-w-0">
                                        <div className="text-[10px] font-medium text-dark-200 truncate">{wm.name || wm.id}</div>
                                        <div className="text-[9px] text-dark-500 truncate">{wm.role || wm.specialization || ""}</div>
                                    </div>
                                </div>
                            );
                        })}
                        {wiseMen.length === 0 && !loading && (
                            <div className="col-span-2 text-center text-dark-500 text-xs py-4">
                                جاري التحميل...
                            </div>
                        )}
                    </div>
                </Section>

                {/* Chat */}
                <Section id="chat" title="محادثة المجلس" icon={MessageSquare}>
                    <div className="space-y-1.5 max-h-48 overflow-auto mb-2">
                        {chatMessages.map((msg, i) => (
                            <div key={i} className={`text-[11px] p-2 rounded ${msg.role === "user"
                                    ? "bg-primary-700/30 text-primary-100"
                                    : "bg-dark-800 text-dark-200"
                                }`}>
                                {msg.source && <span className="text-[9px] text-dark-500 block mb-0.5">{msg.source}</span>}
                                {msg.content}
                            </div>
                        ))}
                    </div>
                    <div className="flex gap-1">
                        <input
                            value={chatInput}
                            onChange={e => setChatInput(e.target.value)}
                            onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); handleSend(); } }}
                            placeholder="أرسل أمر للمجلس..."
                            className="flex-1 px-2 py-1.5 bg-dark-800 border border-dark-700 rounded text-xs text-dark-100 placeholder-dark-500 focus:outline-none focus:border-primary-500"
                            dir="rtl"
                        />
                        <button onClick={handleSend} disabled={isSending} className="px-2 py-1.5 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 rounded text-white">
                            <Send className="w-3 h-3" />
                        </button>
                    </div>
                </Section>

                {/* Quick Actions */}
                <Section id="actions" title="إجراءات سريعة" icon={Zap}>
                    <div className="grid grid-cols-2 gap-1">
                        {[
                            { label: "🔮 حكمة", fn: async () => { const w = await hierarchy.getWisdom(); setChatMessages(p => [...p, { role: "assistant", content: w?.wisdom || JSON.stringify(w) }]); } },
                            { label: "🛡️ أمان", fn: async () => { const g = await hierarchy.getGuardianStatus(); setChatMessages(p => [...p, { role: "assistant", content: `Guardian: ${JSON.stringify(g?.status || g)}` }]); } },
                            { label: "📊 مقاييس", fn: async () => { const m = await council.getMetrics(); setChatMessages(p => [...p, { role: "assistant", content: JSON.stringify(m, null, 1) }]); } },
                            { label: "📜 سجل", fn: async () => { const h = await council.getHistory(); setChatMessages(p => [...p, { role: "assistant", content: `${h.length} رسالة بالسجل` }]); } },
                        ].map(({ label, fn }, i) => (
                            <button key={i} onClick={fn} className="text-[10px] py-1.5 px-2 bg-dark-800 hover:bg-dark-700 rounded text-dark-300 transition-colors">
                                {label}
                            </button>
                        ))}
                    </div>
                </Section>
            </div>
        </div>
    );
}
