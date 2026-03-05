import { useState, useEffect, useCallback, useRef } from "react";
import {
    Crown, Users, MessageSquare, Brain, Shield, Eye,
    Cpu, Zap, RefreshCw, Send, ChevronDown, ChevronRight,
    Activity, Globe, Database, Code, Lock, Layers,
    Thermometer, HardDrive,
} from "lucide-react";
import { sendCouncilMessage } from "../config/api";
import { invoke } from "@tauri-apps/api/core";

// Wise men icons map
const WISE_ICONS: Record<string, any> = {
    president: Crown, identity: Users, strategy: Brain,
    security: Shield, knowledge: Database, code: Code,
    scout: Eye, medicine: Activity, finance: Globe,
    survival: Zap, engineering: Cpu, balance: Layers,
    learning: Brain, meta: Layers, shadow: Eye,
    eternity: Lock, high_council: Users, guardian: Shield,
    meta_team: Cpu, domain_experts: Brain, execution: Zap,
    meta_architect: Layers, builder_council: Code,
    executive_controller: Activity, shadow_light: Globe,
    cosmic_bridge: Database, seventh_dimension: Eye,
    learning_core: Brain,
};

const STATUS_COLORS: Record<string, string> = {
    active: "text-green-400", idle: "text-yellow-400",
    training: "text-cyan-400", offline: "text-red-400",
    online: "text-green-400",
};

// Section component
function Section({ id, title, icon: Icon, badge, expanded, onToggle, children }: {
    id: string; title: string; icon: any; badge?: string; expanded: boolean;
    onToggle: (id: string) => void; children: React.ReactNode;
}) {
    return (
        <div className="border-b border-dark-700/50">
            <button
                onClick={() => onToggle(id)}
                className="w-full flex items-center gap-2 px-3 py-2 text-xs font-semibold text-dark-300 hover:bg-dark-800/50 transition-colors"
            >
                {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                <Icon className="w-3.5 h-3.5 text-primary-400" />
                <span className="flex-1 text-right">{title}</span>
                {badge && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-primary-600/30 text-primary-300 font-mono">
                        {badge}
                    </span>
                )}
            </button>
            {expanded && <div className="px-2 pb-2">{children}</div>}
        </div>
    );
}

interface WiseMan {
    id: string;
    name: string;
    role?: string;
    status: string;
    specialization?: string;
}

interface GPUInfo {
    temperature?: number;
    utilization?: number;
    memory_used_mb?: number;
    memory_total_mb?: number;
}

interface ChatMessage {
    role: string;
    content: string;
    source?: string;
    confidence?: number;
    processing_time_ms?: number;
    wise_man?: string;
}

interface CouncilPanelProps {
    onSwitchToHierarchy?: () => void;
}

export function CouncilPanel({ onSwitchToHierarchy }: CouncilPanelProps) {
    const [wiseMen, setWiseMen] = useState<WiseMan[]>([]);
    const [gpuInfo, setGpuInfo] = useState<GPUInfo>({});
    const [trainingActive, setTrainingActive] = useState(false);
    const [trainingSamples, setTrainingSamples] = useState(0);
    const [chatInput, setChatInput] = useState("");
    const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
    const [isSending, setIsSending] = useState(false);
    const [expandedSection, setExpandedSection] = useState<string>("chat");
    const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Auto-scroll chat
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [chatMessages]);

    // Fetch sages from RTX API
    const fetchSages = useCallback(async () => {
        try {
            const response: any = await invoke("proxy_request", {
                url: "http://100.104.35.44:8090/council/sages",
                method: "GET",
            }).catch(() => null);

            if (response) {
                const data = typeof response === "string" ? JSON.parse(response) : response;
                if (data.sages) setWiseMen(data.sages);
                if (data.gpu) setGpuInfo(data.gpu);
                if (data.training_active !== undefined) setTrainingActive(data.training_active);
                if (data.training_samples !== undefined) setTrainingSamples(data.training_samples);
                setLastRefresh(new Date());
            }
        } catch {
            // Try direct fetch
            try {
                const resp = await fetch("http://100.104.35.44:8090/council/sages");
                const data = await resp.json();
                if (data.sages) setWiseMen(data.sages);
                if (data.gpu) setGpuInfo(data.gpu);
                if (data.training_active !== undefined) setTrainingActive(data.training_active);
                if (data.training_samples !== undefined) setTrainingSamples(data.training_samples);
                setLastRefresh(new Date());
            } catch {
                // Offline — use empty
            }
        }
    }, []);

    // Fetch on mount + interval
    useEffect(() => {
        fetchSages();
        const interval = setInterval(fetchSages, 30000);
        return () => clearInterval(interval);
    }, [fetchSages]);

    const toggleSection = useCallback((id: string) => {
        setExpandedSection(prev => prev === id ? "" : id);
    }, []);

    // Send message
    const handleSend = useCallback(async () => {
        if (!chatInput.trim() || isSending) return;
        const msg = chatInput.trim();
        setChatInput("");
        setChatMessages(prev => [...prev, { role: "user", content: msg }]);
        setIsSending(true);

        try {
            const result = await sendCouncilMessage(msg, {
                session_id: "council-panel",
                previous_messages: chatMessages.slice(-5).map(m => ({
                    role: m.role, content: m.content
                })),
            });

            setChatMessages(prev => [...prev, {
                role: "assistant",
                content: result.response,
                source: result.response_source || result.source || "council",
                confidence: result.confidence,
                processing_time_ms: result.processing_time_ms,
                wise_man: result.wise_man,
            }]);
        } catch (e: any) {
            setChatMessages(prev => [...prev, {
                role: "assistant",
                content: `❌ خطأ: ${e.message}`,
                source: "error",
            }]);
        } finally {
            setIsSending(false);
            setTimeout(() => inputRef.current?.focus(), 50);
        }
    }, [chatInput, isSending, chatMessages]);

    const gpuTemp = gpuInfo.temperature;
    const gpuVram = gpuInfo.memory_used_mb && gpuInfo.memory_total_mb
        ? `${(gpuInfo.memory_used_mb / 1024).toFixed(1)}/${(gpuInfo.memory_total_mb / 1024).toFixed(1)}GB`
        : null;

    return (
        <div className="h-full flex flex-col bg-dark-900">
            {/* Header with GPU Status */}
            <div className="px-3 py-2 border-b border-dark-700">
                <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                        <Crown className="w-4 h-4 text-yellow-400" />
                        <span className="text-sm font-bold text-dark-100">مجلس الحكماء</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                        <button onClick={fetchSages} className="p-1 hover:bg-dark-800 rounded">
                            <RefreshCw className="w-3 h-3 text-dark-400" />
                        </button>
                    </div>
                </div>
                {/* GPU Status Bar */}
                {(gpuTemp || gpuVram) && (
                    <div className="flex items-center gap-3 text-[9px] text-dark-400">
                        {gpuTemp !== undefined && (
                            <span className="flex items-center gap-1">
                                <Thermometer className="w-3 h-3" />
                                <span className={gpuTemp > 80 ? "text-red-400" : gpuTemp > 60 ? "text-yellow-400" : "text-green-400"}>
                                    {gpuTemp}°C
                                </span>
                            </span>
                        )}
                        {gpuVram && (
                            <span className="flex items-center gap-1">
                                <HardDrive className="w-3 h-3" />
                                {gpuVram}
                            </span>
                        )}
                        {trainingActive && (
                            <span className="flex items-center gap-1 text-cyan-400">
                                <Activity className="w-3 h-3 animate-pulse" />
                                تدريب ({trainingSamples})
                            </span>
                        )}
                    </div>
                )}
            </div>

            <div className="flex-1 overflow-auto">
                {/* Chat */}
                <Section id="chat" title="محادثة المجلس" icon={MessageSquare}
                    expanded={expandedSection === "chat"} onToggle={toggleSection}>
                    <div className="space-y-1.5 max-h-80 overflow-auto mb-2">
                        {chatMessages.length === 0 && (
                            <div className="text-center text-dark-500 text-[11px] py-4">
                                أرسل رسالة للمجلس...
                            </div>
                        )}
                        {chatMessages.map((msg, i) => (
                            <div key={i} className={`text-[11px] p-2 rounded ${msg.role === "user"
                                ? "bg-primary-700/30 text-primary-100"
                                : msg.source === "error"
                                    ? "bg-red-500/20 text-red-300"
                                    : "bg-dark-800 text-dark-200"
                                }`}>
                                {/* Response metadata */}
                                {msg.role === "assistant" && msg.source && msg.source !== "error" && (
                                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                                        {msg.wise_man && (
                                            <span className="text-[9px] px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-300">
                                                👳 {msg.wise_man}
                                            </span>
                                        )}
                                        <span className={`text-[9px] px-1.5 py-0.5 rounded ${msg.source?.includes("lora") ? "bg-green-500/20 text-green-300"
                                                : msg.source?.includes("base") ? "bg-blue-500/20 text-blue-300"
                                                    : msg.source?.includes("unavailable") ? "bg-red-500/20 text-red-300"
                                                        : "bg-dark-700 text-dark-400"
                                            }`}>
                                            {msg.source?.includes("lora") ? "🧠 LoRA"
                                                : msg.source?.includes("base") ? "📦 Base"
                                                    : msg.source?.includes("unavailable") ? "⚠️ غير متاح"
                                                        : msg.source}
                                        </span>
                                        {msg.confidence !== undefined && msg.confidence > 0 && (
                                            <span className="text-[9px] px-1.5 py-0.5 rounded bg-dark-700 text-dark-300">
                                                {Math.round(msg.confidence * 100)}%
                                            </span>
                                        )}
                                        {msg.processing_time_ms !== undefined && (
                                            <span className="text-[9px] text-dark-500">
                                                ⏱ {(msg.processing_time_ms / 1000).toFixed(1)}s
                                            </span>
                                        )}
                                    </div>
                                )}
                                <span className="whitespace-pre-wrap" dir="rtl">{msg.content}</span>
                            </div>
                        ))}
                        {isSending && (
                            <div className="text-[11px] p-2 rounded bg-dark-800 text-dark-400">
                                <span className="inline-flex gap-1">
                                    <span className="w-1.5 h-1.5 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                                    <span className="w-1.5 h-1.5 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                                    <span className="w-1.5 h-1.5 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                                </span>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                    <div className="flex gap-1">
                        <input
                            ref={inputRef}
                            value={chatInput}
                            onChange={e => setChatInput(e.target.value)}
                            onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); handleSend(); } }}
                            placeholder="أرسل أمر للمجلس..."
                            className="flex-1 px-2 py-1.5 bg-dark-800 border border-dark-700 rounded text-xs text-dark-100 placeholder-dark-500 focus:outline-none focus:border-primary-500"
                            dir="rtl"
                            disabled={isSending}
                        />
                        <button onClick={handleSend} disabled={isSending || !chatInput.trim()}
                            className="px-2 py-1.5 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 rounded text-white transition-colors">
                            <Send className="w-3 h-3" />
                        </button>
                    </div>
                </Section>

                {/* Wise Men */}
                <Section id="wise-men" title="الحكماء" icon={Users}
                    badge={`${wiseMen.length}`}
                    expanded={expandedSection === "wise-men"} onToggle={toggleSection}>
                    <div className="grid grid-cols-2 gap-1">
                        {wiseMen.map((wm, i) => {
                            const Icon = WISE_ICONS[wm.id] || Brain;
                            const statusColor = STATUS_COLORS[wm.status] || "text-dark-400";
                            return (
                                <div key={wm.id || i}
                                    className="flex items-center gap-1.5 p-1.5 rounded bg-dark-800/60 hover:bg-dark-700/60 transition-colors cursor-pointer"
                                    onClick={() => {
                                        setChatInput(`يا ${wm.name}، `);
                                        setExpandedSection("chat");
                                        setTimeout(() => inputRef.current?.focus(), 100);
                                    }}
                                >
                                    <div className="relative flex-shrink-0">
                                        <Icon className={`w-3.5 h-3.5 ${statusColor}`} />
                                        <span className={`absolute -bottom-0.5 -right-0.5 w-1.5 h-1.5 rounded-full ${wm.status === "active" ? "bg-green-400" : "bg-dark-500"
                                            }`} />
                                    </div>
                                    <div className="min-w-0">
                                        <div className="text-[10px] font-medium text-dark-200 truncate">{wm.name}</div>
                                        <div className="text-[9px] text-dark-500 truncate">{wm.specialization || wm.role || ""}</div>
                                    </div>
                                </div>
                            );
                        })}
                        {wiseMen.length === 0 && (
                            <div className="col-span-2 text-center text-dark-500 text-xs py-2">
                                جاري الاتصال بالحكماء...
                            </div>
                        )}
                    </div>
                </Section>

                {/* Quick Actions */}
                <Section id="actions" title="إجراءات سريعة" icon={Zap}
                    expanded={expandedSection === "actions"} onToggle={toggleSection}>
                    <div className="grid grid-cols-2 gap-1">
                        {[
                            { label: "🔮 حكمة اليوم", prompt: "أعطني حكمة اليوم" },
                            { label: "🛡️ حالة الأمان", prompt: "ما حالة أمان النظام؟" },
                            { label: "📊 تقرير الأداء", prompt: "أعطني تقرير شامل عن أداء المنظومة" },
                            { label: "💻 حالة التدريب", prompt: "ما حالة التدريب الحالي وكم عيّنة متدربة؟" },
                            { label: "🧬 فحص الذكاء", prompt: "اختبر ذكائك: ما هو 15 × 17؟" },
                            { label: "📜 من أنت؟", prompt: "عرّف نفسك: من أنت وما دورك في المجلس؟" },
                        ].map(({ label, prompt }, i) => (
                            <button key={i}
                                onClick={() => { setChatInput(prompt); setExpandedSection("chat"); setTimeout(() => inputRef.current?.focus(), 100); }}
                                className="text-[10px] py-1.5 px-2 bg-dark-800 hover:bg-dark-700 rounded text-dark-300 transition-colors text-right">
                                {label}
                            </button>
                        ))}
                    </div>
                </Section>
            </div>
        </div>
    );
}
