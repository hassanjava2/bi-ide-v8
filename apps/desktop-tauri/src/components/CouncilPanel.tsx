import { useState, useEffect, useCallback, useRef } from "react";
import {
    Crown, Users, MessageSquare, Brain, Shield, Eye,
    Cpu, Zap, RefreshCw, Send, ChevronDown, ChevronRight,
    Activity, Globe, Database, Code, Lock, Layers,
} from "lucide-react";
import { sendCouncilMessage } from "../config/api";

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

// Section component — defined OUTSIDE to prevent re-mount on every keystroke
function Section({ id, title, icon: Icon, expanded, onToggle, children }: {
    id: string; title: string; icon: any; expanded: boolean;
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
                {title}
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

interface CouncilPanelProps {
    onSwitchToHierarchy?: () => void;
}

export function CouncilPanel({ onSwitchToHierarchy }: CouncilPanelProps) {
    const [wiseMen, setWiseMen] = useState<WiseMan[]>([]);
    const [orchHealth, setOrchHealth] = useState<any>(null);
    const [chatInput, setChatInput] = useState("");
    const [chatMessages, setChatMessages] = useState<Array<{ role: string; content: string; source?: string }>>([
        { role: "assistant", content: "🏛️ مرحباً بك في مجلس الحكماء. كلمني شتريد.", source: "council" },
    ]);
    const [isSending, setIsSending] = useState(false);
    const [loading, setLoading] = useState(false);
    const [expandedSection, setExpandedSection] = useState<string>("chat");
    const inputRef = useRef<HTMLInputElement>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Auto-scroll chat
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [chatMessages]);

    const toggleSection = useCallback((id: string) => {
        setExpandedSection(prev => prev === id ? "" : id);
    }, []);

    // Send message via HTTP API (RTX direct → VPS fallback)
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
                source: result.wise_man || result.source || "council",
            }]);
        } catch (e: any) {
            setChatMessages(prev => [...prev, {
                role: "assistant",
                content: `❌ خطأ: ${e.message}`,
                source: "error",
            }]);
        } finally {
            setIsSending(false);
            // Refocus input after send
            setTimeout(() => inputRef.current?.focus(), 50);
        }
    }, [chatInput, isSending, chatMessages]);

    return (
        <div className="h-full flex flex-col bg-dark-900">
            {/* Header */}
            <div className="flex items-center justify-between px-3 py-2 border-b border-dark-700">
                <div className="flex items-center gap-2">
                    <Crown className="w-4 h-4 text-yellow-400" />
                    <span className="text-sm font-bold text-dark-100">مجلس الحكماء</span>
                </div>
                <div className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                    <button onClick={() => setLoading(true)} className="p-1 hover:bg-dark-800 rounded">
                        <RefreshCw className={`w-3 h-3 text-dark-400 ${loading ? "animate-spin" : ""}`} />
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-auto">
                {/* Chat — Primary section */}
                <Section id="chat" title="محادثة المجلس" icon={MessageSquare}
                    expanded={expandedSection === "chat"} onToggle={toggleSection}>
                    <div className="space-y-1.5 max-h-64 overflow-auto mb-2">
                        {chatMessages.map((msg, i) => (
                            <div key={i} className={`text-[11px] p-2 rounded ${msg.role === "user"
                                ? "bg-primary-700/30 text-primary-100"
                                : msg.source === "error"
                                    ? "bg-red-500/20 text-red-300"
                                    : "bg-dark-800 text-dark-200"
                                }`}>
                                {msg.source && msg.source !== "error" && (
                                    <span className="text-[9px] text-dark-500 block mb-0.5">{msg.source}</span>
                                )}
                                <span className="whitespace-pre-wrap">{msg.content}</span>
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
                <Section id="wise-men" title={`الحكماء (${wiseMen.length})`} icon={Users}
                    expanded={expandedSection === "wise-men"} onToggle={toggleSection}>
                    <div className="grid grid-cols-2 gap-1">
                        {wiseMen.map((wm, i) => {
                            const Icon = WISE_ICONS[wm.id] || Brain;
                            const statusColor = STATUS_COLORS[wm.status] || "text-dark-400";
                            return (
                                <div key={wm.id || i} className="flex items-center gap-1.5 p-1.5 rounded bg-dark-800/60 hover:bg-dark-700/60 transition-colors cursor-default">
                                    <Icon className={`w-3.5 h-3.5 ${statusColor} flex-shrink-0`} />
                                    <div className="min-w-0">
                                        <div className="text-[10px] font-medium text-dark-200 truncate">{wm.name || wm.id}</div>
                                        <div className="text-[9px] text-dark-500 truncate">{wm.role || wm.specialization || ""}</div>
                                    </div>
                                </div>
                            );
                        })}
                        {wiseMen.length === 0 && (
                            <div className="col-span-2 text-center text-dark-500 text-xs py-2">
                                لا يوجد حكماء متصلين حالياً
                            </div>
                        )}
                    </div>
                </Section>

                {/* Quick Actions */}
                <Section id="actions" title="إجراءات سريعة" icon={Zap}
                    expanded={expandedSection === "actions"} onToggle={toggleSection}>
                    <div className="grid grid-cols-2 gap-1">
                        {[
                            { label: "🔮 حكمة", prompt: "أعطني حكمة اليوم" },
                            { label: "🛡️ أمان", prompt: "ما حالة الأمان؟" },
                            { label: "📊 مقاييس", prompt: "أعطني مقاييس الأداء" },
                            { label: "📜 سجل", prompt: "أعطني ملخص السجل" },
                        ].map(({ label, prompt }, i) => (
                            <button key={i}
                                onClick={() => { setChatInput(prompt); setExpandedSection("chat"); setTimeout(() => inputRef.current?.focus(), 100); }}
                                className="text-[10px] py-1.5 px-2 bg-dark-800 hover:bg-dark-700 rounded text-dark-300 transition-colors">
                                {label}
                            </button>
                        ))}
                    </div>
                </Section>
            </div>
        </div>
    );
}
