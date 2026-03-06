import { useState, useCallback, useRef } from "react";
import {
    FolderKanban, Send, Loader2, CheckCircle2, XCircle,
    Clock, Brain, ChevronDown, ChevronRight, Sparkles,
    Code, Database, Shield, Globe, Package, Cpu,
} from "lucide-react";
import { sendBrainProject } from "../config/api";

// ═══ Types ═══

interface ProjectTask {
    capsule_id: string;
    task: string;
    status: "pending" | "running" | "done" | "error";
    result?: string;
}

interface Project {
    id: string;
    command: string;
    status: "analyzing" | "running" | "done" | "error";
    tasks: ProjectTask[];
    progress: number;
    created_at: string;
    result?: any;
    error?: string;
}

// ═══ Capsule Icons ═══

const CAPSULE_ICONS: Record<string, any> = {
    code_python: Code, code_typescript: Code, code_rust: Code,
    code_css: Code, code_web: Globe, code_sql: Database,
    code_debugging: Code, code_testing: Shield,
    erp_accounting: Package, erp_hr: Package, erp_inventory: Package,
    erp_purchasing: Package, erp_sales: Package,
    devops: Cpu, security: Shield, database_design: Database,
};

const STATUS_BADGE: Record<string, { color: string; icon: any; label: string }> = {
    analyzing: { color: "bg-yellow-500/20 text-yellow-300", icon: Sparkles, label: "تحليل..." },
    running: { color: "bg-cyan-500/20 text-cyan-300", icon: Loader2, label: "قيد التنفيذ" },
    done: { color: "bg-green-500/20 text-green-300", icon: CheckCircle2, label: "مكتمل" },
    error: { color: "bg-red-500/20 text-red-300", icon: XCircle, label: "خطأ" },
    pending: { color: "bg-dark-700 text-dark-400", icon: Clock, label: "بانتظار" },
};

// ═══ Quick Templates ═══

const TEMPLATES = [
    { label: "🏗️ نظام ERP", command: "سوولي نظام ERP كامل" },
    { label: "🌐 موقع ويب", command: "سوولي موقع ويب متكامل" },
    { label: "🔒 نظام أمان", command: "سوولي نظام أمان وحماية" },
    { label: "📊 لوحة تحكم", command: "سوولي لوحة تحكم إدارية" },
    { label: "🤖 بوت ذكي", command: "سوولي بوت ذكاء اصطناعي" },
    { label: "📱 تطبيق موبايل", command: "سوولي تطبيق موبايل" },
];

// ═══ Component ═══

export function ProjectsPanel() {
    const [command, setCommand] = useState("");
    const [projects, setProjects] = useState<Project[]>([]);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [expandedProject, setExpandedProject] = useState<string | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    // إرسال مشروع جديد
    const handleSubmit = useCallback(async () => {
        if (!command.trim() || isSubmitting) return;
        const cmd = command.trim();
        setCommand("");
        setIsSubmitting(true);

        const newProject: Project = {
            id: `proj-${Date.now()}`,
            command: cmd,
            status: "analyzing",
            tasks: [],
            progress: 0,
            created_at: new Date().toISOString(),
        };

        setProjects(prev => [newProject, ...prev]);
        setExpandedProject(newProject.id);

        try {
            const result = await sendBrainProject(cmd);

            // تحديث المشروع بالنتائج
            setProjects(prev => prev.map(p => {
                if (p.id !== newProject.id) return p;
                const tasks: ProjectTask[] = (result?.tasks || []).map((t: any) => ({
                    capsule_id: t.capsule_id || t.capsule || "unknown",
                    task: t.task || t.description || cmd,
                    status: t.status || "done",
                    result: t.result || t.output,
                }));
                return {
                    ...p,
                    status: result?.error ? "error" : "done",
                    tasks,
                    progress: 100,
                    result,
                    error: result?.error,
                };
            }));
        } catch (e: any) {
            setProjects(prev => prev.map(p =>
                p.id === newProject.id
                    ? { ...p, status: "error" as const, error: e.message || String(e), progress: 0 }
                    : p
            ));
        } finally {
            setIsSubmitting(false);
        }
    }, [command, isSubmitting]);

    return (
        <div className="h-full flex flex-col bg-dark-900">
            {/* Header */}
            <div className="px-3 py-2 border-b border-dark-700">
                <div className="flex items-center gap-2 mb-2">
                    <FolderKanban className="w-4 h-4 text-cyan-400" />
                    <span className="text-sm font-bold text-dark-100">المشاريع</span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-cyan-600/30 text-cyan-300 font-mono">
                        {projects.length}
                    </span>
                </div>

                {/* Command Input */}
                <div className="flex gap-1">
                    <input
                        ref={inputRef}
                        value={command}
                        onChange={e => setCommand(e.target.value)}
                        onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); handleSubmit(); } }}
                        placeholder="سوولي نظام ERP كامل..."
                        className="flex-1 px-2 py-1.5 bg-dark-800 border border-dark-700 rounded text-xs text-dark-100 placeholder-dark-500 focus:outline-none focus:border-cyan-500"
                        dir="rtl"
                        disabled={isSubmitting}
                    />
                    <button
                        onClick={handleSubmit}
                        disabled={isSubmitting || !command.trim()}
                        className="px-2 py-1.5 bg-cyan-600 hover:bg-cyan-700 disabled:opacity-50 rounded text-white transition-colors"
                    >
                        {isSubmitting
                            ? <Loader2 className="w-3 h-3 animate-spin" />
                            : <Send className="w-3 h-3" />
                        }
                    </button>
                </div>
            </div>

            {/* Quick Templates */}
            <div className="px-3 py-2 border-b border-dark-700/50">
                <div className="grid grid-cols-2 gap-1">
                    {TEMPLATES.map(({ label, command: cmd }, i) => (
                        <button
                            key={i}
                            onClick={() => {
                                setCommand(cmd);
                                setTimeout(() => inputRef.current?.focus(), 50);
                            }}
                            className="text-[10px] py-1.5 px-2 bg-dark-800 hover:bg-dark-700 rounded text-dark-300 transition-colors text-right"
                        >
                            {label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Projects List */}
            <div className="flex-1 overflow-auto px-2 py-2 space-y-2">
                {projects.length === 0 && (
                    <div className="text-center text-dark-500 text-[11px] py-8">
                        <FolderKanban className="w-8 h-8 mx-auto mb-2 text-dark-700" />
                        <div>أرسل أمر لبدء مشروع جديد</div>
                        <div className="text-dark-600 mt-1">مثلاً: "سوولي نظام ERP كامل"</div>
                    </div>
                )}

                {projects.map(project => {
                    const badge = STATUS_BADGE[project.status] || STATUS_BADGE.pending;
                    const BadgeIcon = badge.icon;
                    const isExpanded = expandedProject === project.id;
                    const doneTasks = project.tasks.filter(t => t.status === "done").length;

                    return (
                        <div key={project.id} className="rounded-lg border border-dark-700/50 bg-dark-800/40 overflow-hidden">
                            {/* Project Header */}
                            <button
                                onClick={() => setExpandedProject(isExpanded ? null : project.id)}
                                className="w-full px-3 py-2 flex items-center gap-2 hover:bg-dark-800/60 transition-colors"
                            >
                                {isExpanded
                                    ? <ChevronDown className="w-3 h-3 text-dark-400 flex-shrink-0" />
                                    : <ChevronRight className="w-3 h-3 text-dark-400 flex-shrink-0" />
                                }
                                <div className="flex-1 text-right min-w-0">
                                    <div className="text-[11px] font-medium text-dark-200 truncate" dir="rtl">
                                        {project.command}
                                    </div>
                                    <div className="flex items-center gap-2 mt-0.5">
                                        <span className={`text-[9px] px-1.5 py-0.5 rounded flex items-center gap-1 ${badge.color}`}>
                                            <BadgeIcon className={`w-2.5 h-2.5 ${project.status === "running" ? "animate-spin" : ""}`} />
                                            {badge.label}
                                        </span>
                                        {project.tasks.length > 0 && (
                                            <span className="text-[9px] text-dark-500">
                                                {doneTasks}/{project.tasks.length} مهام
                                            </span>
                                        )}
                                    </div>
                                </div>
                            </button>

                            {/* Progress Bar */}
                            {project.status === "running" && (
                                <div className="px-3 pb-1">
                                    <div className="w-full h-1 bg-dark-700 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-gradient-to-r from-cyan-500 to-cyan-400 rounded-full transition-all duration-500"
                                            style={{ width: `${project.progress}%` }}
                                        />
                                    </div>
                                </div>
                            )}

                            {/* Expanded: Tasks */}
                            {isExpanded && (
                                <div className="px-3 pb-2 space-y-1">
                                    {project.error && (
                                        <div className="text-[10px] p-2 rounded bg-red-500/10 text-red-300 border border-red-500/20" dir="rtl">
                                            ❌ {project.error}
                                        </div>
                                    )}

                                    {project.tasks.map((task, ti) => {
                                        const TaskIcon = CAPSULE_ICONS[task.capsule_id] || Brain;
                                        const taskBadge = STATUS_BADGE[task.status] || STATUS_BADGE.pending;
                                        return (
                                            <div key={ti} className="flex items-start gap-2 p-1.5 rounded bg-dark-800/60">
                                                <TaskIcon className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0 mt-0.5" />
                                                <div className="flex-1 min-w-0">
                                                    <div className="flex items-center gap-1.5">
                                                        <span className="text-[9px] font-mono text-dark-400">
                                                            {task.capsule_id}
                                                        </span>
                                                        <span className={`text-[8px] px-1 py-0.5 rounded ${taskBadge.color}`}>
                                                            {taskBadge.label}
                                                        </span>
                                                    </div>
                                                    <div className="text-[10px] text-dark-300 mt-0.5 truncate" dir="rtl">
                                                        {task.task}
                                                    </div>
                                                    {task.result && (
                                                        <div className="text-[9px] text-dark-500 mt-0.5 line-clamp-2" dir="rtl">
                                                            {typeof task.result === "string"
                                                                ? task.result.slice(0, 200)
                                                                : JSON.stringify(task.result).slice(0, 200)
                                                            }
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        );
                                    })}

                                    {project.tasks.length === 0 && project.status !== "error" && (
                                        <div className="text-center text-[10px] text-dark-500 py-2">
                                            {project.status === "analyzing"
                                                ? "⏳ جاري تحليل الأمر..."
                                                : "لا توجد مهام"
                                            }
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
