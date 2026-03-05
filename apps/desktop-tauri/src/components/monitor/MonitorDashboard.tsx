import { useEffect, useState, useCallback } from "react";
import {
    Monitor,
    Cpu,
    MemoryStick,
    HardDrive,
    Thermometer,
    Activity,
    Wifi,
    WifiOff,
    Server,
    RefreshCw,
    ChevronDown,
    ChevronRight,
    Zap,
    Download,
    Brain,
} from "lucide-react";

interface MachineStats {
    machine_id: string;
    hostname: string;
    platform: string;
    cpu_percent: number;
    ram_percent: number;
    ram_used_gb: number;
    ram_total_gb: number;
    gpu_name?: string;
    gpu_temp?: number;
    gpu_util?: number;
    gpu_mem_used_mb?: number;
    gpu_mem_total_mb?: number;
    disk_percent: number;
    disk_used_gb: number;
    disk_total_gb: number;
    temperature?: number;
    uptime?: string;
    services?: Record<string, string>;
    training_status?: string;
    training_samples?: number;
    download_status?: string;
    last_heartbeat?: string;
    is_online?: boolean;
}

interface MonitorState {
    machines: MachineStats[];
    source: "vps" | "lan" | "offline";
    lastUpdate: string;
    error?: string;
}

// API endpoints — try VPS first, fallback to LAN
const VPS_API = "https://bi-iq.com/api/v1/machines";
const LAN_API_RTX = "http://192.168.1.164:8090/api/v1/monitor";
const TAILSCALE_API_RTX = "http://100.104.35.44:8090/api/v1/monitor";

// New router endpoints for enriched data
const RTX_BRAIN_API = "http://100.104.35.44:8090/api/v1/brain/status";
const RTX_HEALTH_API = "http://100.104.35.44:8090/api/v1/rtx5090/health";

async function fetchWithTimeout(url: string, timeout = 5000): Promise<Response> {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    try {
        const res = await fetch(url, { signal: controller.signal });
        return res;
    } finally {
        clearTimeout(id);
    }
}

export function MonitorDashboard() {
    const [state, setState] = useState<MonitorState>({
        machines: [],
        source: "offline",
        lastUpdate: "",
    });
    const [isRefreshing, setIsRefreshing] = useState(false);
    const [expandedMachines, setExpandedMachines] = useState<Set<string>>(new Set(["all"]));
    const [autoRefresh, setAutoRefresh] = useState(true);

    const fetchMachines = useCallback(async () => {
        setIsRefreshing(true);
        const now = new Date().toLocaleTimeString("ar-IQ");

        // Try VPS first
        try {
            const res = await fetchWithTimeout(VPS_API, 4000);
            if (res.ok) {
                const data = await res.json();
                const machines = Array.isArray(data) ? data : data.machines || [];
                setState({ machines, source: "vps", lastUpdate: now });
                setIsRefreshing(false);
                return;
            }
        } catch { /* VPS unavailable */ }

        // Try Tailscale
        try {
            const res = await fetchWithTimeout(TAILSCALE_API_RTX, 3000);
            if (res.ok) {
                const data = await res.json();
                setState({
                    machines: Array.isArray(data) ? data : [data],
                    source: "lan",
                    lastUpdate: now,
                });
                setIsRefreshing(false);
                return;
            }
        } catch { /* Tailscale unavailable */ }

        // Try LAN
        try {
            const res = await fetchWithTimeout(LAN_API_RTX, 3000);
            if (res.ok) {
                const data = await res.json();
                setState({
                    machines: Array.isArray(data) ? data : [data],
                    source: "lan",
                    lastUpdate: now,
                });
                setIsRefreshing(false);
                return;
            }
        } catch { /* LAN unavailable */ }

        setState((prev) => ({
            ...prev,
            source: "offline",
            lastUpdate: now,
            error: "لا يمكن الاتصال بأي جهاز",
        }));
        setIsRefreshing(false);
    }, []);

    useEffect(() => {
        fetchMachines();
        if (!autoRefresh) return;
        const interval = setInterval(fetchMachines, 10000);
        return () => clearInterval(interval);
    }, [fetchMachines, autoRefresh]);

    const toggleMachine = (id: string) => {
        setExpandedMachines((prev) => {
            const next = new Set(prev);
            next.has(id) ? next.delete(id) : next.add(id);
            return next;
        });
    };

    const sourceColor = {
        vps: "text-green-400",
        lan: "text-blue-400",
        offline: "text-red-400",
    };

    const sourceLabel = {
        vps: "VPS (bi-iq.com)",
        lan: "LAN / Tailscale",
        offline: "غير متصل",
    };

    return (
        <div className="h-full flex flex-col p-3 gap-3 overflow-auto">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm font-medium text-dark-200">
                    <Monitor className="w-4 h-4 text-cyan-400" />
                    مراقبة الأجهزة
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={fetchMachines}
                        disabled={isRefreshing}
                        className="p-1 hover:bg-dark-800 rounded transition-colors"
                        title="تحديث"
                    >
                        <RefreshCw className={`w-3.5 h-3.5 text-dark-400 ${isRefreshing ? "animate-spin" : ""}`} />
                    </button>
                    <label className="flex items-center gap-1 text-xs text-dark-500 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={autoRefresh}
                            onChange={(e) => setAutoRefresh(e.target.checked)}
                            className="w-3 h-3"
                        />
                        تلقائي
                    </label>
                </div>
            </div>

            {/* Connection Source Badge */}
            <div className="flex items-center justify-between bg-dark-800 rounded-lg px-3 py-2">
                <div className="flex items-center gap-2">
                    {state.source === "offline" ? (
                        <WifiOff className="w-3.5 h-3.5 text-red-400" />
                    ) : (
                        <Wifi className="w-3.5 h-3.5 text-green-400" />
                    )}
                    <span className={`text-xs font-medium ${sourceColor[state.source]}`}>
                        {sourceLabel[state.source]}
                    </span>
                </div>
                <span className="text-xs text-dark-500">{state.lastUpdate}</span>
            </div>

            {/* Error */}
            {state.error && state.machines.length === 0 && (
                <div className="bg-red-900/30 border border-red-800/50 rounded-lg p-3 text-xs text-red-300">
                    {state.error}
                </div>
            )}

            {/* Machines */}
            {state.machines.length === 0 && !state.error && (
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center text-dark-500 text-xs">
                        <Activity className="w-8 h-8 mx-auto mb-2 opacity-30" />
                        جاري الاتصال...
                    </div>
                </div>
            )}

            {state.machines.map((machine) => (
                <MachineCard
                    key={machine.machine_id || machine.hostname}
                    machine={machine}
                    isExpanded={expandedMachines.has(machine.machine_id) || expandedMachines.has("all")}
                    onToggle={() => toggleMachine(machine.machine_id)}
                />
            ))}
        </div>
    );
}

function MachineCard({
    machine,
    isExpanded,
    onToggle,
}: {
    machine: MachineStats;
    isExpanded: boolean;
    onToggle: () => void;
}) {
    const isOnline = machine.is_online !== false;
    const gpuTemp = machine.gpu_temp || machine.temperature || 0;

    // Color helpers
    const tempColor = (t: number) =>
        t > 85 ? "text-red-400" : t > 70 ? "text-orange-400" : t > 50 ? "text-yellow-400" : "text-green-400";
    const usageColor = (p: number) =>
        p > 90 ? "text-red-400" : p > 70 ? "text-orange-400" : p > 50 ? "text-yellow-400" : "text-cyan-400";
    const usageBg = (p: number) =>
        p > 90 ? "bg-red-500" : p > 70 ? "bg-orange-500" : p > 50 ? "bg-yellow-500" : "bg-cyan-500";

    return (
        <div className="bg-dark-800 rounded-lg border border-dark-700 overflow-hidden">
            {/* Machine Header */}
            <button
                onClick={onToggle}
                className="w-full flex items-center gap-2 px-3 py-2.5 hover:bg-dark-750 transition-colors"
            >
                {isExpanded ? (
                    <ChevronDown className="w-3.5 h-3.5 text-dark-400" />
                ) : (
                    <ChevronRight className="w-3.5 h-3.5 text-dark-400" />
                )}
                <Server className={`w-3.5 h-3.5 ${isOnline ? "text-green-400" : "text-red-400"}`} />
                <span className="text-xs font-medium text-dark-200 flex-1 text-left truncate">
                    {machine.hostname || machine.machine_id}
                </span>
                <span className={`w-2 h-2 rounded-full ${isOnline ? "bg-green-500 animate-pulse" : "bg-red-500"}`} />
                {machine.platform && (
                    <span className="text-xs text-dark-500">{machine.platform}</span>
                )}
            </button>

            {isExpanded && (
                <div className="px-3 pb-3 space-y-2">
                    {/* Usage Bars */}
                    <div className="grid grid-cols-2 gap-2">
                        {/* CPU */}
                        <UsageBar
                            icon={<Cpu className="w-3 h-3" />}
                            label="CPU"
                            value={machine.cpu_percent}
                            suffix="%"
                            color={usageColor(machine.cpu_percent)}
                            bgColor={usageBg(machine.cpu_percent)}
                        />
                        {/* RAM */}
                        <UsageBar
                            icon={<MemoryStick className="w-3 h-3" />}
                            label="RAM"
                            value={machine.ram_percent}
                            suffix={`% (${machine.ram_used_gb?.toFixed(1) || "?"}/${machine.ram_total_gb?.toFixed(0) || "?"}GB)`}
                            color={usageColor(machine.ram_percent)}
                            bgColor={usageBg(machine.ram_percent)}
                        />
                        {/* Disk */}
                        <UsageBar
                            icon={<HardDrive className="w-3 h-3" />}
                            label="Disk"
                            value={machine.disk_percent}
                            suffix={`% (${machine.disk_used_gb?.toFixed(0) || "?"}/${machine.disk_total_gb?.toFixed(0) || "?"}GB)`}
                            color={usageColor(machine.disk_percent)}
                            bgColor={usageBg(machine.disk_percent)}
                        />
                        {/* Temperature */}
                        {gpuTemp > 0 && (
                            <UsageBar
                                icon={<Thermometer className="w-3 h-3" />}
                                label="Temp"
                                value={Math.min(gpuTemp, 100)}
                                suffix={`°C`}
                                displayValue={gpuTemp}
                                color={tempColor(gpuTemp)}
                                bgColor={gpuTemp > 85 ? "bg-red-500" : gpuTemp > 70 ? "bg-orange-500" : "bg-green-500"}
                            />
                        )}
                    </div>

                    {/* GPU Section */}
                    {machine.gpu_name && (
                        <div className="bg-dark-900 rounded p-2 space-y-1.5">
                            <div className="flex items-center gap-1.5 text-xs">
                                <Zap className="w-3 h-3 text-green-400" />
                                <span className="text-dark-300 font-medium">{machine.gpu_name}</span>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                                <div className="text-xs text-dark-400">
                                    Util: <span className={usageColor(machine.gpu_util || 0)}>{machine.gpu_util || 0}%</span>
                                </div>
                                <div className="text-xs text-dark-400">
                                    VRAM: <span className="text-dark-300">
                                        {((machine.gpu_mem_used_mb || 0) / 1024).toFixed(1)}/{((machine.gpu_mem_total_mb || 0) / 1024).toFixed(0)}GB
                                    </span>
                                </div>
                            </div>
                            {(machine.gpu_util || 0) > 0 && (
                                <div className="w-full bg-dark-700 rounded-full h-1.5">
                                    <div
                                        className={`h-1.5 rounded-full transition-all duration-500 ${usageBg(machine.gpu_util || 0)}`}
                                        style={{ width: `${machine.gpu_util || 0}%` }}
                                    />
                                </div>
                            )}
                        </div>
                    )}

                    {/* Training & Download Status */}
                    <div className="flex gap-2">
                        {machine.training_status && (
                            <div className="flex-1 bg-dark-900 rounded p-2">
                                <div className="flex items-center gap-1.5 text-xs">
                                    <Brain className="w-3 h-3 text-purple-400" />
                                    <span className="text-dark-400">تدريب</span>
                                </div>
                                <div className="text-xs text-dark-300 mt-1 truncate">{machine.training_status}</div>
                                {machine.training_samples != null && (
                                    <div className="text-xs text-dark-500 mt-0.5">{machine.training_samples} sample</div>
                                )}
                            </div>
                        )}
                        {machine.download_status && (
                            <div className="flex-1 bg-dark-900 rounded p-2">
                                <div className="flex items-center gap-1.5 text-xs">
                                    <Download className="w-3 h-3 text-blue-400" />
                                    <span className="text-dark-400">تنزيل</span>
                                </div>
                                <div className="text-xs text-dark-300 mt-1 truncate">{machine.download_status}</div>
                            </div>
                        )}
                    </div>

                    {/* Services */}
                    {machine.services && Object.keys(machine.services).length > 0 && (
                        <div className="bg-dark-900 rounded p-2">
                            <div className="text-xs text-dark-400 mb-1">خدمات</div>
                            <div className="grid grid-cols-2 gap-1">
                                {Object.entries(machine.services).map(([name, status]) => (
                                    <div key={name} className="flex items-center gap-1 text-xs">
                                        <span className={`w-1.5 h-1.5 rounded-full ${status === "active" ? "bg-green-500" : status === "running" ? "bg-green-500" : "bg-red-500"
                                            }`} />
                                        <span className="text-dark-400 truncate">{name.replace("bi-", "")}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Last Heartbeat */}
                    {machine.last_heartbeat && (
                        <div className="text-xs text-dark-600 text-right">
                            آخر heartbeat: {new Date(machine.last_heartbeat).toLocaleTimeString("ar-IQ")}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

function UsageBar({
    icon,
    label,
    value,
    suffix,
    displayValue,
    color,
    bgColor,
}: {
    icon: React.ReactNode;
    label: string;
    value: number;
    suffix: string;
    displayValue?: number;
    color: string;
    bgColor: string;
}) {
    return (
        <div className="space-y-1">
            <div className="flex items-center justify-between">
                <div className={`flex items-center gap-1 text-xs ${color}`}>
                    {icon}
                    <span className="text-dark-400">{label}</span>
                </div>
                <span className={`text-xs font-mono ${color}`}>
                    {displayValue != null ? displayValue : value}{suffix}
                </span>
            </div>
            <div className="w-full bg-dark-700 rounded-full h-1.5">
                <div
                    className={`h-1.5 rounded-full transition-all duration-500 ${bgColor}`}
                    style={{ width: `${Math.min(value, 100)}%` }}
                />
            </div>
        </div>
    );
}
