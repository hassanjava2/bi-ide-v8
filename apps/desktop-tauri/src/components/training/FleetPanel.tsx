import { useState, useEffect, useCallback } from "react";
import {
    Wifi,
    WifiOff,
    Cpu,
    HardDrive,
    RefreshCw,
    Plus,
    Trash2,
    Edit2,
    Check,
    X,
    Server,
    Monitor,
    Smartphone,
} from "lucide-react";

interface WorkerInfo {
    name: string;
    host: string;
    user: string;
    gpus: number;
    vram_gb: number;
    model: string;
    role: string;
    max_concurrent: number;
    cpu_percent: number;
    status?: "online" | "offline" | "training" | "unknown";
}

export function FleetPanel() {
    const [mode, setMode] = useState<"online" | "offline">("online");
    const [workers, setWorkers] = useState<WorkerInfo[]>([
        {
            name: "rtx-5090",
            host: "100.104.35.44",
            user: "bi",
            gpus: 1,
            vram_gb: 24,
            model: "RTX 5090",
            role: "primary",
            max_concurrent: 2,
            cpu_percent: 30,
            status: "training",
        },
        {
            name: "vps-hostinger",
            host: "bi-iq.com",
            user: "root",
            gpus: 0,
            vram_gb: 0,
            model: "CPU",
            role: "training",
            max_concurrent: 1,
            cpu_percent: 75,
            status: "online",
        },
        {
            name: "windows-desktop",
            host: "100.76.169.110",
            user: "bi",
            gpus: 1,
            vram_gb: 6,
            model: "RTX 4050",
            role: "training",
            max_concurrent: 1,
            cpu_percent: 50,
            status: "training",
        },
    ]);
    const [editingWorker, setEditingWorker] = useState<string | null>(null);
    const [editHost, setEditHost] = useState("");
    const [isRefreshing, setIsRefreshing] = useState(false);

    // توابع LAN IPs المعروفة
    const lanIPs: Record<string, string> = {
        "rtx-5090": "192.168.1.164",
        "windows-desktop": "192.168.1.100",
        "vps-hostinger": "192.168.1.200",
    };

    const tailscaleIPs: Record<string, string> = {
        "rtx-5090": "100.104.35.44",
        "windows-desktop": "100.76.169.110",
        "vps-hostinger": "bi-iq.com",
    };

    const toggleMode = useCallback(() => {
        const newMode = mode === "online" ? "offline" : "online";
        setMode(newMode);
        // تبديل IPs
        setWorkers(prev =>
            prev.map(w => ({
                ...w,
                host: newMode === "offline"
                    ? (lanIPs[w.name] || w.host)
                    : (tailscaleIPs[w.name] || w.host),
            }))
        );
    }, [mode]);

    const startEditHost = (name: string, currentHost: string) => {
        setEditingWorker(name);
        setEditHost(currentHost);
    };

    const saveEditHost = (name: string) => {
        setWorkers(prev =>
            prev.map(w => w.name === name ? { ...w, host: editHost } : w)
        );
        setEditingWorker(null);
    };

    const refreshStatus = async () => {
        setIsRefreshing(true);
        // محاكاة فحص الاتصال
        await new Promise(r => setTimeout(r, 1500));
        setIsRefreshing(false);
    };

    const getStatusColor = (status?: string) => {
        switch (status) {
            case "online": return "bg-green-400";
            case "training": return "bg-blue-400 animate-pulse";
            case "offline": return "bg-red-400";
            default: return "bg-gray-500";
        }
    };

    const getStatusText = (status?: string) => {
        switch (status) {
            case "online": return "متصل";
            case "training": return "يدرّب";
            case "offline": return "غير متصل";
            default: return "غير معروف";
        }
    };

    const getWorkerIcon = (worker: WorkerInfo) => {
        if (worker.model.includes("RTX 5090")) return <Server className="w-4 h-4" />;
        if (worker.model === "CPU") return <Cpu className="w-4 h-4" />;
        if (worker.model.includes("4050")) return <Monitor className="w-4 h-4" />;
        return <HardDrive className="w-4 h-4" />;
    };

    const totalVRAM = workers.reduce((sum, w) => sum + w.vram_gb, 0);
    const totalGPUs = workers.reduce((sum, w) => sum + w.gpus, 0);
    const activeWorkers = workers.filter(w => w.status === "training" || w.status === "online").length;

    return (
        <div className="h-full flex flex-col overflow-auto text-sm">
            {/* Header + Mode Toggle */}
            <div className="p-3 border-b border-dark-700">
                <div className="flex items-center justify-between mb-3">
                    <h3 className="text-xs font-semibold text-dark-200 uppercase tracking-wider">
                        Training Fleet
                    </h3>
                    <button
                        onClick={refreshStatus}
                        className="p-1 text-dark-400 hover:text-dark-200 transition-colors"
                        title="Refresh status"
                    >
                        <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? "animate-spin" : ""}`} />
                    </button>
                </div>

                {/* Online/Offline Toggle */}
                <button
                    onClick={toggleMode}
                    className={`w-full flex items-center justify-between px-3 py-2 rounded-lg 
            transition-all duration-300 ${mode === "online"
                            ? "bg-gradient-to-r from-green-900/40 to-emerald-900/40 border border-green-700/50"
                            : "bg-gradient-to-r from-orange-900/40 to-amber-900/40 border border-orange-700/50"
                        }`}
                >
                    <div className="flex items-center gap-2">
                        {mode === "online" ? (
                            <Wifi className="w-4 h-4 text-green-400" />
                        ) : (
                            <WifiOff className="w-4 h-4 text-orange-400" />
                        )}
                        <span className={`text-xs font-medium ${mode === "online" ? "text-green-300" : "text-orange-300"
                            }`}>
                            {mode === "online" ? "Online Mode" : "Offline Mode (LAN)"}
                        </span>
                    </div>
                    <div className={`w-10 h-5 rounded-full relative transition-colors duration-300 ${mode === "online" ? "bg-green-600" : "bg-orange-600"
                        }`}>
                        <div className={`absolute w-4 h-4 rounded-full bg-white top-0.5 
              transition-transform duration-300 ${mode === "online" ? "left-5" : "left-0.5"
                            }`} />
                    </div>
                </button>
            </div>

            {/* Stats Bar */}
            <div className="px-3 py-2 border-b border-dark-700 grid grid-cols-3 gap-2">
                <div className="text-center">
                    <div className="text-lg font-bold text-primary-400">{activeWorkers}</div>
                    <div className="text-[10px] text-dark-500">Workers</div>
                </div>
                <div className="text-center">
                    <div className="text-lg font-bold text-blue-400">{totalGPUs}</div>
                    <div className="text-[10px] text-dark-500">GPUs</div>
                </div>
                <div className="text-center">
                    <div className="text-lg font-bold text-purple-400">{totalVRAM}GB</div>
                    <div className="text-[10px] text-dark-500">VRAM</div>
                </div>
            </div>

            {/* Workers List */}
            <div className="flex-1 overflow-auto p-2 space-y-2">
                {workers.map(worker => (
                    <div
                        key={worker.name}
                        className="bg-dark-800 rounded-lg border border-dark-700 p-3
              hover:border-dark-600 transition-colors"
                    >
                        {/* Worker Header */}
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                                <div className="text-dark-300">
                                    {getWorkerIcon(worker)}
                                </div>
                                <span className="text-xs font-medium text-dark-200">
                                    {worker.name}
                                </span>
                                <div className={`w-2 h-2 rounded-full ${getStatusColor(worker.status)}`} />
                            </div>
                            <span className="text-[10px] text-dark-500">
                                {getStatusText(worker.status)}
                            </span>
                        </div>

                        {/* Worker Details */}
                        <div className="space-y-1.5">
                            {/* IP/Host */}
                            <div className="flex items-center justify-between">
                                <span className="text-[10px] text-dark-500">Host:</span>
                                {editingWorker === worker.name ? (
                                    <div className="flex items-center gap-1">
                                        <input
                                            value={editHost}
                                            onChange={(e) => setEditHost(e.target.value)}
                                            className="bg-dark-900 text-[10px] text-dark-200 px-1.5 py-0.5 
                        rounded border border-dark-600 w-28 focus:border-primary-500 
                        focus:outline-none"
                                            autoFocus
                                        />
                                        <button
                                            onClick={() => saveEditHost(worker.name)}
                                            className="text-green-400 hover:text-green-300"
                                        >
                                            <Check className="w-3 h-3" />
                                        </button>
                                        <button
                                            onClick={() => setEditingWorker(null)}
                                            className="text-red-400 hover:text-red-300"
                                        >
                                            <X className="w-3 h-3" />
                                        </button>
                                    </div>
                                ) : (
                                    <div className="flex items-center gap-1">
                                        <code className="text-[10px] text-primary-400 bg-dark-900 
                      px-1.5 py-0.5 rounded font-mono">
                                            {worker.host}
                                        </code>
                                        <button
                                            onClick={() => startEditHost(worker.name, worker.host)}
                                            className="text-dark-500 hover:text-dark-300"
                                        >
                                            <Edit2 className="w-2.5 h-2.5" />
                                        </button>
                                    </div>
                                )}
                            </div>

                            {/* GPU */}
                            <div className="flex items-center justify-between">
                                <span className="text-[10px] text-dark-500">GPU:</span>
                                <span className="text-[10px] text-dark-300">
                                    {worker.gpus > 0 ? `${worker.model} (${worker.vram_gb}GB)` : "CPU Only"}
                                </span>
                            </div>

                            {/* CPU % */}
                            <div className="flex items-center justify-between">
                                <span className="text-[10px] text-dark-500">CPU:</span>
                                <div className="flex items-center gap-1.5">
                                    <div className="w-16 h-1.5 bg-dark-700 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-gradient-to-r from-primary-500 to-blue-500 rounded-full"
                                            style={{ width: `${worker.cpu_percent}%` }}
                                        />
                                    </div>
                                    <span className="text-[10px] text-dark-400">{worker.cpu_percent}%</span>
                                </div>
                            </div>

                            {/* Concurrent */}
                            <div className="flex items-center justify-between">
                                <span className="text-[10px] text-dark-500">Slots:</span>
                                <span className="text-[10px] text-dark-300">
                                    {worker.max_concurrent} concurrent
                                </span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Footer */}
            <div className="p-3 border-t border-dark-700">
                <div className="text-[10px] text-dark-500 text-center">
                    Install on any machine:
                    <code className="block mt-1 text-primary-400 bg-dark-800 px-2 py-1 rounded">
                        curl bi-iq.com/install-worker.sh | bash
                    </code>
                </div>
            </div>
        </div>
    );
}
