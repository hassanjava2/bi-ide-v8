import { useState, useEffect } from "react";
import { Eye, Scan, Brain, RefreshCw, Play, Wifi, WifiOff } from "lucide-react";
import { apiGet, apiPost } from "../lib/api-config";

export function VisionScoutPanel() {
    const [status, setStatus] = useState<any>(null);
    const [training, setTraining] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    const fetchAll = async () => {
        setLoading(true);
        try {
            const [st, tr] = await Promise.all([
                apiGet("/api/vision-scout/status").catch(() => null),
                apiGet("/api/vision-scout/training").catch(() => null),
            ]);
            setStatus(st);
            setTraining(tr);
        } finally { setLoading(false); }
    };

    useEffect(() => { fetchAll(); }, []);

    const runCycle = async () => {
        setLoading(true);
        try {
            await apiPost("/api/vision-scout/cycle", {});
            await fetchAll();
        } finally { setLoading(false); }
    };

    const trainStep = async () => {
        setLoading(true);
        try {
            await apiPost("/api/vision-scout/train", {});
            await fetchAll();
        } finally { setLoading(false); }
    };

    return (
        <div className="p-3">
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Scan className="w-4 h-4 text-emerald-400" />
                    <span className="text-sm font-bold text-dark-200">كشافة صورية 📷</span>
                </div>
                <button onClick={fetchAll} disabled={loading} className="p-1 hover:bg-dark-800 rounded">
                    <RefreshCw className={`w-3 h-3 text-dark-400 ${loading ? "animate-spin" : ""}`} />
                </button>
            </div>

            {/* Connection status */}
            {status && (
                <div className="flex items-center gap-2 mb-3 bg-dark-800 rounded p-2">
                    {status.online ? (
                        <><Wifi className="w-3 h-3 text-green-400" />
                            <span className="text-[10px] text-green-400 font-bold">Online — يجمع صور</span></>
                    ) : (
                        <><WifiOff className="w-3 h-3 text-red-400" />
                            <span className="text-[10px] text-red-400 font-bold">Offline — YOLO يعلّم</span></>
                    )}
                    <span className="text-[10px] text-dark-500 ml-auto">
                        YOLO: {status.yolo_available ? "✅" : "❌"}
                    </span>
                </div>
            )}

            {/* Scout stats */}
            {status?.scout && (
                <div className="mb-3">
                    <div className="flex items-center gap-1 mb-1">
                        <Eye className="w-3 h-3 text-emerald-400" />
                        <span className="text-xs font-medium text-dark-300">الكشافة</span>
                    </div>
                    <div className="bg-dark-800 rounded p-2">
                        <div className="flex justify-between text-[10px] text-dark-400 mb-1">
                            <span>محمّل: <span className="text-emerald-400 font-bold">{status.scout.total_downloaded}</span></span>
                            <span>فريده: <span className="text-blue-400 font-bold">{status.scout.unique_images}</span></span>
                        </div>
                        <button onClick={runCycle} disabled={loading}
                            className="w-full mt-1 py-1 bg-emerald-700 hover:bg-emerald-600 rounded text-[10px] font-medium text-white disabled:opacity-50">
                            🔍 دورة كشافة صورية
                        </button>
                    </div>
                </div>
            )}

            {/* Vision capsules */}
            {status?.capsules && (
                <div className="mb-3">
                    <div className="flex items-center gap-1 mb-1">
                        <Eye className="w-3 h-3 text-cyan-400" />
                        <span className="text-xs font-medium text-dark-300">كبسولات الرؤية (6)</span>
                    </div>
                    <div className="space-y-0.5">
                        {Object.entries(status.capsules).map(([key, cap]: [string, any]) => (
                            <div key={key} className="flex items-center justify-between bg-dark-800 rounded px-2 py-1">
                                <div className="flex items-center gap-1">
                                    <span className="text-[10px] text-dark-200">{cap.name_ar}</span>
                                    {cap.trained && <span className="text-[8px] text-green-400">✅</span>}
                                </div>
                                <span className="text-[10px] text-dark-500">
                                    {cap.data_count} <span className="text-dark-600">عيّنة</span>
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Training */}
            <div>
                <div className="flex items-center gap-1 mb-1">
                    <Brain className="w-3 h-3 text-purple-400" />
                    <span className="text-xs font-medium text-dark-300">تدريب BI-Vision</span>
                </div>
                <div className="bg-dark-800 rounded p-2">
                    {training ? (
                        <>
                            <div className="flex justify-between text-[10px] text-dark-400 mb-1">
                                <span>مستوى: <span className="text-purple-400 font-bold">L{training.current_level}</span></span>
                                <span>دورات: <span className="text-blue-400 font-bold">{training.total_epochs}</span></span>
                                <span>دقة: <span className="text-green-400 font-bold">{(training.best_accuracy * 100).toFixed(1)}%</span></span>
                            </div>
                            {/* Progress bar */}
                            <div className="w-full bg-dark-700 rounded-full h-1.5 mb-1">
                                <div className="bg-purple-500 h-1.5 rounded-full transition-all"
                                    style={{ width: `${training.best_accuracy * 100}%` }} />
                            </div>
                        </>
                    ) : (
                        <div className="text-[10px] text-dark-500">لم يبدأ التدريب</div>
                    )}
                    <button onClick={trainStep} disabled={loading}
                        className="w-full mt-1 py-1 bg-purple-700 hover:bg-purple-600 rounded text-[10px] font-medium text-white disabled:opacity-50">
                        <Play className="w-3 h-3 inline mr-1" /> خطوة تدريب
                    </button>
                </div>
            </div>
        </div>
    );
}
