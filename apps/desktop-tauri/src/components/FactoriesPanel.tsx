import { useState, useEffect } from "react";
import { Factory, Play, Ban, Plus, BarChart3, RefreshCw, ChevronDown, ChevronRight } from "lucide-react";
import { apiGet, apiPost } from "../lib/api-config";

interface FactorySummary {
    factories: number;
    products: string[];
    total_output_tons: number;
    total_workers: number;
    total_problems: number;
    total_solved: number;
    vetoed: string[];
    pending_proposals: number;
}

interface CatalogItem {
    name_ar: string;
    capacity: number;
    category: string;
}

export function FactoriesPanel() {
    const [summary, setSummary] = useState<FactorySummary | null>(null);
    const [catalog, setCatalog] = useState<Record<string, CatalogItem>>({});
    const [report, setReport] = useState("");
    const [loading, setLoading] = useState(false);
    const [showCatalog, setShowCatalog] = useState(false);
    const [simDays, setSimDays] = useState(30);

    const fetchSummary = async () => {
        try {
            setSummary(await apiGet("/api/factories"));
        } catch { }
    };

    const fetchCatalog = async () => {
        try {
            const data = await apiGet("/api/factories/catalog");
            setCatalog(data.catalog);
        } catch { }
    };

    const fetchReport = async () => {
        try {
            const data = await apiGet("/api/factories/report");
            setReport(data.report);
        } catch { }
    };

    useEffect(() => {
        fetchSummary();
        fetchCatalog();
    }, []);

    const createFactory = async (product: string) => {
        setLoading(true);
        try {
            await apiPost("/api/factories/create", { product });
            await fetchSummary();
            await fetchReport();
        } finally { setLoading(false); }
    };

    const createAll = async () => {
        setLoading(true);
        try {
            await apiPost("/api/factories/create-all", {});
            await fetchSummary();
            await fetchReport();
        } finally { setLoading(false); }
    };

    const simulate = async () => {
        setLoading(true);
        try {
            await apiPost("/api/factories/simulate?days=" + simDays, {});
            await fetchSummary();
            await fetchReport();
        } finally { setLoading(false); }
    };

    const veto = async (product: string) => {
        await apiPost("/api/factories/veto", { product, reason: "User veto from IDE" });
        await fetchSummary();
    };

    return (
        <div className="p-3">
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Factory className="w-4 h-4 text-orange-400" />
                    <span className="text-sm font-bold text-dark-200">المصانع 🏭</span>
                </div>
                <button onClick={fetchSummary} className="p-1 hover:bg-dark-800 rounded">
                    <RefreshCw className="w-3 h-3 text-dark-400" />
                </button>
            </div>

            {summary && (
                <div className="grid grid-cols-2 gap-2 mb-3">
                    <div className="bg-dark-800 rounded p-2 text-center">
                        <div className="text-lg font-bold text-orange-400">{summary.factories}</div>
                        <div className="text-[10px] text-dark-400">مصنع</div>
                    </div>
                    <div className="bg-dark-800 rounded p-2 text-center">
                        <div className="text-lg font-bold text-green-400">{summary.total_workers}</div>
                        <div className="text-[10px] text-dark-400">عامل AI</div>
                    </div>
                    <div className="bg-dark-800 rounded p-2 text-center">
                        <div className="text-sm font-bold text-blue-400">{summary.total_output_tons.toLocaleString()}</div>
                        <div className="text-[10px] text-dark-400">طن إنتاج</div>
                    </div>
                    <div className="bg-dark-800 rounded p-2 text-center">
                        <div className="text-sm font-bold text-purple-400">
                            {summary.total_problems > 0 ? Math.round(summary.total_solved / summary.total_problems * 100) : 0}%
                        </div>
                        <div className="text-[10px] text-dark-400">حلول</div>
                    </div>
                </div>
            )}

            {/* Actions */}
            <div className="flex gap-1 mb-3">
                <button onClick={createAll} disabled={loading}
                    className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 bg-orange-600 hover:bg-orange-500 rounded text-xs font-medium text-white disabled:opacity-50">
                    <Plus className="w-3 h-3" /> الكل
                </button>
                <button onClick={simulate} disabled={loading}
                    className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 bg-green-600 hover:bg-green-500 rounded text-xs font-medium text-white disabled:opacity-50">
                    <Play className="w-3 h-3" /> محاكاة {simDays}d
                </button>
                <button onClick={fetchReport} disabled={loading}
                    className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-xs font-medium text-white disabled:opacity-50">
                    <BarChart3 className="w-3 h-3" /> تقرير
                </button>
            </div>

            {/* Report */}
            {report && (
                <pre className="bg-dark-800 rounded p-2 text-[10px] text-dark-300 overflow-x-auto mb-3 whitespace-pre-wrap">{report}</pre>
            )}

            {/* Catalog */}
            <button onClick={() => setShowCatalog(!showCatalog)}
                className="flex items-center gap-1 text-xs text-dark-400 hover:text-dark-200 mb-2">
                {showCatalog ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                الكتالوج ({Object.keys(catalog).length} نوع)
            </button>

            {showCatalog && (
                <div className="space-y-1">
                    {Object.entries(catalog).map(([key, item]) => (
                        <div key={key} className="flex items-center justify-between bg-dark-800 rounded px-2 py-1">
                            <div>
                                <span className="text-xs text-dark-200">{item.name_ar}</span>
                                <span className="text-[10px] text-dark-500 ml-1">({item.capacity.toLocaleString()}ط)</span>
                            </div>
                            <div className="flex gap-1">
                                <button onClick={() => createFactory(key)}
                                    className="p-1 hover:bg-green-600 rounded text-green-400" title="إنشاء">
                                    <Plus className="w-3 h-3" />
                                </button>
                                <button onClick={() => veto(key)}
                                    className="p-1 hover:bg-red-600 rounded text-red-400" title="فيتو">
                                    <Ban className="w-3 h-3" />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
