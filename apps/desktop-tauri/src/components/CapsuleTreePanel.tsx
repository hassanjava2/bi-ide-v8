import { useState, useEffect } from "react";
import { TreePine, Search, RefreshCw, GitMerge, AlertTriangle, Wrench, ChevronDown, ChevronRight } from "lucide-react";

const API = "http://localhost:8400";

interface TreeStats {
    trees: number;
    total_nodes: number;
    pyramids: number;
    capsules: number;
    trained: number;
    auto_created: number;
    orphans: number;
    inheritance_links: number;
}

export function CapsuleTreePanel() {
    const [treeView, setTreeView] = useState("");
    const [stats, setStats] = useState<TreeStats | null>(null);
    const [searchQuery, setSearchQuery] = useState("");
    const [searchResults, setSearchResults] = useState<any[]>([]);
    const [expanded, setExpanded] = useState(true);
    const [loading, setLoading] = useState(false);

    const fetchTree = async () => {
        try {
            const res = await fetch(`${API}/api/capsules/tree`);
            const data = await res.json();
            setTreeView(data.tree);
            setStats(data.stats);
        } catch { }
    };

    useEffect(() => { fetchTree(); }, []);

    const search = async () => {
        if (!searchQuery.trim()) return;
        try {
            const res = await fetch(`${API}/api/capsules/search?q=${encodeURIComponent(searchQuery)}`);
            const data = await res.json();
            setSearchResults(data.results);
        } catch { }
    };

    const cascadeAll = async () => {
        setLoading(true);
        try {
            await fetch(`${API}/api/capsules/cascade/engineering`, { method: "POST" });
            await fetch(`${API}/api/capsules/cascade/science`, { method: "POST" });
            await fetch(`${API}/api/capsules/cascade/manufacturing`, { method: "POST" });
            await fetchTree();
        } finally { setLoading(false); }
    };

    const fixOrphans = async () => {
        setLoading(true);
        try {
            await fetch(`${API}/api/capsules/fix-orphans`, { method: "POST" });
            await fetchTree();
        } finally { setLoading(false); }
    };

    const autoExpand = async () => {
        setLoading(true);
        try {
            await fetch(`${API}/api/capsules/expand`, {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: "robotics solar nano biotech renewable 3dprint" }),
            });
            await fetchTree();
        } finally { setLoading(false); }
    };

    return (
        <div className="p-3">
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <TreePine className="w-4 h-4 text-green-400" />
                    <span className="text-sm font-bold text-dark-200">أشجار الكبسولات 🌳🧬</span>
                </div>
                <button onClick={fetchTree} className="p-1 hover:bg-dark-800 rounded">
                    <RefreshCw className="w-3 h-3 text-dark-400" />
                </button>
            </div>

            {stats && (
                <div className="grid grid-cols-4 gap-1 mb-3">
                    <div className="bg-dark-800 rounded p-1.5 text-center">
                        <div className="text-sm font-bold text-green-400">{stats.trees}</div>
                        <div className="text-[9px] text-dark-400">شجرة</div>
                    </div>
                    <div className="bg-dark-800 rounded p-1.5 text-center">
                        <div className="text-sm font-bold text-blue-400">{stats.total_nodes}</div>
                        <div className="text-[9px] text-dark-400">عقدة</div>
                    </div>
                    <div className="bg-dark-800 rounded p-1.5 text-center">
                        <div className="text-sm font-bold text-purple-400">{stats.inheritance_links}</div>
                        <div className="text-[9px] text-dark-400">وراثة</div>
                    </div>
                    <div className="bg-dark-800 rounded p-1.5 text-center">
                        <div className="text-sm font-bold" style={{ color: stats.orphans > 0 ? "#ef4444" : "#22c55e" }}>
                            {stats.orphans}
                        </div>
                        <div className="text-[9px] text-dark-400">يتيمة</div>
                    </div>
                </div>
            )}

            {/* Actions */}
            <div className="flex gap-1 mb-3">
                <button onClick={cascadeAll} disabled={loading}
                    className="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-green-700 hover:bg-green-600 rounded text-[10px] font-medium text-white disabled:opacity-50">
                    <GitMerge className="w-3 h-3" /> وراثة
                </button>
                <button onClick={autoExpand} disabled={loading}
                    className="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-blue-700 hover:bg-blue-600 rounded text-[10px] font-medium text-white disabled:opacity-50">
                    <TreePine className="w-3 h-3" /> توسيع
                </button>
                {stats && stats.orphans > 0 && (
                    <button onClick={fixOrphans} disabled={loading}
                        className="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-red-700 hover:bg-red-600 rounded text-[10px] font-medium text-white disabled:opacity-50">
                        <Wrench className="w-3 h-3" /> إصلاح
                    </button>
                )}
            </div>

            {/* Search */}
            <div className="flex gap-1 mb-3">
                <input
                    value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && search()}
                    placeholder="بحث... (steel, solar, quantum)"
                    className="flex-1 bg-dark-800 text-xs text-dark-200 rounded px-2 py-1 outline-none"
                />
                <button onClick={search} className="p-1 bg-dark-800 hover:bg-dark-700 rounded">
                    <Search className="w-3 h-3 text-dark-400" />
                </button>
            </div>

            {searchResults.length > 0 && (
                <div className="space-y-1 mb-3">
                    {searchResults.map((r: any) => (
                        <div key={r.id} className="bg-dark-800 rounded px-2 py-1">
                            <span className="text-xs text-dark-200">{r.name_ar}</span>
                            <span className="text-[10px] text-dark-500 ml-1">({r.type})</span>
                        </div>
                    ))}
                </div>
            )}

            {/* Tree View */}
            <button onClick={() => setExpanded(!expanded)}
                className="flex items-center gap-1 text-xs text-dark-400 hover:text-dark-200 mb-1">
                {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                شجرة كاملة
            </button>
            {expanded && treeView && (
                <pre className="bg-dark-850 rounded p-2 text-[10px] text-dark-300 overflow-auto max-h-[300px] whitespace-pre-wrap">{treeView}</pre>
            )}
        </div>
    );
}
