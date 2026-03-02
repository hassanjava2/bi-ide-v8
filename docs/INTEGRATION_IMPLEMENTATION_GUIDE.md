# دليل تنفيذ الربط — Integration Implementation Guide
## BI-IDE v8 — الكود الفعلي الجاهز للنسخ

> **آخر تحديث**: 2026-03-02 — مصحح ومطابق لـ APIs الفعلية
> **الملفات المطلوب تعديلها**: 3 فقط
> **الوقت المقدر**: 30 دقيقة

---

## 📁 الملفات الثلاثة المطلوب تعديلها

```
apps/desktop-tauri/src/
├── components/
│   ├── Layout.tsx      ← تعديل: MonacoEditor + RealTerminal
│   ├── Sidebar.tsx     ← تعديل: GitPanel + SearchPanel + SyncPanel + TrainingDashboard
│   └── editor/
│       ├── MonacoEditor.tsx     ✅ جاهز (363 سطر)
│       ├── CommandPalette.tsx   ✅ جاهز (637 سطر, 30+ أمر)
│       ├── QuickOpen.tsx        ✅ جاهز (305 سطر, fuzzy search)
│       └── SearchPanel.tsx      ✅ جاهز
│   └── terminal/
│       └── RealTerminal.tsx     ✅ جاهز (280 سطر, xterm.js)
│   └── git/
│       └── GitPanel.tsx         ✅ جاهز (331 سطر)
│   └── sync/
│       └── SyncPanel.tsx        ✅ جاهز
│   └── training/
│       └── TrainingDashboard.tsx ✅ جاهز
├── App.tsx             ← تعديل: CommandPalette + QuickOpen
└── lib/
    ├── store.ts        ✅ جاهز (Zustand)
    └── tauri.ts        ✅ جاهز (528 سطر API wrapper)
```

---

## 1️⃣ Layout.tsx — الكود الكامل المعدل

```typescript
//! Main Layout Component — INTEGRATED VERSION
//! يربط: MonacoEditor + RealTerminal + Sidebar + Header + StatusBar

import { useState } from "react";
import { useStore } from "../lib/store";
import { Sidebar } from "./Sidebar";
import { StatusBar } from "./StatusBar";
import { Header } from "./Header";
import { MonacoEditor } from "./editor/MonacoEditor";
import { RealTerminal } from "./terminal/RealTerminal";

interface LayoutProps {
  deviceId: string;
}

export function Layout({ deviceId }: LayoutProps) {
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const [isResizingTerminal, setIsResizingTerminal] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const { sidebarVisible, terminalVisible, terminalHeight, setTerminalHeight } = useStore();

  const handleSidebarResize = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizingSidebar(true);
    const startX = e.clientX;
    const startWidth = sidebarWidth;

    const handleMouseMove = (e: MouseEvent) => {
      const newWidth = Math.max(200, Math.min(500, startWidth + e.clientX - startX));
      setSidebarWidth(newWidth);
    };

    const handleMouseUp = () => {
      setIsResizingSidebar(false);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  const handleTerminalResize = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizingTerminal(true);
    const startY = e.clientY;
    const startHeight = terminalHeight;

    const handleMouseMove = (e: MouseEvent) => {
      const newHeight = Math.max(100, Math.min(600, startHeight - (e.clientY - startY)));
      setTerminalHeight(newHeight);
    };

    const handleMouseUp = () => {
      setIsResizingTerminal(false);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  return (
    <div className="h-full flex flex-col">
      <Header deviceId={deviceId} />

      <div className="flex-1 flex overflow-hidden">
        {sidebarVisible && (
          <>
            <div className="h-full flex-shrink-0 overflow-hidden" style={{ width: sidebarWidth }}>
              <Sidebar />
            </div>
            <div
              className={`w-1 h-full cursor-col-resize transition-colors ${
                isResizingSidebar ? "bg-primary-500" : "bg-dark-700 hover:bg-primary-500"
              }`}
              onMouseDown={handleSidebarResize}
            />
          </>
        )}

        <div className="flex-1 flex flex-col min-w-0">
          {/* Monaco Editor — الأساسي */}
          <div className="flex-1 overflow-hidden">
            <MonacoEditor />
          </div>

          {/* Real Terminal — PTY حقيقي */}
          {terminalVisible && (
            <>
              <div
                className={`h-1 cursor-row-resize transition-colors ${
                  isResizingTerminal ? "bg-primary-500" : "bg-dark-700 hover:bg-primary-500"
                }`}
                onMouseDown={handleTerminalResize}
              />
              <div className="flex-shrink-0 overflow-hidden" style={{ height: terminalHeight }}>
                <RealTerminal />
              </div>
            </>
          )}
        </div>
      </div>

      <StatusBar deviceId={deviceId} />
    </div>
  );
}
```

---

## 2️⃣ App.tsx — الكود الكامل المعدل

```typescript
//! App.tsx — with CommandPalette (Ctrl+Shift+P) + QuickOpen (Ctrl+P)

import { useEffect, useState } from "react";
import { useStore } from "./lib/store";
import { system, workspace, sync, training } from "./lib/tauri";
import { Layout } from "./components/Layout";
import { WelcomeScreen } from "./components/WelcomeScreen";
import { CommandPalette } from "./components/editor/CommandPalette";
import { QuickOpen } from "./components/editor/QuickOpen";
import { listen } from "@tauri-apps/api/event";

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [deviceId, setDeviceId] = useState<string>("");

  // ─── Modal States ───
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [quickOpenOpen, setQuickOpenOpen] = useState(false);

  const {
    currentWorkspace,
    setCurrentWorkspace,
    setSyncStatus,
    setTrainingStatus,
    updateSettings,
  } = useStore();

  // ─── Initialize App ───
  useEffect(() => {
    const init = async () => {
      try {
        const info = await system.getInfo();
        setDeviceId(info.device_id);
        console.log("BI-IDE Desktop v" + info.app_version);
        console.log("Device ID:", info.device_id);
        console.log("Platform:", info.platform, info.arch);

        const savedWorkspace = localStorage.getItem("bi-ide-last-workspace");
        if (savedWorkspace) {
          try {
            const ws = await workspace.open(savedWorkspace);
            setCurrentWorkspace({ id: ws.id, path: ws.path, name: ws.name });
          } catch (e) {
            console.error("Failed to restore workspace:", e);
          }
        }

        const syncStatus = await sync.getStatus();
        setSyncStatus({
          isEnabled: syncStatus.enabled,
          isConnected: syncStatus.is_connected,
          lastSync: syncStatus.last_sync,
          pendingCount: syncStatus.pending_count,
        });

        const trainingStatus = await training.getStatus();
        setTrainingStatus({
          isEnabled: trainingStatus.enabled,
          currentJob: trainingStatus.current_job
            ? {
                id: trainingStatus.current_job.job_id,
                type: trainingStatus.current_job.job_type,
                progress: trainingStatus.current_job.progress_percent,
                status: trainingStatus.current_job.status,
              }
            : undefined,
        });
      } catch (error) {
        console.error("Initialization error:", error);
      } finally {
        setIsLoading(false);
      }
    };

    init();
  }, [setCurrentWorkspace, setSyncStatus, setTrainingStatus]);

  // ─── Global Keyboard Shortcuts ───
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+Shift+P → Command Palette
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "p") {
        e.preventDefault();
        setCommandPaletteOpen(prev => !prev);
        setQuickOpenOpen(false);
        return;
      }

      // Ctrl+P → Quick Open (not in inputs)
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "p" && !e.shiftKey) {
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
        e.preventDefault();
        setQuickOpenOpen(prev => !prev);
        setCommandPaletteOpen(false);
        return;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // ─── Listen for Tauri Events ───
  useEffect(() => {
    const unlistenResource = listen("resource-usage", (event) => {
      console.log("Resource usage:", event.payload);
    });

    const unlistenSync = listen("sync-complete", (event) => {
      console.log("Sync complete:", event.payload);
      sync.getStatus().then((status) => {
        setSyncStatus({
          isEnabled: status.enabled,
          isConnected: status.is_connected,
          lastSync: status.last_sync,
          pendingCount: status.pending_count,
        });
      });
    });

    const unlistenFileChange = listen("file-changed", (event) => {
      console.log("File changed:", event.payload);
    });

    return () => {
      unlistenResource.then((fn) => fn());
      unlistenSync.then((fn) => fn());
      unlistenFileChange.then((fn) => fn());
    };
  }, [setSyncStatus]);

  // ─── Auto-save Workspace ───
  useEffect(() => {
    if (currentWorkspace) {
      localStorage.setItem("bi-ide-last-workspace", currentWorkspace.path);
    }
  }, [currentWorkspace]);

  // ─── Loading Screen ───
  if (isLoading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-dark-900">
        <div className="text-center">
          <div className="text-2xl font-bold text-primary-400 mb-2">BI-IDE Desktop</div>
          <div className="text-dark-400">Initializing...</div>
        </div>
      </div>
    );
  }

  // ─── Main Render ───
  return (
    <div className="h-screen w-screen bg-dark-900 text-dark-100 overflow-hidden">
      {currentWorkspace ? (
        <Layout deviceId={deviceId} />
      ) : (
        <WelcomeScreen deviceId={deviceId} />
      )}

      {/* ═══ Global Modals ═══ */}
      <CommandPalette
        isOpen={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
      />
      <QuickOpen
        isOpen={quickOpenOpen}
        onClose={() => setQuickOpenOpen(false)}
      />
    </div>
  );
}

export default App;
```

---

## 3️⃣ Sidebar.tsx — التعديلات المطلوبة

> **ملاحظة:** لا نعيد كتابة Sidebar كاملاً. فقط نستبدل المحتوى المضمن بالكمبوننتس الجاهزة.

### التعديلات (diff):

```diff
 // ─── سطر 1: إضافة imports ───
 import { CouncilPanel } from "./CouncilPanel";
 import { HierarchyPanel } from "./HierarchyPanel";
+import { GitPanel } from "./git/GitPanel";
+import { SearchPanel } from "./editor/SearchPanel";
+import { SyncPanel } from "./sync/SyncPanel";
+import { TrainingDashboard } from "./training/TrainingDashboard";

 // ─── سطر 100: إضافة "sync" للـ type ───
-const [activeTab, setActiveTab] = useState<"explorer" | "search" | "git" | "ai" | "training" | "council" | "hierarchy">("explorer");
+const [activeTab, setActiveTab] = useState<"explorer" | "search" | "git" | "sync" | "ai" | "training" | "council" | "hierarchy">("explorer");

 // ─── بعد زر Git وقبل زر AI: إضافة زر Sync ───
+<button
+  onClick={() => setActiveTab("sync")}
+  className={`flex-1 py-2 text-xs font-medium transition-colors ${
+    activeTab === "sync"
+      ? "text-blue-400 border-b-2 border-blue-500"
+      : "text-dark-400 hover:text-dark-200"
+  }`}
+  title="المزامنة"
+>
+  ☁️
+</button>

 // ─── استبدال محتوى Search (السطور 458-472) ───
-{activeTab === "search" && (
-  <div className="p-3">
-    <div className="relative">
-      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-500" />
-      <input type="text" placeholder="Search files..." ... />
-    </div>
-    <p className="text-xs text-dark-500 mt-4 text-center">Search across workspace</p>
-  </div>
-)}
+{activeTab === "search" && <SearchPanel />}

 // ─── استبدال محتوى Git (السطور 474-533) ───
-{activeTab === "git" && (
-  <div className="p-3">
-    {!gitState ? ( ... ) : ( ... )}
-  </div>
-)}
+{activeTab === "git" && <GitPanel />}

 // ─── إضافة Sync ───
+{activeTab === "sync" && <SyncPanel />}

 // ─── استبدال محتوى Training (السطور 535-579) ───
-{activeTab === "training" && (
-  <div className="p-3">
-    <div className="flex items-center gap-2 mb-3"> ... </div>
-    {trainingStatus.currentJob && ( ... )}
-    <div className="grid grid-cols-2 gap-2"> ... </div>
-  </div>
-)}
+{activeTab === "training" && <TrainingDashboard />}
```

> **هام:** نحافظ على Explorer tab بالكود المضمن (TreeNode) والـ AI chat — فقط نستبدل search/git/training ونضيف sync.

---

## 🔨 أوامر التنفيذ

```bash
# 1. بعد تعديل الملفات الثلاثة:
cd ~/Documents/bi-ide-v8/apps/desktop-tauri

# 2. بناء وتحقق
npm run build          # frontend فقط — سريع

# 3. بناء كامل
npm run tauri build    # frontend + Rust → .app

# 4. تثبيت
cp -R "../../target/release/bundle/macos/BI-IDE Desktop.app" /Applications/

# 5. commit + push
cd ~/Documents/bi-ide-v8
git add -A
git commit -m "feat: wire all components - Monaco, Terminal, CommandPalette, QuickOpen, GitPanel, SyncPanel, TrainingDashboard"
git push origin main

# 6. التشغيل
open "/Applications/BI-IDE Desktop.app"
```

---

## ✅ Checklist التحقق

| # | الاختبار | المتوقع |
|---|---------|---------|
| 1 | فتح التطبيق | WelcomeScreen يظهر |
| 2 | Open Folder | Monaco Editor يعرض الملفات |
| 3 | Ctrl+Shift+P | Command Palette يظهر (30+ أمر) |
| 4 | Ctrl+P | Quick Open يظهر (fuzzy search) |
| 5 | Toggle Terminal | xterm.js مع shell حقيقي |
| 6 | Git tab | stage/commit/push/pull |
| 7 | Search tab | بحث في كل الملفات |
| 8 | Sync tab | حالة المزامنة |
| 9 | Training tab | لوحة التدريب |
| 10 | Council tab | مجلس الحكماء |
| 11 | Hierarchy tab | النظام الهرمي |

---

**الحالة**: جاهز للتنفيذ — انسخ الكود مباشرة
