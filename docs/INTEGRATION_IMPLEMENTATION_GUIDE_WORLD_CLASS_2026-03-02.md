# BI-IDE v8 — Integration Implementation Guide (World-Class)

> Practical execution guide aligned with verified codebase reality.
> Date: 2026-03-02

---

## 0) Execution Contract

- Edit only 3 files: `Layout.tsx`, `App.tsx`, `Sidebar.tsx`
- No feature creep — wire existing components only
- Preserve existing theme system and UI structure
- Validate after each wave
- Rollback = revert 3 files via `git checkout`

---

## 1) Wave 1 — Core Wiring

### Step 1: Upgrade shell in `Layout.tsx`

**File:** `apps/desktop-tauri/src/components/Layout.tsx`

**Exact changes:**
```diff
-import { Editor } from "./Editor";
-import { Terminal } from "./Terminal";
+import { MonacoEditor } from "./editor/MonacoEditor";
+import { RealTerminal } from "./terminal/RealTerminal";
```
```diff
 <div className="flex-1 overflow-hidden">
-  <Editor />
+  <MonacoEditor />
 </div>
```
```diff
 <div className="flex-shrink-0 overflow-hidden" style={{ height: terminalHeight }}>
-  <Terminal />
+  <RealTerminal />
 </div>
```

**Do not modify:** resize logic, Header/StatusBar interfaces, className props.

---

### Step 2: Add global command layer + store sync in `App.tsx`

**File:** `apps/desktop-tauri/src/App.tsx`

**2a. Add imports (after existing imports):**
```typescript
import { CommandPalette } from "./components/editor/CommandPalette";
import { QuickOpen } from "./components/editor/QuickOpen";
```

**2b. Add modal states (inside `App()`, after existing states):**
```typescript
const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
const [quickOpenOpen, setQuickOpenOpen] = useState(false);
```

**2c. Add `setDeviceId` to store destructuring:**
```diff
 const {
   currentWorkspace,
   setCurrentWorkspace,
   setSyncStatus,
   setTrainingStatus,
   updateSettings,
+  setDeviceId: setStoreDeviceId,
 } = useStore();
```

**2d. Call `setStoreDeviceId` during init (after `setDeviceId(info.device_id)`):**
```diff
 const info = await system.getInfo();
 setDeviceId(info.device_id);
+setStoreDeviceId(info.device_id);
```

**2e. Add global keyboard handler (new `useEffect` after existing ones):**
```typescript
useEffect(() => {
  const handleKeyDown = (e: KeyboardEvent) => {
    // Ctrl/Cmd+Shift+P → Command Palette
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "p") {
      e.preventDefault();
      setCommandPaletteOpen(prev => !prev);
      setQuickOpenOpen(false);
      return;
    }
    // Ctrl/Cmd+P → Quick Open (not in inputs)
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
```

**2f. Mount modals at end of root render (after `</div>` closing the main wrapper, before final `</div>`):**
```diff
       {currentWorkspace ? (
         <Layout deviceId={deviceId} />
       ) : (
         <WelcomeScreen deviceId={deviceId} />
       )}
+
+      <CommandPalette isOpen={commandPaletteOpen} onClose={() => setCommandPaletteOpen(false)} />
+      <QuickOpen isOpen={quickOpenOpen} onClose={() => setQuickOpenOpen(false)} />
     </div>
```

---

### Step 3: Modularize `Sidebar.tsx`

**File:** `apps/desktop-tauri/src/components/Sidebar.tsx`

**3a. Add imports (after existing imports):**
```diff
 import { CouncilPanel } from "./CouncilPanel";
 import { HierarchyPanel } from "./HierarchyPanel";
+import { GitPanel } from "./git/GitPanel";
+import { SearchPanel } from "./editor/SearchPanel";
+import { SyncPanel } from "./sync/SyncPanel";
+import { TrainingDashboard } from "./training/TrainingDashboard";
```

**3b. Extend activeTab union:**
```diff
-useState<"explorer" | "search" | "git" | "ai" | "training" | "council" | "hierarchy">
+useState<"explorer" | "search" | "git" | "sync" | "ai" | "training" | "council" | "hierarchy">
```

**3c. Add `sync` + `training` tab buttons (in tab bar, after Git button, before AI button):**
```tsx
<button
  onClick={() => setActiveTab("sync")}
  className={`flex-1 py-2 text-xs font-medium transition-colors relative ${
    activeTab === "sync"
      ? "text-blue-400 border-b-2 border-blue-500"
      : "text-dark-400 hover:text-dark-200"
  }`}
  title="المزامنة"
>
  ☁️
</button>
<button
  onClick={() => setActiveTab("training")}
  className={`flex-1 py-2 text-xs font-medium transition-colors relative ${
    activeTab === "training"
      ? "text-green-400 border-b-2 border-green-500"
      : "text-dark-400 hover:text-dark-200"
  }`}
  title="التدريب"
>
  🎓
</button>
```

**3d. Replace Search inline (lines 458-472):**
```diff
-{activeTab === "search" && (
-  <div className="p-3">
-    <div className="relative">
-      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-500" />
-      <input type="text" placeholder="Search files..." ... />
-    </div>
-    <p className="text-xs text-dark-500 mt-4 text-center">Search across workspace</p>
-  </div>
-)}
+{activeTab === "search" && (
+  <SearchPanel isOpen={activeTab === "search"} onClose={() => setActiveTab("explorer")} />
+)}
```

**3e. Replace Git inline (lines 474-533):**
```diff
-{activeTab === "git" && (
-  <div className="p-3">
-    {!gitState ? ( ... ) : ( ... )}
-  </div>
-)}
+{activeTab === "git" && <GitPanel />}
```

**3f. Add Sync panel (new, after Git):**
```diff
+{activeTab === "sync" && <SyncPanel />}
```

**3g. Replace Training inline (lines 535-579):**
```diff
-{activeTab === "training" && (
-  <div className="p-3">
-    <div className="flex items-center gap-2 mb-3">...</div>
-    {trainingStatus.currentJob && (...)}
-    <div className="grid grid-cols-2 gap-2">...</div>
-  </div>
-)}
+{activeTab === "training" && <TrainingDashboard />}
```

**Keep intact:** Explorer (TreeNode lines 31-96), AI Chat (lines 399-456), Council (line 581), Hierarchy (line 582).

---

## 2) Wave 2 — Validation

### Build
```bash
cd apps/desktop-tauri
npm run build              # TypeScript compile check
npm run tauri build        # Full desktop .app build
```

### Smoke Test (11 items)
| # | Test | Expected |
|---|------|----------|
| 1 | Launch app (no workspace) | WelcomeScreen |
| 2 | Open Folder | Explorer file tree |
| 3 | Click file | Monaco Editor + syntax highlighting |
| 4 | Ctrl+Shift+P | Command Palette (30+ commands) |
| 5 | Ctrl+P | Quick Open (fuzzy search) |
| 6 | Toggle Terminal | xterm.js with real shell |
| 7 | Search tab | SearchPanel with ripgrep |
| 8 | Git tab | GitPanel with stage/commit |
| 9 | Sync tab | SyncPanel with "This device" |
| 10 | Training tab | TrainingDashboard with start/pause |
| 11 | Council + Hierarchy | Unchanged panels |

---

## 3) Wave 3 — Hardening (Recommended)

- Verify behavior with no active workspace
- Verify behavior inside non-git directory
- Confirm no stuck keyboard listeners after closing modals
- Confirm panel switch does not freeze UI

---

## 4) Rollback Plan

If regression appears:
1. Revert only 3 touched files:
   ```bash
   git checkout HEAD~1 -- \
     apps/desktop-tauri/src/components/Layout.tsx \
     apps/desktop-tauri/src/App.tsx \
     apps/desktop-tauri/src/components/Sidebar.tsx
   ```
2. Component files stay untouched (zero risk)
3. Re-run `npm run build` to validate rollback

---

## 5) File Paths Summary

### Primary (edit)
- `apps/desktop-tauri/src/components/Layout.tsx` — Algorithm A
- `apps/desktop-tauri/src/App.tsx` — Algorithms B + C
- `apps/desktop-tauri/src/components/Sidebar.tsx` — Algorithm D

### Components (wire, don't modify)
- `apps/desktop-tauri/src/components/editor/MonacoEditor.tsx`
- `apps/desktop-tauri/src/components/editor/CommandPalette.tsx`
- `apps/desktop-tauri/src/components/editor/QuickOpen.tsx`
- `apps/desktop-tauri/src/components/editor/SearchPanel.tsx`
- `apps/desktop-tauri/src/components/terminal/RealTerminal.tsx`
- `apps/desktop-tauri/src/components/git/GitPanel.tsx`
- `apps/desktop-tauri/src/components/sync/SyncPanel.tsx`
- `apps/desktop-tauri/src/components/training/TrainingDashboard.tsx`

### Shared (reference only)
- `apps/desktop-tauri/src/lib/store.ts` — `setDeviceId` (L111)
- `apps/desktop-tauri/src/lib/tauri.ts` — all API wrappers

---

## 6) Quality Bar (10/10 Standard)

هذه الخطة مكتملة فقط إذا تحقق:
- ✅ Zero build errors
- ✅ Zero dead tabs
- ✅ Zero shortcut collisions
- ✅ Store identity synced (`deviceId`)
- ✅ `SearchPanel` mounted with correct `{isOpen, onClose}` props
- ✅ Training tab button visible
- ✅ User flow متماسك من first open to commit/sync/training
- ✅ Rollback tested or documented

When all above pass, integration is globally competitive by engineering quality, not by marketing claims.