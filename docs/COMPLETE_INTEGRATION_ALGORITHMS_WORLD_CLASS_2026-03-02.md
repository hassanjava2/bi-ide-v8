# BI-IDE v8 — Complete Integration Algorithms (World-Class Edition)

> Date: 2026-03-02
> Objective: make desktop integration production-grade, measurable, and rollback-safe.
> Scope: desktop frontend wiring + stability guardrails only (no unrelated feature expansion).

---

## 1) North Star

تحويل تطبيق `desktop-tauri` من تجميع مكونات إلى نظام مترابط تشغيليًا (Operationally Integrated) مع:
- تجربة استخدام موحّدة
- اختصارات عالمية بدون تضارب
- لوحات (Panels) مرتبطة فعلًا بحالة النظام
- قابلية قياس (KPIs) + rollback سريع

---

## 2) Current Reality (Verified from Source)

### A. Core Gaps (6 items)

| # | الفجوة | الملف | السطور | الأثر |
|---|--------|-------|--------|-------|
| 1 | `Layout.tsx` يستخدم `Editor` و `Terminal` القديمين | `Layout.tsx:8-9,84,96` | imports + JSX | محرر textarea بدل Monaco |
| 2 | `App.tsx` لا يركّب `CommandPalette` و `QuickOpen` | `App.tsx` | missing imports + render | بدون Ctrl+P / Ctrl+Shift+P |
| 3 | `App.tsx` لا يستدعي `store.setDeviceId()` | `App.tsx:26` | init block | `SyncPanel` يعرض deviceId فارغ |
| 4 | `Sidebar.tsx` يملك Search/Git/Training كـ inline blocks | `Sidebar.tsx:458-579` | ~120 سطر inline | panels جاهزة لكن غير مستخدمة |
| 5 | `SearchPanel` يتطلب `{isOpen, onClose}` | `SearchPanel.tsx:13-16,150` | conditional render | يرجع `null` بدون `isOpen` |
| 6 | زر `training` غير موجود بشريط التبويبات | `Sidebar.tsx:302-362` | tab bar | المحتوى موجود (L535) بدون tab |

### B. Verified Component Signatures

| Component | File | Props Required | Self-Managed State |
|-----------|------|----------------|-------------------|
| `MonacoEditor` | `editor/MonacoEditor.tsx` (363L) | `{ className? }` | store: openFiles, activeFilePath, settings |
| `RealTerminal` | `terminal/RealTerminal.tsx` (280L) | none | tauri: terminal.spawn/read/write/kill |
| `CommandPalette` | `editor/CommandPalette.tsx` (637L) | `{ isOpen, onClose }` | store: 30+ commands built-in |
| `QuickOpen` | `editor/QuickOpen.tsx` (305L) | `{ isOpen, onClose }` | store + tauri: fs recursive scan |
| `SearchPanel` | `editor/SearchPanel.tsx` (402L) | `{ isOpen, onClose }` | tauri: invoke("search_workspace") |
| `GitPanel` | `git/GitPanel.tsx` (331L) | none | store + tauri: git.* commands |
| `SyncPanel` | `sync/SyncPanel.tsx` | none | store: `deviceId` (must be set first) |
| `TrainingDashboard` | `training/TrainingDashboard.tsx` | none | store + tauri: training.* commands |

### C. Store Dependency Chain

```
App.tsx init → system.getInfo() → store.setDeviceId(info.device_id)
                                                    ↓
SyncPanel reads store.deviceId → identifies "This device" vs "Remote"
```

> **سطر 111 في store.ts:** `setDeviceId: (id: string) => void`
> **سطر 258 في store.ts:** `setDeviceId: (id) => set({ deviceId: id })`
> **سطر 19 في SyncPanel.tsx:** `const { currentWorkspace, deviceId } = useStore()`

---

## 3) Integration Algorithms (Execution Logic)

### Algorithm A — Shell Upgrade (P0)

**File:** `Layout.tsx`

**Changes:**
```diff
-import { Editor } from "./Editor";
-import { Terminal } from "./Terminal";
+import { MonacoEditor } from "./editor/MonacoEditor";
+import { RealTerminal } from "./terminal/RealTerminal";

-<Editor />
+<MonacoEditor />

-<Terminal />
+<RealTerminal />
```

**Keep unchanged:** sidebar/terminal resize behavior, Header/StatusBar interfaces.

**Acceptance:**
- [ ] فتح ملف يظهر داخل Monaco tabs
- [ ] terminal toggle يشغّل PTY حقيقي
- [ ] resize يعمل بدون regression

---

### Algorithm B — Global Command Layer (P0)

**File:** `App.tsx`

**Changes:**
1. Import `CommandPalette` + `QuickOpen`
2. Add states: `commandPaletteOpen`, `quickOpenOpen`
3. Add global keydown handler:
   - `Ctrl/Cmd+Shift+P` → toggle CommandPalette, close QuickOpen
   - `Ctrl/Cmd+P` → toggle QuickOpen, close CommandPalette
   - Ignore in `HTMLInputElement` / `HTMLTextAreaElement`
4. Mount both modals at root render (after Layout/WelcomeScreen):
   ```tsx
   <CommandPalette isOpen={commandPaletteOpen} onClose={() => setCommandPaletteOpen(false)} />
   <QuickOpen isOpen={quickOpenOpen} onClose={() => setQuickOpenOpen(false)} />
   ```

**Acceptance:**
- [ ] hotkeys تعمل بأي شاشة
- [ ] لا collision مع typing في inputs
- [ ] فتح واحد يغلق الآخر

---

### Algorithm C — Store Identity Sync (P0)

**File:** `App.tsx`

**Change:** After `system.getInfo()`, call `setDeviceId` from store:
```typescript
const { setDeviceId, ...rest } = useStore();
// in init:
const info = await system.getInfo();
setDeviceId(info.device_id);  // ← NEW: critical for SyncPanel
```

**Acceptance:**
- [ ] `SyncPanel` يميّز "This device" بشكل صحيح
- [ ] `deviceId` غير فارغ بعد init

---

### Algorithm D — Sidebar Modularization (P1)

**File:** `Sidebar.tsx`

**Changes:**

#### D1. Imports
```diff
+import { GitPanel } from "./git/GitPanel";
+import { SearchPanel } from "./editor/SearchPanel";
+import { SyncPanel } from "./sync/SyncPanel";
+import { TrainingDashboard } from "./training/TrainingDashboard";
```

#### D2. Tab type extension
```diff
-useState<"explorer" | "search" | "git" | "ai" | "training" | "council" | "hierarchy">
+useState<"explorer" | "search" | "git" | "sync" | "ai" | "training" | "council" | "hierarchy">
```

#### D3. Tab bar buttons — add `sync` + `training` (currently missing)
```tsx
{/* After Git button, before AI button: */}
<button onClick={() => setActiveTab("sync")} ...>☁️</button>
<button onClick={() => setActiveTab("training")} ...>🎓</button>
```

#### D4. Content mounts — replace inline blocks
```diff
-{activeTab === "search" && ( <div className="p-3">..inline 14 lines..</div> )}
+{activeTab === "search" && (
+  <SearchPanel isOpen={activeTab === "search"} onClose={() => setActiveTab("explorer")} />
+)}

-{activeTab === "git" && ( <div className="p-3">..inline 60 lines..</div> )}
+{activeTab === "git" && <GitPanel />}

+{activeTab === "sync" && <SyncPanel />}

-{activeTab === "training" && ( <div className="p-3">..inline 44 lines..</div> )}
+{activeTab === "training" && <TrainingDashboard />}
```

**Keep intact:** Explorer (TreeNode), AI Chat, Council, Hierarchy — no changes.

**Acceptance:**
- [ ] كل tab يظهر panel الصحيح
- [ ] لا tabs ميتة (dead tabs)
- [ ] SearchPanel يعرض ويخفي بشكل سليم
- [ ] training tab button مرئي

---

### Algorithm E — Operational Hardening (P1)

**Scope:** Verify during testing, no code changes needed.

- Validate commands invoked by panels (`search_workspace`, `force_sync`, git methods)
- Commands that fail gracefully show error state (already built into components)
- No crash on: no workspace, non-git repo, backend unavailable

**Acceptance:**
- [ ] لا crash عند فشل command
- [ ] رسائل خطأ واضحة للمستخدم

---

## 4) Execution Order (World-Class Rollout)

### Wave 1 — Core Wiring (30 minutes)

| # | Algorithm | File | Lines Changed |
|---|-----------|------|---------------|
| 1 | A: Shell Upgrade | `Layout.tsx` | ~4 lines |
| 2 | B: Command Layer | `App.tsx` | ~25 lines |
| 3 | C: Store Identity | `App.tsx` | ~2 lines |
| 4 | D: Sidebar Modular | `Sidebar.tsx` | ~15 lines (replace ~120 inline) |

### Wave 2 — Validation (10 minutes)

| # | Task | Command |
|---|------|---------|
| 1 | TypeScript compile | `npm run build` |
| 2 | Desktop build | `npm run tauri build` |
| 3 | Smoke test 11 items | Manual checklist below |

### Wave 3 — Optimization (Optional, future)

- Lazy-mount heavy panels if startup latency observed
- Telemetry counters for tab usage

---

## 5) File Map (Exact Paths)

### Primary files to edit (3)
```
/Users/bi/Documents/bi-ide-v8/apps/desktop-tauri/src/components/Layout.tsx
/Users/bi/Documents/bi-ide-v8/apps/desktop-tauri/src/App.tsx
/Users/bi/Documents/bi-ide-v8/apps/desktop-tauri/src/components/Sidebar.tsx
```

### Components to wire (8, reuse only — no modifications)
```
.../components/editor/MonacoEditor.tsx     363L  {className?}
.../components/editor/CommandPalette.tsx   637L  {isOpen, onClose}
.../components/editor/QuickOpen.tsx        305L  {isOpen, onClose}
.../components/editor/SearchPanel.tsx      402L  {isOpen, onClose}
.../components/terminal/RealTerminal.tsx   280L  (none)
.../components/git/GitPanel.tsx            331L  (none)
.../components/sync/SyncPanel.tsx          ~250L (none, reads store.deviceId)
.../components/training/TrainingDashboard.tsx ~300L (none)
```

### Shared APIs/state (2, read-only — no modifications)
```
.../lib/store.ts    — setDeviceId (L111), deviceId state
.../lib/tauri.ts    — terminal, git, sync, training, ai, fs APIs
```

---

## 6) KPI Success Criteria

| # | KPI | Target | How to Measure |
|---|-----|--------|----------------|
| 1 | TypeScript compile | 0 errors | `npm run build` |
| 2 | Tab coverage | 8/8 tabs route to panels | Manual click-through |
| 3 | Global shortcut latency | < 100ms perceived | Ctrl+P / Ctrl+Shift+P |
| 4 | Explorer regression | 0 regressions | Open folder → click file |
| 5 | Device identity | correct "This device" | Open Sync tab |
| 6 | No-workspace safety | no crash | App without workspace → toggle tabs |
| 7 | Non-git safety | no crash | Open non-git folder → git tab |

---

## 7) Risk Register + Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Hotkey conflict between global handlers and modal handlers | UX confusion | Medium | Gate by `isOpen`, close peer modal on open |
| `SearchPanel` invisible due to missing `isOpen` | Dead tab | High if missed | Mount with `isOpen={activeTab === "search"}` |
| Sync identity wrong | Wrong device label | High if missed | Call `store.setDeviceId()` during `App.tsx` init |
| Heavy dashboard affects startup perf | Slow UI | Low | Conditional render by active tab only (already) |
| xterm.js CSS not loaded | Broken terminal | Low | Verify `import "xterm/css/xterm.css"` in RealTerminal (already present) |

---

## 8) Smoke Test Checklist (11 items)

| # | Test | Expected |
|---|------|----------|
| 1 | Launch app (no workspace) | WelcomeScreen appears |
| 2 | Open Folder | Explorer shows file tree |
| 3 | Click file | Monaco Editor with syntax highlighting |
| 4 | Ctrl+Shift+P | Command Palette (30+ commands) |
| 5 | Ctrl+P | Quick Open (fuzzy file search) |
| 6 | Toggle Terminal (Ctrl+`) | xterm.js with real shell |
| 7 | Search tab | SearchPanel with ripgrep |
| 8 | Git tab | GitPanel with status/stage/commit |
| 9 | Sync tab | SyncPanel with "This device" label |
| 10 | Training tab | TrainingDashboard with start/pause |
| 11 | Council + Hierarchy | Existing panels unchanged |

---

## 9) Rollback Plan

If regression appears:
1. Revert only 3 touched files: `Layout.tsx`, `App.tsx`, `Sidebar.tsx`
2. Component files stay untouched (no risk)
3. `git checkout HEAD~1 -- apps/desktop-tauri/src/components/Layout.tsx apps/desktop-tauri/src/App.tsx apps/desktop-tauri/src/components/Sidebar.tsx`
4. Re-run `npm run build` to validate rollback

---

## 10) Definition of Done

- [ ] All 5 algorithms (A-E) implemented
- [ ] `npm run build` succeeds with 0 errors
- [ ] `npm run tauri build` produces .app
- [ ] All 11 smoke test items pass
- [ ] All 7 KPIs met
- [ ] No unrelated refactors introduced
- [ ] Committed with descriptive message
- [ ] Pushed to `origin/main`

---

## Final Verdict

بهذا التصميم، الربط يصير فعلاً production-grade: واضح، قابل للقياس، وقابل للرجوع (rollback) بسرعة.

الفرق عن خطط سابقة:
- **دقة تقنية**: كل prop محقق من الكود الفعلي
- **store sync**: `setDeviceId` مربوط بشكل صحيح
- **SearchPanel**: يُعامل بـ `isOpen` prop وليس drop-in
- **training tab**: زر مرئي مضاف
- **قابلية القياس**: 7 KPIs + 11 smoke tests
- **قابلية الرجوع**: 3 ملفات فقط + أمر git واحد