# IDE Ideas Master Backlog
# سجل أفكار تطوير الـ IDE (ملف موحّد)

آخر تحديث: 2026-02-22
المصدر: `docs/ROADMAP.md` + `docs/TASKS.md` + مراجعة الكود الفعلي (`ide/` + `ui/src/pages/IDE.tsx`)

---

## 1) الوضع الحالي الفعلي (Reality Check)

### Backend IDE (`ide/ide_service.py`)
- `FileSystemManager` يعمل على ملفات المشروع الحقيقية ضمن sandbox.
- `AICopilot` يدعم context-aware v1 مع fallback (ما زال يحتاج تحسين الجودة).
- `TerminalManager` يعمل بتنفيذ فعلي مع session isolation.
- يوجد Static diagnostics MVP + Git MVP (status/diff/commit).

### Frontend IDE (`ui/src/pages/IDE.tsx`)
- صفحة IDE مربوطة فعلياً مع backend (files/editor/terminal/diagnostics/git).
- ما زالت بحاجة تحسينات UX وتوسيع ميزات copilot/debug.

---

## 2) الأفكار غير المطبقة (مرتبة حسب الأولوية)

## P0 (ضروري الآن)
- [x] ربط صفحة IDE بالـAPI الحقيقي (`/api/v1/ide/files`, `/api/v1/ide/copilot/suggest`, `/api/v1/ide/terminal/execute`).
- [x] تحويل `FileSystemManager` من demo إلى file-system حقيقي مع sandbox آمن.
- [x] تحويل `TerminalManager` من mock إلى تنفيذ فعلي مع session isolation.
- [x] تحسين Copilot ليكون context-aware على مستوى الملف + المشروع.
- [x] تحسين جودة الردود/الاقتراحات لمنع التكرار.

## P1 (مهم جداً)
- [x] Static analysis حقيقي (AST + lint + diagnostics).
- [x] Refactoring suggestions (rename/extract/fix imports) [MVP].
- [x] Debugging tools (breakpoints + variable inspection + call stack) [MVP].
- [x] Git integration حقيقي (status/diff/commit/branch).
- [x] توليد اختبارات للوحدات مع templates واقعية لكل لغة [MVP].

## P2 (مرحلة توسعة)
- [ ] Multi-language depth (Python/TS/JS/Rust/Go) مع provider per language.
- [x] Documentation lookup from symbol context.
- [ ] Multi-line completions and ranked suggestions.
- [ ] Inline explain/fix buttons داخل المحرر.
- [ ] Live collaboration features داخل IDE.

## P3 (تحسينات إنتاج)
- [ ] Performance profiling داخل IDE.
- [ ] Caching لاقتراحات Copilot.
- [ ] Telemetry اختياري لجودة الاقتراحات.
- [ ] Feature flags لتفعيل/تعطيل ميزات IDE حسب البيئة.

---

## 3) المهام المستخرجة من ROADMAP/TASKS الخاصة بالـ IDE

### من Phase 3.1 (IDE Enhancement)
- [ ] تطوير Copilot المتقدم (Context-aware).
- [ ] إضافة Static Analysis.
- [ ] بناء Debugging tools.
- [ ] تكامل Git.
- [ ] دعم Multi-language.
- [ ] تحسين UI/UX.

### تعريف “جاهز” لكل بند
- Copilot المتقدم: نتائج مرتبطة بالسياق + confidence + multi-line.
- Static Analysis: تشخيصات واضحة مع fix suggestions.
- Debugging: تشغيل/إيقاف/خطوات + مشاهدة متغيرات.
- Git: status/diff/commit/push/pull من الواجهة.
- UI/UX: صفحة IDE مرتبطة API فعلياً بدون بيانات mock.

---

## 4) خطة تنفيذ سريعة (سبرنتات)

### Sprint A (3-5 أيام) — إزالة الـmock
- [x] ربط IDE.tsx مع API الحقيقي.
- [x] شجرة ملفات حقيقية + فتح/حفظ ملف.
- [x] Terminal execute فعلي (جلسة واحدة كبداية).

### Sprint B (5-7 أيام) — Copilot + تحليل
- [ ] سياق الملف + المشروع للـcopilot.
- [ ] parse suggestion format موحد.
- [ ] static diagnostics (lint/errors).

### Sprint C (7-10 أيام) — Debug/Git
- [x] أدوات debug الأساسية.
- [x] Git status/diff/commit.
- [x] تحسينات UX نهائية.

---

## 5) قائمة متابعة أسبوعية (حتى ما تنسى)

- [ ] أي ميزة جديدة بالـIDE لازم تتربط بـAPI وليس mock.
- [ ] أي اقتراح AI لازم ينقاس: `accept_rate`, `edit_distance_after_accept`.
- [ ] أي تغيير بالواجهة لازم يمر على test smoke بسيط.
- [ ] كل نهاية أسبوع: تحديث هذا الملف (Completed / Next Week).

---

## 6) سجل القرارات

### 2026-02-21
- تم اعتماد هذا الملف كمرجع واحد لأفكار وخطط IDE.
- تم تحديد أن أكبر فجوة حالياً: الفرق بين واجهة IDE الحالية (mock) والخدمات الخلفية.
- تم تنفيذ Terminal فعلي مع sessions معزولة + endpoint إنشاء session.
- تم تنفيذ FileSystem حقيقي داخل sandbox + ربط فتح/حفظ من الواجهة.
- تم تنفيذ النسخة الأولى من Copilot context-aware (سياق الملف + المشروع + fallback ذكي).
- تم تنفيذ MVP للـ Static diagnostics وربطه في واجهة IDE (Analyze + issues panel).

### 2026-02-22
- تم تنفيذ Git MVP داخل IDE: status + diff + commit عبر API وربطه بالواجهة.
- تم توسيع Git في الـIDE بإضافة push/pull مع اختيار remote/branch من نفس لوحة Git.
- تم تنفيذ Debug MVP داخل IDE لملفات Python عبر pdb: start/stop session + breakpoint + continue/step/next + stack/locals.
- تم تنفيذ Refactor Suggestions MVP وربطه بالواجهة (اقتراحات extract/reduce complexity/rename params/split class).
- تم تنفيذ Test Generation MVP عبر endpoint مخصص وربطه بواجهة IDE مع معاينة المحتوى والمسار المقترح للاختبار.
- تم تحسين UX للوحة أدوات IDE عبر تبويبات سريعة (Diagnostics/Refactor/Tests/Git/Debug) لتقليل الزحام وسهولة التنقل.
- تم إضافة Collapse/Expand للوحة الأدوات مع حفظ الحالة في localStorage.
- تم تنفيذ Copilot quality pass: dedupe + confidence normalization + heuristic ranking + multi-line fallback.
- تم توسيع Multi-language depth (مرحلة أولى) عبر تحسينات Rust/Go في الإكمال الذكي + refactor hints + test generation templates.
- تم تنفيذ Documentation Lookup end-to-end: backend symbol inference + API endpoint + Docs tab في واجهة IDE.
- تم إضافة Docs UX shortcuts: `Ctrl+Click` و `Ctrl+Shift+D` مع فتح تلقائي لتبويب Docs.
- تم ربط Location بالنقر لفتح الملف المرجعي والانتقال للسطر مع highlight مؤقت.
- تم تنفيذ Docs caching محلي (TTL=60s + LRU cap=100) مع Refresh/Clear/Reset وإظهار cache stats (size/evictions/source).
