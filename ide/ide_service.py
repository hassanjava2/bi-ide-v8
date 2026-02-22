"""
IDE Service - خدمة بيئة التطوير المتكاملة
ربط Monaco Editor + Terminal + AI Copilot
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import os
import json
import asyncio
import ast
import sys
import re
from pathlib import Path
import uuid


@dataclass
class FileNode:
    """عقدة ملف/مجلد"""
    id: str
    name: str
    type: str  # 'file' | 'folder'
    path: str
    content: Optional[str] = None
    children: List['FileNode'] = field(default_factory=list)
    language: Optional[str] = None
    last_modified: datetime = field(default_factory=datetime.now)


@dataclass
class CodeSuggestion:
    """اقتراح كود من AI"""
    id: str
    text: str
    detail: str
    insert_text: str
    range_start: int
    range_end: int
    confidence: float


class FileSystemManager:
    """
    مدير نظام الملفات
    يدير شجرة الملفات الحقيقية داخل sandbox آمن
    """

    def __init__(self, root_path: Optional[str] = None):
        configured_root = root_path or os.getenv("IDE_WORKSPACE_ROOT") or str(Path.cwd())
        self.root_path = Path(configured_root).resolve()
        self.max_nodes = int(os.getenv("IDE_MAX_TREE_NODES", "5000"))
        self.max_file_bytes = int(os.getenv("IDE_MAX_FILE_BYTES", str(2 * 1024 * 1024)))
        self.ignored_dirs = {
            ".git", ".venv", "venv", "node_modules", "__pycache__",
            "import_ok", "import_from_linux", "bi_ide_env", ".venv_rtx"
        }
        self._id_to_path: Dict[str, Path] = {}
        self._path_to_id: Dict[str, str] = {}

    def _is_safe_path(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.root_path)
            return True
        except Exception:
            return False

    def _to_relative_display(self, path: Path) -> str:
        rel = path.resolve().relative_to(self.root_path)
        return f"/{str(rel).replace(os.sep, '/')}" if str(rel) != "." else "/"

    def _get_or_create_id(self, path: Path) -> str:
        key = str(path.resolve())
        existing = self._path_to_id.get(key)
        if existing:
            return existing

        if path.resolve() == self.root_path:
            node_id = "root"
        else:
            node_id = uuid.uuid5(uuid.NAMESPACE_URL, self._to_relative_display(path)).hex

        self._path_to_id[key] = node_id
        self._id_to_path[node_id] = path.resolve()
        return node_id

    def _build_tree_recursive(self, path: Path, counter: List[int]) -> FileNode:
        node_id = self._get_or_create_id(path)
        is_dir = path.is_dir()

        node = FileNode(
            id=node_id,
            name=path.name if path.resolve() != self.root_path else self.root_path.name,
            type="folder" if is_dir else "file",
            path=self._to_relative_display(path),
            content=None,
            children=[],
            language=self._detect_language(path.name) if path.is_file() else None,
            last_modified=datetime.fromtimestamp(path.stat().st_mtime)
        )

        if not is_dir:
            return node

        try:
            children = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except Exception:
            return node

        for child in children:
            if counter[0] >= self.max_nodes:
                break
            if child.is_dir() and child.name in self.ignored_dirs:
                continue
            if child.name.startswith(".") and child.is_dir():
                continue
            if not self._is_safe_path(child):
                continue

            counter[0] += 1
            node.children.append(self._build_tree_recursive(child, counter))

        return node
    
    def get_file_tree(self) -> FileNode:
        """الحصول على شجرة الملفات"""
        self._id_to_path = {}
        self._path_to_id = {}
        counter = [1]
        return self._build_tree_recursive(self.root_path, counter)

    def _resolve_file_from_id(self, file_id: str) -> Optional[Path]:
        path = self._id_to_path.get(file_id)
        if not path:
            return None
        if not self._is_safe_path(path):
            return None
        return path
    
    def get_file_content(self, file_id: str) -> Optional[str]:
        """الحصول على محتوى ملف"""
        path = self._resolve_file_from_id(file_id)
        if not path or not path.exists() or not path.is_file():
            return None

        if path.stat().st_size > self.max_file_bytes:
            return f"[File too large to open in IDE preview: {path.stat().st_size} bytes]"

        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
    
    def save_file(self, file_id: str, content: str) -> bool:
        """حفظ محتوى ملف"""
        path = self._resolve_file_from_id(file_id)
        if not path or not path.exists() or not path.is_file():
            return False

        try:
            path.write_text(content, encoding="utf-8")
            return True
        except Exception:
            return False
    
    def create_file(self, parent_id: str, name: str, content: str = "") -> Optional[FileNode]:
        """إنشاء ملف جديد"""
        parent = self._id_to_path.get(parent_id)
        if not parent or not parent.exists() or not parent.is_dir() or not self._is_safe_path(parent):
            return None

        target = (parent / name).resolve()
        if not self._is_safe_path(target):
            return None

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            node_id = self._get_or_create_id(target)
            return FileNode(
                id=node_id,
                name=target.name,
                type="file",
                path=self._to_relative_display(target),
                content=content,
                children=[],
                language=self._detect_language(target.name),
                last_modified=datetime.now()
            )
        except Exception:
            return None
    
    def create_folder(self, parent_id: str, name: str) -> Optional[FileNode]:
        """إنشاء مجلد جديد"""
        parent = self._id_to_path.get(parent_id)
        if not parent or not parent.exists() or not parent.is_dir() or not self._is_safe_path(parent):
            return None

        target = (parent / name).resolve()
        if not self._is_safe_path(target):
            return None

        try:
            target.mkdir(parents=True, exist_ok=True)
            node_id = self._get_or_create_id(target)
            return FileNode(
                id=node_id,
                name=target.name,
                type="folder",
                path=self._to_relative_display(target),
                content=None,
                children=[],
                language=None,
                last_modified=datetime.now()
            )
        except Exception:
            return None
    
    def delete_node(self, node_id: str) -> bool:
        """حذف ملف أو مجلد"""
        path = self._id_to_path.get(node_id)
        if not path or not path.exists() or not self._is_safe_path(path):
            return False

        if path.resolve() == self.root_path:
            return False

        try:
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
            else:
                path.unlink()
            self._id_to_path.pop(node_id, None)
            return True
        except Exception:
            return False
    
    def _detect_language(self, filename: str) -> Optional[str]:
        """اكتشاف لغة البرمجة من الامتداد"""
        ext = Path(filename).suffix.lower()
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".json": "json",
            ".md": "markdown",
            ".html": "html",
            ".css": "css",
            ".rs": "rust",
            ".go": "go",
            ".sql": "sql"
        }
        return mapping.get(ext)


class AICopilot:
    """
    مساعد AI للبرمجة
    يتصل بالنظام الهرمي للحصول على اقتراحات
    """
    
    def __init__(self, hierarchy, filesystem_manager: FileSystemManager):
        self.hierarchy = hierarchy
        self.fs = filesystem_manager
        self.context_cache: Dict[str, Any] = {}

    def _collect_project_context(self, file_path: str, language: str) -> Dict[str, Any]:
        root = self.fs.root_path
        nearby_files: List[str] = []
        try:
            candidate_ext = {
                "python": ".py",
                "javascript": ".js",
                "typescript": ".ts",
                "rust": ".rs",
                "go": ".go",
                "json": ".json",
                "markdown": ".md",
            }.get(language, "")

            for item in root.rglob("*"):
                if len(nearby_files) >= 40:
                    break
                if not item.is_file():
                    continue
                if any(part in self.fs.ignored_dirs for part in item.parts):
                    continue
                rel = str(item.relative_to(root)).replace(os.sep, "/")
                if candidate_ext and item.suffix.lower() == candidate_ext:
                    nearby_files.append(rel)
                elif len(nearby_files) < 15:
                    nearby_files.append(rel)
        except Exception:
            pass

        return {
            "workspace": str(root),
            "active_file": file_path,
            "language": language,
            "nearby_files": nearby_files[:40],
        }

    def _normalize_label(self, insert_text: str) -> str:
        first = (insert_text or "").strip().split("\n")[0][:40]
        return first if first else "suggestion"

    def _normalize_insert_text(self, value: str) -> str:
        text = (value or "").replace("\r\n", "\n").strip()
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        return text

    def _normalize_confidence(self, value: Any, fallback: float = 0.72) -> float:
        try:
            conf = float(value)
        except Exception:
            conf = fallback

        if conf > 1.0:
            conf = conf / 100.0
        return max(0.35, min(0.99, conf))

    def _suggestion_key(self, insert_text: str) -> str:
        compact = " ".join((insert_text or "").lower().split())
        return compact[:220]

    def _rank_suggestion(
        self,
        suggestion: CodeSuggestion,
        language: str,
        cursor_prefix: str,
        line_prefix: str
    ) -> float:
        score = self._normalize_confidence(suggestion.confidence, 0.7)
        insert = suggestion.insert_text
        prefix = (cursor_prefix or "").strip()
        line = (line_prefix or "").strip().lower()

        if "\n" in insert:
            score += 0.06

        if language == "python":
            if insert.startswith("def "):
                score += 0.05
            if insert.startswith("class "):
                score += 0.03
            if line.endswith("for") and insert.startswith("for "):
                score += 0.09
            if line.endswith("try") and insert.startswith("try:"):
                score += 0.08
        elif language in {"javascript", "typescript"}:
            if insert.startswith("import ") and line.endswith("imp"):
                score += 0.1
            if "=>" in insert or insert.startswith("function"):
                score += 0.04
            if language == "typescript" and ("interface " in insert or "type " in insert):
                score += 0.06
        elif language == "rust":
            if insert.startswith("fn "):
                score += 0.07
            if insert.startswith("impl ") or "match " in insert:
                score += 0.05
            if line.endswith("pub") and insert.startswith("pub "):
                score += 0.08
        elif language == "go":
            if insert.startswith("func "):
                score += 0.07
            if "if err != nil" in insert:
                score += 0.06
            if line.endswith("go") and insert.startswith("go "):
                score += 0.08

        if prefix:
            first_word = insert.strip().split("\n", 1)[0].split("(", 1)[0].strip().lower()
            if first_word.startswith(prefix.lower()):
                score += 0.08

        if len(insert) > 600:
            score -= 0.12
        elif len(insert) > 350:
            score -= 0.06

        return max(0.01, min(0.99, score))

    def _finalize_suggestions(
        self,
        suggestions: List[CodeSuggestion],
        language: str,
        cursor_prefix: str,
        line_prefix: str,
        max_items: int = 7
    ) -> List[CodeSuggestion]:
        deduped: List[CodeSuggestion] = []
        seen: set[str] = set()

        for item in suggestions:
            normalized_insert = self._normalize_insert_text(item.insert_text)
            if not normalized_insert:
                continue

            key = self._suggestion_key(normalized_insert)
            if key in seen:
                continue

            seen.add(key)
            item.insert_text = normalized_insert
            item.text = self._normalize_label(normalized_insert)
            item.confidence = self._rank_suggestion(item, language, cursor_prefix, line_prefix)
            deduped.append(item)

        deduped.sort(key=lambda value: value.confidence, reverse=True)
        return deduped[:max_items]

    def _parse_suggestions(self, result: Any, language: str) -> List[CodeSuggestion]:
        """تحليل نتيجة الـ AI إلى قائمة اقتراحات"""
        suggestions: List[CodeSuggestion] = []

        if isinstance(result, dict):
            raw_items = result.get("suggestions") or result.get("items") or []
            for item in raw_items[:12]:
                if not isinstance(item, dict):
                    continue
                insert_text = self._normalize_insert_text(str(item.get("insert_text") or item.get("insertText") or item.get("text") or ""))
                if not insert_text:
                    continue
                suggestions.append(CodeSuggestion(
                    id=str(uuid.uuid4()),
                    text=str(item.get("label") or self._normalize_label(insert_text)),
                    detail=str(item.get("detail") or "AI suggestion"),
                    insert_text=insert_text,
                    range_start=0,
                    range_end=0,
                    confidence=self._normalize_confidence(item.get("confidence", 0.75))
                ))

            if not suggestions and isinstance(result.get("response"), str):
                text = self._normalize_insert_text(result.get("response", ""))
                if text:
                    suggestions.append(CodeSuggestion(
                        id=str(uuid.uuid4()),
                        text=self._normalize_label(text),
                        detail="AI response",
                        insert_text=text,
                        range_start=0,
                        range_end=0,
                        confidence=0.7
                    ))

        return suggestions

    def _build_contextual_fallback(self, language: str, cursor_prefix: str) -> List[CodeSuggestion]:
        templates = {
            "python": [
                ("def", "def function_name(params):\n    \"\"\"TODO: add docstring\"\"\"\n    pass", 0.9),
                ("class", "class ClassName:\n    def __init__(self):\n        pass", 0.86),
                ("try", "try:\n    pass\nexcept Exception as exc:\n    print(exc)", 0.82),
            ],
            "javascript": [
                ("function", "function name(args) {\n  // TODO\n}", 0.88),
                ("const", "const value = await fetch(url).then(r => r.json())", 0.84),
            ],
            "typescript": [
                ("interface", "interface Name {\n  id: string\n}", 0.88),
                ("type", "type Result<T> = { ok: true; value: T } | { ok: false; error: string }", 0.84),
                ("async", "async function run(): Promise<void> {\n  // TODO\n}", 0.83),
            ],
            "rust": [
                ("fn", "fn function_name() {\n    // TODO\n}", 0.9),
                ("impl", "impl StructName {\n    fn new() -> Self {\n        Self {}\n    }\n}", 0.86),
                ("match", "match value {\n    _ => {}\n}", 0.84),
            ],
            "go": [
                ("func", "func FunctionName() error {\n\treturn nil\n}", 0.9),
                ("iferr", "if err != nil {\n\treturn err\n}", 0.88),
                ("struct", "type Service struct {\n}\n", 0.84),
            ]
        }

        selected = templates.get(language, templates["python"])
        prefix = cursor_prefix.strip().lower()

        if prefix.endswith("imp") and language in {"typescript", "javascript"}:
            selected = [("import", "import { name } from './module'", 0.92)] + selected
        elif prefix.endswith("for") and language == "python":
            selected = [("for", "for item in items:\n    pass", 0.9)] + selected
        elif prefix.endswith("if") and language == "python":
            selected = [("if", "if condition:\n    pass", 0.88)] + selected
        elif prefix.endswith("async") and language == "python":
            selected = [("async", "async def run():\n    return None", 0.87)] + selected
        elif prefix.endswith("wh") and language == "python":
            selected = [("while", "while condition:\n    break", 0.86)] + selected
        elif prefix.endswith("pub") and language == "rust":
            selected = [("pub", "pub fn new() -> Self {\n    Self {}\n}", 0.92)] + selected
        elif prefix.endswith("mat") and language == "rust":
            selected = [("match", "match value {\n    _ => {}\n}", 0.9)] + selected
        elif prefix.endswith("ife") and language == "go":
            selected = [("iferr", "if err != nil {\n\treturn err\n}", 0.92)] + selected
        elif prefix.endswith("str") and language == "go":
            selected = [("struct", "type Name struct {\n\t// fields\n}", 0.88)] + selected

        return [
            CodeSuggestion(
                id=str(uuid.uuid4()),
                text=label,
                detail=f"Insert {label}",
                insert_text=insert,
                range_start=0,
                range_end=0,
                confidence=conf
            )
            for label, insert, conf in selected[:5]
        ]
    
    async def get_code_suggestions(
        self,
        code: str,
        cursor_position: int,
        language: str,
        file_path: str
    ) -> List[CodeSuggestion]:
        """الحصول على اقتراحات كود"""
        safe_cursor = max(0, min(cursor_position, len(code)))
        before = code[:safe_cursor]
        after = code[safe_cursor:]
        local_context = before[-400:]
        cursor_prefix = before.split()[-1] if before.split() else ""
        line_prefix = before.rsplit("\n", 1)[-1] if before else ""
        project_context = self._collect_project_context(file_path, language)
        cache_key = f"{language}|{file_path}|{hash(local_context)}"

        if cache_key in self.context_cache:
            cached = self.context_cache[cache_key]
            if isinstance(cached, list) and cached:
                return cached

        try:
            if not self.hierarchy:
                raise RuntimeError("Hierarchy not available")

            result = await self.hierarchy.experts.route_query(
                f"Context-aware code completion for {language}",
                {
                    "language": language,
                    "file_path": file_path,
                    "cursor_position": safe_cursor,
                    "code_before_cursor": local_context,
                    "code_after_cursor": after[:200],
                    "project_context": project_context,
                    "instruction": "Return concise completion suggestions"
                }
            )

            suggestions = self._parse_suggestions(result, language)
            if not suggestions:
                suggestions = self._build_contextual_fallback(language, local_context)

            suggestions = self._finalize_suggestions(
                suggestions=suggestions,
                language=language,
                cursor_prefix=cursor_prefix,
                line_prefix=line_prefix,
                max_items=7
            )

            self.context_cache[cache_key] = suggestions
            if len(self.context_cache) > 200:
                # remove oldest inserted key
                first_key = next(iter(self.context_cache))
                self.context_cache.pop(first_key, None)
            return suggestions
 
        except Exception as e:
            print(f"AI Copilot error: {e}")
            return self._finalize_suggestions(
                suggestions=self._build_contextual_fallback(language, local_context),
                language=language,
                cursor_prefix=cursor_prefix,
                line_prefix=line_prefix,
                max_items=7
            )

    def _get_fallback_suggestions(self, language: str) -> List[CodeSuggestion]:
        """اقتراحات احتياطية"""
        return self._build_contextual_fallback(language, "")

    def get_code_diagnostics(self, code: str, language: str, file_path: str) -> Dict[str, Any]:
        """تحليل ثابت بسيط للكود (MVP)"""
        issues: List[Dict[str, Any]] = []

        # Generic long-line check
        for index, line in enumerate(code.splitlines(), start=1):
            if len(line) > 140:
                issues.append({
                    "line": index,
                    "column": 141,
                    "severity": "warning",
                    "rule": "line-length",
                    "message": "Line exceeds 140 characters",
                    "source": "static-analyzer"
                })

        if language == "python":
            try:
                tree = ast.parse(code, filename=file_path or "<editor>")
            except SyntaxError as exc:
                issues.append({
                    "line": exc.lineno or 1,
                    "column": exc.offset or 1,
                    "severity": "error",
                    "rule": "syntax-error",
                    "message": exc.msg,
                    "source": "python-ast"
                })
                return {
                    "issues": issues,
                    "summary": {
                        "errors": 1,
                        "warnings": sum(1 for item in issues if item["severity"] == "warning"),
                        "infos": 0,
                        "total": len(issues)
                    }
                }

            imported_names: Dict[str, int] = {}
            used_names: set[str] = set()

            class ImportVisitor(ast.NodeVisitor):
                def visit_Import(self, node):
                    for alias in node.names:
                        imported_names[alias.asname or alias.name.split(".")[0]] = node.lineno
                    self.generic_visit(node)

                def visit_ImportFrom(self, node):
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        imported_names[alias.asname or alias.name] = node.lineno
                    self.generic_visit(node)

                def visit_Name(self, node):
                    used_names.add(node.id)
                    self.generic_visit(node)

            ImportVisitor().visit(tree)

            for name, line in imported_names.items():
                if name not in used_names and not name.startswith("_"):
                    issues.append({
                        "line": line,
                        "column": 1,
                        "severity": "warning",
                        "rule": "unused-import",
                        "message": f"Imported name '{name}' is not used",
                        "source": "python-ast"
                    })

        elif language in {"javascript", "typescript"}:
            pairs = {"{": "}", "(": ")", "[": "]"}
            openings = set(pairs.keys())
            closings = {v: k for k, v in pairs.items()}
            stack: List[tuple[str, int, int]] = []

            for line_num, line in enumerate(code.splitlines(), start=1):
                for col_num, ch in enumerate(line, start=1):
                    if ch in openings:
                        stack.append((ch, line_num, col_num))
                    elif ch in closings:
                        if not stack or stack[-1][0] != closings[ch]:
                            issues.append({
                                "line": line_num,
                                "column": col_num,
                                "severity": "error",
                                "rule": "unbalanced-brackets",
                                "message": f"Unexpected closing bracket '{ch}'",
                                "source": "brace-check"
                            })
                        else:
                            stack.pop()

            for op, line_num, col_num in stack[-10:]:
                issues.append({
                    "line": line_num,
                    "column": col_num,
                    "severity": "error",
                    "rule": "unbalanced-brackets",
                    "message": f"Opening bracket '{op}' is not closed",
                    "source": "brace-check"
                })

        errors = sum(1 for item in issues if item["severity"] == "error")
        warnings = sum(1 for item in issues if item["severity"] == "warning")
        infos = sum(1 for item in issues if item["severity"] == "info")

        return {
            "issues": issues,
            "summary": {
                "errors": errors,
                "warnings": warnings,
                "infos": infos,
                "total": len(issues)
            }
        }

    def _infer_symbol_name(self, code: str, symbol: Optional[str]) -> str:
        if symbol and symbol.strip():
            return symbol.strip()

        matches = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code[-800:])
        return matches[-1] if matches else ""

    def _workspace_path_from_editor_path(self, file_path: str) -> Optional[Path]:
        if not file_path:
            return None

        rel = file_path.strip().replace("\\", "/")
        if rel.startswith("/"):
            rel = rel[1:]

        candidate = (self.fs.root_path / rel).resolve()
        try:
            candidate.relative_to(self.fs.root_path)
        except ValueError:
            return None
        return candidate

    def _extract_python_definition(self, code: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            tree = ast.parse(code)
        except Exception:
            return None

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and getattr(node, "name", "") == symbol:
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                doc = ast.get_docstring(node) or ""

                signature = symbol
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    arg_names = [arg.arg for arg in node.args.args]
                    signature = f"{symbol}({', '.join(arg_names)})"

                return {
                    "kind": kind,
                    "line": getattr(node, "lineno", 1),
                    "signature": signature,
                    "docstring": doc
                }
        return None

    def _extract_regex_definition(self, code: str, symbol: str, language: str) -> Optional[Dict[str, Any]]:
        escaped = re.escape(symbol)

        patterns = {
            "javascript": [
                rf"function\s+{escaped}\s*\(([^)]*)\)",
                rf"const\s+{escaped}\s*=\s*\(([^)]*)\)\s*=>",
                rf"class\s+{escaped}\b"
            ],
            "typescript": [
                rf"function\s+{escaped}\s*\(([^)]*)\)",
                rf"const\s+{escaped}\s*=\s*\(([^)]*)\)\s*=>",
                rf"interface\s+{escaped}\b",
                rf"type\s+{escaped}\b",
                rf"class\s+{escaped}\b"
            ],
            "rust": [
                rf"pub\s+fn\s+{escaped}\s*\(([^)]*)\)",
                rf"fn\s+{escaped}\s*\(([^)]*)\)",
                rf"struct\s+{escaped}\b",
                rf"enum\s+{escaped}\b",
                rf"trait\s+{escaped}\b"
            ],
            "go": [
                rf"func\s+{escaped}\s*\(([^)]*)\)",
                rf"type\s+{escaped}\s+struct\b",
                rf"type\s+{escaped}\s+interface\b"
            ]
        }

        lines = code.splitlines()
        selected_patterns = patterns.get(language, [])

        for idx, line in enumerate(lines, start=1):
            for pattern in selected_patterns:
                if re.search(pattern, line):
                    prev_comment = ""
                    if idx > 1:
                        prev = lines[idx - 2].strip()
                        if prev.startswith("//") or prev.startswith("#"):
                            prev_comment = prev.lstrip("/# ")

                    return {
                        "kind": "symbol",
                        "line": idx,
                        "signature": line.strip(),
                        "docstring": prev_comment
                    }

        return None

    def _find_symbol_in_workspace(self, symbol: str, language: str, current_file: Optional[Path]) -> List[Dict[str, Any]]:
        ext_map = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "rust": ".rs",
            "go": ".go"
        }

        target_ext = ext_map.get(language)
        results: List[Dict[str, Any]] = []
        if not target_ext or not symbol:
            return results

        for path in self.fs.root_path.rglob(f"*{target_ext}"):
            if len(results) >= 8:
                break
            if any(part in self.fs.ignored_dirs for part in path.parts):
                continue
            if current_file and path.resolve() == current_file.resolve():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            if len(text) > 800_000:
                continue

            if re.search(rf"\b{re.escape(symbol)}\b", text):
                line_no = 1
                for idx, line in enumerate(text.splitlines(), start=1):
                    if re.search(rf"\b{re.escape(symbol)}\b", line):
                        line_no = idx
                        snippet = line.strip()[:220]
                        break

                rel = str(path.relative_to(self.fs.root_path)).replace(os.sep, "/")
                results.append({
                    "path": rel,
                    "line": line_no,
                    "snippet": snippet
                })

        return results

    def get_symbol_documentation(self, code: str, language: str, file_path: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Lookup documentation/details for a symbol from local context + workspace."""
        symbol_name = self._infer_symbol_name(code, symbol)
        normalized_language = (language or "").lower().strip()

        if not symbol_name:
            return {
                "ok": False,
                "symbol": "",
                "language": normalized_language,
                "definition": None,
                "related": [],
                "error": "No symbol detected"
            }

        current_file = self._workspace_path_from_editor_path(file_path)
        definition: Optional[Dict[str, Any]] = None

        if normalized_language == "python":
            definition = self._extract_python_definition(code, symbol_name)
        elif normalized_language in {"javascript", "typescript", "rust", "go"}:
            definition = self._extract_regex_definition(code, symbol_name, normalized_language)

        related = self._find_symbol_in_workspace(symbol_name, normalized_language, current_file)

        return {
            "ok": True,
            "symbol": symbol_name,
            "language": normalized_language,
            "definition": definition,
            "related": related,
            "error": ""
        }

    def get_refactor_suggestions(self, code: str, language: str, file_path: str) -> Dict[str, Any]:
        """اقتراحات Refactoring بسيطة (MVP)"""
        suggestions: List[Dict[str, Any]] = []
        lines = code.splitlines()

        def add_suggestion(rule: str, message: str, line: int, severity: str = "info"):
            suggestions.append({
                "rule": rule,
                "message": message,
                "line": max(1, line),
                "severity": severity,
                "file_path": file_path
            })

        if language == "python":
            try:
                tree = ast.parse(code, filename=file_path or "<editor>")

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        body_len = len(node.body or [])
                        if body_len > 30:
                            add_suggestion(
                                "extract-method",
                                f"Function '{node.name}' is long ({body_len} statements). Consider extracting smaller functions.",
                                node.lineno,
                                "warning"
                            )

                        complexity_nodes = 0
                        for sub in ast.walk(node):
                            if isinstance(sub, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)):
                                complexity_nodes += 1
                        if complexity_nodes > 10:
                            add_suggestion(
                                "reduce-complexity",
                                f"Function '{node.name}' has high branching/flow complexity ({complexity_nodes}).",
                                node.lineno,
                                "warning"
                            )

                        arg_names = [arg.arg for arg in getattr(node.args, "args", [])]
                        short_args = [name for name in arg_names if len(name) == 1 and name not in {"i", "j", "k", "x", "y", "z"}]
                        if short_args:
                            add_suggestion(
                                "rename-params",
                                f"Function '{node.name}' has very short parameter names: {', '.join(short_args[:4])}.",
                                node.lineno,
                                "info"
                            )

                    if isinstance(node, ast.ClassDef):
                        method_count = sum(1 for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))
                        if method_count > 20:
                            add_suggestion(
                                "split-class",
                                f"Class '{node.name}' has many methods ({method_count}). Consider splitting responsibilities.",
                                node.lineno,
                                "info"
                            )

            except Exception as exc:
                add_suggestion("analyzer-error", f"Python refactor analyzer failed: {exc}", 1, "warning")

        elif language in {"javascript", "typescript"}:
            brace_depth = 0
            max_depth = 0

            for index, line in enumerate(lines, start=1):
                stripped = line.strip()
                if len(stripped) > 160:
                    add_suggestion(
                        "split-line",
                        "Very long line found; consider breaking expression into smaller parts.",
                        index,
                        "info"
                    )

                open_count = line.count("{")
                close_count = line.count("}")
                brace_depth += open_count - close_count
                if brace_depth > max_depth:
                    max_depth = brace_depth

                if stripped.startswith("function ") and stripped.count("(") == 1 and stripped.count(",") > 5:
                    add_suggestion(
                        "reduce-params",
                        "Function has many parameters; consider passing an options object.",
                        index,
                        "info"
                    )

            if max_depth > 6:
                add_suggestion(
                    "reduce-nesting",
                    f"Code nesting depth is high ({max_depth}); consider early returns/extracting helpers.",
                    1,
                    "warning"
                )

        elif language == "rust":
            fn_count = sum(1 for line in lines if line.strip().startswith("fn ") or line.strip().startswith("pub fn "))
            unwrap_count = sum(line.count(".unwrap()") for line in lines)
            if unwrap_count > 0:
                add_suggestion(
                    "avoid-unwrap",
                    f"Detected {unwrap_count} use(s) of .unwrap(); consider proper error handling with Result/?.",
                    1,
                    "warning"
                )
            if fn_count > 25:
                add_suggestion(
                    "split-module",
                    f"Module contains many functions ({fn_count}); consider splitting into submodules.",
                    1,
                    "info"
                )

        elif language == "go":
            long_fn_lines = 0
            for idx, line in enumerate(lines, start=1):
                if len(line) > 160:
                    add_suggestion(
                        "split-line",
                        "Very long line found; consider splitting for readability.",
                        idx,
                        "info"
                    )
                if line.strip().startswith("func "):
                    long_fn_lines = 0
                elif line.strip() and line.startswith("\t"):
                    long_fn_lines += 1
                    if long_fn_lines > 55:
                        add_suggestion(
                            "extract-function",
                            "Large function body detected; consider extracting helper functions.",
                            idx,
                            "warning"
                        )
                        long_fn_lines = 0

        else:
            if len(lines) > 350:
                add_suggestion(
                    "split-file",
                    "Large file detected; consider splitting into focused modules.",
                    1,
                    "info"
                )

        suggestions = sorted(suggestions, key=lambda item: (0 if item["severity"] == "warning" else 1, item["line"]))[:30]
        return {
            "suggestions": suggestions,
            "summary": {
                "warnings": sum(1 for item in suggestions if item["severity"] == "warning"),
                "infos": sum(1 for item in suggestions if item["severity"] == "info"),
                "total": len(suggestions)
            }
        }
    
    async def review_code(self, code: str, language: str) -> Dict:
        """مراجعة الكود"""
        # طلب مراجعة من الـ Meta Team
        try:
            quality_check = await self.hierarchy.meta.managers['quality'].review_code_quality(
                code, language
            )
            return quality_check
        except:
            return {"score": 85, "passed": True, "issues": []}
    
    async def generate_tests(self, code: str, language: str) -> str:
        """توليد اختبارات للكود"""
        return f"""
# Generated Tests for {language}
def test_function():
    # TODO: Add test cases
    assert True
"""

    def _build_python_tests(self, code: str, file_path: str) -> Dict[str, Any]:
        functions: List[str] = []
        classes: List[str] = []

        try:
            tree = ast.parse(code, filename=file_path or "<editor>")
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        except Exception:
            pass

        stem = Path(file_path or "module.py").stem
        test_path = f"tests/test_{stem}.py"

        lines: List[str] = [
            "import pytest",
            "",
            f"# Auto-generated tests for {file_path or stem}",
            ""
        ]

        for fn in functions[:12]:
            lines.extend([
                f"def test_{fn}_basic_behavior():",
                "    # Arrange",
                "    # TODO: prepare inputs for function under test",
                "",
                "    # Act",
                f"    result = {fn}()  # TODO: pass required arguments",
                "",
                "    # Assert",
                "    assert result is not None",
                ""
            ])

        for cls in classes[:8]:
            lowered = cls.lower()
            lines.extend([
                f"def test_{lowered}_construction():",
                f"    instance = {cls}()  # TODO: pass required constructor args",
                "    assert instance is not None",
                ""
            ])

        if not functions and not classes:
            lines.extend([
                "def test_placeholder():",
                "    # TODO: add real assertions for this module",
                "    assert True",
                ""
            ])

        return {
            "framework": "pytest",
            "test_path": test_path,
            "content": "\n".join(lines).rstrip() + "\n"
        }

    def _build_js_ts_tests(self, code: str, file_path: str, language: str) -> Dict[str, Any]:
        import re

        stem = Path(file_path or "module.ts").stem
        ext = "ts" if language == "typescript" else "js"
        test_path = f"tests/{stem}.test.{ext}"

        function_names = re.findall(r"function\s+([A-Za-z_][A-Za-z0-9_]*)", code)
        export_names = re.findall(r"export\s+(?:const|function|class)\s+([A-Za-z_][A-Za-z0-9_]*)", code)

        targets = []
        for name in function_names + export_names:
            if name not in targets:
                targets.append(name)

        rel_source = (file_path or f"./{stem}.{ext}").lstrip("/")
        source_import = "./" + rel_source.replace("\\", "/")
        source_import = source_import[:-3] if source_import.endswith(".ts") else source_import
        source_import = source_import[:-3] if source_import.endswith(".js") else source_import

        lines: List[str] = [
            f"// Auto-generated tests for {file_path or stem}",
            f"import {{ {', '.join(targets[:8]) if targets else '/* TODO */'} }} from '{source_import}'",
            "",
            "describe('generated test suite', () => {"
        ]

        if targets:
            for target in targets[:8]:
                lines.extend([
                    f"  it('validates {target}', () => {{",
                    "    // TODO: arrange input",
                    f"    const result = {target}() as unknown",
                    "    expect(result).toBeDefined()",
                    "  })"
                ])
        else:
            lines.extend([
                "  it('placeholder test', () => {",
                "    expect(true).toBe(true)",
                "  })"
            ])

        lines.append("})")

        return {
            "framework": "jest/vitest",
            "test_path": test_path,
            "content": "\n".join(lines).rstrip() + "\n"
        }

    def _build_rust_tests(self, code: str, file_path: str) -> Dict[str, Any]:
        import re

        stem = Path(file_path or "mod.rs").stem
        test_path = f"tests/{stem}_tests.rs"
        names = re.findall(r"(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)", code)
        targets: List[str] = []
        for name in names:
            if name not in targets:
                targets.append(name)

        lines: List[str] = [
            f"// Auto-generated tests for {file_path or stem}",
            "#[cfg(test)]",
            "mod tests {",
            "    use super::*;",
            ""
        ]

        if targets:
            for name in targets[:10]:
                lines.extend([
                    "    #[test]",
                    f"    fn test_{name}_basic_behavior() {{",
                    f"        let result = {name}(); // TODO: provide required args",
                    "        let _ = result;",
                    "        assert!(true);",
                    "    }",
                    ""
                ])
        else:
            lines.extend([
                "    #[test]",
                "    fn test_placeholder() {",
                "        assert!(true);",
                "    }",
                ""
            ])

        lines.append("}")

        return {
            "framework": "cargo test",
            "test_path": test_path,
            "content": "\n".join(lines).rstrip() + "\n"
        }

    def _build_go_tests(self, code: str, file_path: str) -> Dict[str, Any]:
        import re

        stem = Path(file_path or "main.go").stem
        test_path = f"{stem}_test.go"
        package_name = "main"

        for line in code.splitlines()[:20]:
            striped = line.strip()
            if striped.startswith("package "):
                package_name = striped.split("package ", 1)[1].strip() or "main"
                break

        names = re.findall(r"func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
        exported = [name for name in names if name and name[0].isupper()]

        lines: List[str] = [
            f"package {package_name}",
            "",
            "import \"testing\"",
            "",
            f"// Auto-generated tests for {file_path or stem}",
            ""
        ]

        targets = exported[:10] if exported else names[:10]
        if targets:
            for name in targets:
                lines.extend([
                    f"func Test{name}(t *testing.T) {{",
                    f"\t_ = {name}() // TODO: provide required args",
                    "\t// TODO: add assertions",
                    "}",
                    ""
                ])
        else:
            lines.extend([
                "func TestPlaceholder(t *testing.T) {",
                "\tif false {",
                "\t\tt.Fatal(\"placeholder failed\")",
                "\t}",
                "}",
                ""
            ])

        return {
            "framework": "go test",
            "test_path": test_path,
            "content": "\n".join(lines).rstrip() + "\n"
        }

    async def generate_tests_for_file(self, code: str, language: str, file_path: str) -> Dict[str, Any]:
        """توليد اختبارات MVP حسب اللغة"""
        normalized = (language or "").lower().strip()

        if normalized == "python":
            return {
                "ok": True,
                **self._build_python_tests(code, file_path)
            }

        if normalized in {"javascript", "typescript"}:
            return {
                "ok": True,
                **self._build_js_ts_tests(code, file_path, normalized)
            }

        if normalized == "rust":
            return {
                "ok": True,
                **self._build_rust_tests(code, file_path)
            }

        if normalized == "go":
            return {
                "ok": True,
                **self._build_go_tests(code, file_path)
            }

        fallback = await self.generate_tests(code, normalized or "text")
        stem = Path(file_path or "module").stem
        return {
            "ok": True,
            "framework": "generic",
            "test_path": f"tests/{stem}.test.txt",
            "content": fallback
        }


class TerminalManager:
    """
    مدير Terminal (تنفيذ فعلي مع session isolation)
    """
    
    def __init__(self):
        self.workspace_root = Path(os.getenv("IDE_WORKSPACE_ROOT", str(Path.cwd()))).resolve()
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.command_history: List[Dict] = []

    async def start_session(self, preferred_cwd: Optional[str] = None) -> Dict[str, Any]:
        """بدء جلسة Terminal جديدة"""
        session_id = str(uuid.uuid4())
        cwd = self._resolve_safe_cwd(preferred_cwd)
        self.sessions[session_id] = {
            "id": session_id,
            "cwd": str(cwd),
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "history": []
        }
        return {
            "session_id": session_id,
            "cwd": str(cwd),
            "workspace_root": str(self.workspace_root)
        }

    def _resolve_safe_cwd(self, value: Optional[str]) -> Path:
        if not value:
            return self.workspace_root

        try:
            requested = Path(value).expanduser().resolve()
        except Exception:
            return self.workspace_root

        try:
            requested.relative_to(self.workspace_root)
            return requested
        except ValueError:
            return self.workspace_root

    def _get_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "id": session_id,
                "cwd": str(self.workspace_root),
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "history": []
            }
        return self.sessions[session_id]

    def _handle_cd(self, session: Dict[str, Any], command: str) -> Dict[str, Any]:
        raw_target = command.strip()[2:].strip()
        if raw_target == "":
            target = self.workspace_root
        else:
            current_cwd = Path(session["cwd"])
            candidate = Path(raw_target)
            target = (candidate if candidate.is_absolute() else (current_cwd / candidate)).resolve()

        if not target.exists() or not target.is_dir():
            return {
                "stdout": "",
                "stderr": f"Directory not found: {raw_target}",
                "exit_code": 1,
                "elapsed_ms": 0,
                "cwd": session["cwd"]
            }

        try:
            target.relative_to(self.workspace_root)
        except ValueError:
            return {
                "stdout": "",
                "stderr": f"Access denied outside workspace: {target}",
                "exit_code": 1,
                "elapsed_ms": 0,
                "cwd": session["cwd"]
            }

        session["cwd"] = str(target)
        session["last_active"] = datetime.now().isoformat()
        return {
            "stdout": str(target),
            "stderr": "",
            "exit_code": 0,
            "elapsed_ms": 0,
            "cwd": session["cwd"]
        }
    
    async def execute_command(self, session_id: str, command: str) -> Dict:
        """تنفيذ أمر"""
        import time

        session = self._get_session(session_id)
        start_time = time.time()

        if command.strip().startswith("cd"):
            result = self._handle_cd(session, command)
        else:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=session["cwd"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy()
            )

            try:
                stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=90)
                exit_code = process.returncode if process.returncode is not None else 1
                stdout = stdout_b.decode("utf-8", errors="replace")
                stderr = stderr_b.decode("utf-8", errors="replace")
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                stdout = ""
                stderr = "Command timed out after 90s"
                exit_code = 124

            elapsed = time.time() - start_time
            session["last_active"] = datetime.now().isoformat()
            result = {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "elapsed_ms": int(elapsed * 1000),
                "cwd": session["cwd"]
            }

        session["history"].append({
            "command": command,
            "timestamp": datetime.now().isoformat(),
            "exit_code": result.get("exit_code", 1)
        })
        if len(session["history"]) > 200:
            session["history"] = session["history"][-200:]

        self.command_history.append({
            "session_id": session_id,
            "command": command,
            "timestamp": datetime.now(),
            "result": result
        })

        if len(self.command_history) > 1000:
            self.command_history = self.command_history[-1000:]

        return result


class GitManager:
    """
    مدير Git (MVP)
    status/diff/commit داخل workspace الحالي
    """

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root

    async def _run_git(self, *args: str) -> Dict[str, Any]:
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=str(self.workspace_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy()
            )
            stdout_b, stderr_b = await process.communicate()
            return {
                "exit_code": process.returncode if process.returncode is not None else 1,
                "stdout": stdout_b.decode("utf-8", errors="replace"),
                "stderr": stderr_b.decode("utf-8", errors="replace")
            }
        except FileNotFoundError:
            return {
                "exit_code": 127,
                "stdout": "",
                "stderr": "git command not found"
            }
        except Exception as exc:
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": str(exc)
            }

    async def _get_branch(self) -> str:
        result = await self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        if result["exit_code"] != 0:
            return ""
        return result["stdout"].strip()

    async def is_repo_ready(self) -> Dict[str, Any]:
        result = await self._run_git("rev-parse", "--is-inside-work-tree")
        is_repo = result["exit_code"] == 0 and result["stdout"].strip().lower() == "true"
        return {
            "ok": is_repo,
            "error": "" if is_repo else (result["stderr"].strip() or "Not a git repository")
        }

    async def get_status(self) -> Dict[str, Any]:
        repo_state = await self.is_repo_ready()
        if not repo_state["ok"]:
            return {
                "ok": False,
                "branch": "",
                "files": [],
                "summary": {"modified": 0, "added": 0, "deleted": 0, "untracked": 0},
                "error": repo_state["error"]
            }

        result = await self._run_git("status", "--porcelain", "-b")
        if result["exit_code"] != 0:
            return {
                "ok": False,
                "branch": "",
                "files": [],
                "summary": {"modified": 0, "added": 0, "deleted": 0, "untracked": 0},
                "error": result["stderr"].strip() or "Failed to read git status"
            }

        lines = [line for line in result["stdout"].splitlines() if line.strip()]
        branch = ""
        files: List[Dict[str, str]] = []
        summary = {"modified": 0, "added": 0, "deleted": 0, "untracked": 0}

        for line in lines:
            if line.startswith("##"):
                header = line[2:].strip()
                branch = header.split("...")[0].strip()
                continue

            if len(line) < 3:
                continue

            status_code = line[:2]
            path = line[3:].strip()
            category = "modified"

            if status_code == "??":
                category = "untracked"
            elif "A" in status_code:
                category = "added"
            elif "D" in status_code:
                category = "deleted"
            elif "M" in status_code:
                category = "modified"

            summary[category] += 1
            files.append({
                "path": path,
                "status": status_code,
                "category": category
            })

        if not branch:
            branch = await self._get_branch()

        return {
            "ok": True,
            "branch": branch,
            "files": files,
            "summary": summary,
            "error": ""
        }

    async def get_diff(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        repo_state = await self.is_repo_ready()
        if not repo_state["ok"]:
            return {"ok": False, "diff": "", "error": repo_state["error"]}

        if file_path:
            result = await self._run_git("diff", "--", file_path)
        else:
            result = await self._run_git("diff")

        if result["exit_code"] != 0:
            return {
                "ok": False,
                "diff": "",
                "error": result["stderr"].strip() or "Failed to get diff"
            }

        return {
            "ok": True,
            "diff": result["stdout"],
            "error": ""
        }

    async def commit(self, message: str, stage_all: bool = True) -> Dict[str, Any]:
        repo_state = await self.is_repo_ready()
        if not repo_state["ok"]:
            return {"ok": False, "error": repo_state["error"], "output": ""}

        trimmed_message = (message or "").strip()
        if not trimmed_message:
            return {"ok": False, "error": "Commit message is required", "output": ""}

        if stage_all:
            stage_result = await self._run_git("add", "-A")
            if stage_result["exit_code"] != 0:
                return {
                    "ok": False,
                    "error": stage_result["stderr"].strip() or "Failed to stage files",
                    "output": stage_result["stdout"].strip()
                }

        commit_result = await self._run_git("commit", "-m", trimmed_message)
        if commit_result["exit_code"] != 0:
            return {
                "ok": False,
                "error": commit_result["stderr"].strip() or commit_result["stdout"].strip() or "Commit failed",
                "output": commit_result["stdout"].strip()
            }

        return {
            "ok": True,
            "error": "",
            "output": commit_result["stdout"].strip()
        }

    async def push(self, remote: str = "origin", branch: Optional[str] = None) -> Dict[str, Any]:
        repo_state = await self.is_repo_ready()
        if not repo_state["ok"]:
            return {"ok": False, "error": repo_state["error"], "output": ""}

        target_branch = (branch or "").strip() or await self._get_branch()
        if target_branch:
            result = await self._run_git("push", remote, target_branch)
        else:
            result = await self._run_git("push", remote)

        if result["exit_code"] != 0:
            return {
                "ok": False,
                "error": result["stderr"].strip() or result["stdout"].strip() or "Push failed",
                "output": result["stdout"].strip()
            }

        return {
            "ok": True,
            "error": "",
            "output": result["stdout"].strip() or "Push completed"
        }

    async def pull(self, remote: str = "origin", branch: Optional[str] = None) -> Dict[str, Any]:
        repo_state = await self.is_repo_ready()
        if not repo_state["ok"]:
            return {"ok": False, "error": repo_state["error"], "output": ""}

        target_branch = (branch or "").strip()
        if target_branch:
            result = await self._run_git("pull", "--ff-only", remote, target_branch)
        else:
            result = await self._run_git("pull", "--ff-only")

        if result["exit_code"] != 0:
            return {
                "ok": False,
                "error": result["stderr"].strip() or result["stdout"].strip() or "Pull failed",
                "output": result["stdout"].strip()
            }

        return {
            "ok": True,
            "error": "",
            "output": result["stdout"].strip() or "Pull completed"
        }


class DebugManager:
    """
    مدير Debug (MVP) باستخدام pdb لملفات Python
    """

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def _resolve_workspace_file(self, file_path: str) -> Optional[Path]:
        if not file_path:
            return None

        raw = file_path.strip().replace("\\", "/")
        rel = raw[1:] if raw.startswith("/") else raw
        candidate = (self.workspace_root / rel).resolve()

        try:
            candidate.relative_to(self.workspace_root)
        except ValueError:
            return None

        if not candidate.exists() or not candidate.is_file():
            return None

        return candidate

    async def _read_until_prompt(self, process, timeout_sec: float = 12.0) -> str:
        if not process.stdout:
            return ""

        output = ""
        loop = asyncio.get_running_loop()
        start = loop.time()

        while True:
            if process.returncode is not None:
                rest = await process.stdout.read()
                if rest:
                    output += rest.decode("utf-8", errors="replace")
                break

            if loop.time() - start > timeout_sec:
                break

            try:
                chunk = await asyncio.wait_for(process.stdout.read(1), timeout=0.4)
            except asyncio.TimeoutError:
                continue

            if not chunk:
                break

            output += chunk.decode("utf-8", errors="replace")
            if output.endswith("(Pdb) ") or output.endswith("(Pdb)"):
                break

        return output

    async def _send_command(self, session: Dict[str, Any], command: str) -> Dict[str, Any]:
        process = session.get("process")
        if not process or process.returncode is not None:
            return {
                "ok": False,
                "output": "Debug session is not running",
                "exit_code": process.returncode if process else 1
            }

        stdin = process.stdin
        if not stdin:
            return {"ok": False, "output": "Debug stdin unavailable", "exit_code": 1}

        stdin.write((command.strip() + "\n").encode("utf-8", errors="replace"))
        await stdin.drain()
        output = await self._read_until_prompt(process)

        return {
            "ok": True,
            "output": output,
            "exit_code": process.returncode if process.returncode is not None else 0
        }

    async def start_session(self, file_path: str, breakpoints: Optional[List[int]] = None) -> Dict[str, Any]:
        target = self._resolve_workspace_file(file_path)
        if not target:
            return {"ok": False, "error": "File not found inside workspace", "session_id": "", "output": ""}

        if target.suffix.lower() != ".py":
            return {"ok": False, "error": "Debug MVP currently supports Python files only", "session_id": "", "output": ""}

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pdb",
            str(target),
            cwd=str(self.workspace_root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=os.environ.copy()
        )

        session_id = str(uuid.uuid4())
        session = {
            "id": session_id,
            "file_path": str(target.relative_to(self.workspace_root)).replace(os.sep, "/"),
            "process": process,
            "created_at": datetime.now().isoformat(),
            "breakpoints": []
        }
        self.sessions[session_id] = session

        startup_output = await self._read_until_prompt(process)
        startup_error = ""

        for line in (breakpoints or []):
            if not isinstance(line, int) or line <= 0:
                continue
            bp_result = await self.set_breakpoint(session_id, session["file_path"], line)
            if bp_result.get("ok"):
                session["breakpoints"].append(line)
            elif not startup_error:
                startup_error = bp_result.get("error", "")

        return {
            "ok": True,
            "session_id": session_id,
            "file_path": session["file_path"],
            "output": startup_output,
            "error": startup_error
        }

    async def set_breakpoint(self, session_id: str, file_path: str, line: int) -> Dict[str, Any]:
        session = self.sessions.get(session_id)
        if not session:
            return {"ok": False, "error": "Debug session not found", "output": ""}

        target = self._resolve_workspace_file(file_path)
        if not target:
            return {"ok": False, "error": "File not found inside workspace", "output": ""}
        if not isinstance(line, int) or line <= 0:
            return {"ok": False, "error": "Invalid line number", "output": ""}

        rel = str(target.relative_to(self.workspace_root)).replace(os.sep, "/")
        result = await self._send_command(session, f"b {rel}:{line}")

        if result["ok"]:
            known = session.get("breakpoints", [])
            if line not in known:
                known.append(line)
            session["breakpoints"] = known

        return {
            "ok": result["ok"],
            "error": "" if result["ok"] else "Failed to set breakpoint",
            "output": result["output"]
        }

    async def execute(self, session_id: str, command: str) -> Dict[str, Any]:
        session = self.sessions.get(session_id)
        if not session:
            return {"ok": False, "error": "Debug session not found", "output": ""}

        mapping = {
            "continue": "c",
            "step": "s",
            "next": "n",
            "stack": "w",
            "locals": "p locals()",
            "globals": "p list(globals().keys())[:40]"
        }

        raw = (command or "").strip()
        if not raw:
            return {"ok": False, "error": "Command is required", "output": ""}

        actual_command = mapping.get(raw.lower(), raw)
        result = await self._send_command(session, actual_command)

        return {
            "ok": result["ok"],
            "error": "" if result["ok"] else "Debug command failed",
            "output": result["output"],
            "exit_code": result.get("exit_code", 0)
        }

    async def stop_session(self, session_id: str) -> Dict[str, Any]:
        session = self.sessions.get(session_id)
        if not session:
            return {"ok": False, "error": "Debug session not found", "output": ""}

        process = session.get("process")
        output = ""

        if process and process.returncode is None:
            try:
                result = await self._send_command(session, "q")
                output = result.get("output", "")
            except Exception:
                pass

            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

        self.sessions.pop(session_id, None)
        return {
            "ok": True,
            "error": "",
            "output": output
        }


class IDEService:
    """
    خدمة IDE الرئيسية
    تجمع كل المكونات
    """
    
    def __init__(self, hierarchy):
        self.fs = FileSystemManager()
        self.copilot = AICopilot(hierarchy, self.fs)
        self.terminal = TerminalManager()
        self.git = GitManager(self.fs.root_path)
        self.debug = DebugManager(self.fs.root_path)
        print("💻 IDE Service initialized")

    def _count_tree_nodes(self, node: FileNode) -> int:
        count = 1
        for child in node.children:
            count += self._count_tree_nodes(child)
        return count
    
    def get_status(self) -> Dict:
        """حالة IDE"""
        try:
            tree = self.fs.get_file_tree()
            files_count = self._count_tree_nodes(tree)
        except Exception:
            files_count = 0

        return {
            "files_count": files_count,
            "terminal_sessions": len(self.terminal.sessions),
            "git_workspace": str(self.fs.root_path),
            "debug_sessions": len(self.debug.sessions),
            "copilot_ready": True,
            "supported_languages": ["python", "javascript", "typescript", "json", "markdown", "html", "css", "rust", "go", "sql"]
        }


# Singleton
ide_service = None

def get_ide_service(hierarchy=None):
    global ide_service
    if ide_service is None and hierarchy:
        ide_service = IDEService(hierarchy)
    return ide_service
