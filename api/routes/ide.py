"""
IDE Routes - نقاط النهاية لبيئة التطوير
"""

from typing import Dict, Optional

from fastapi import APIRouter, HTTPException

from api.schemas import (
    CodeSuggestionRequest,
    CodeAnalysisRequest,
    RefactorSuggestRequest,
    TestGenerateRequest,
    SymbolDocumentationRequest,
    TerminalCommandRequest,
    TerminalSessionStartRequest,
    GitCommitRequest,
    GitSyncRequest,
    DebugStartRequest,
    DebugBreakpointRequest,
    DebugCommandRequest,
    DebugStopRequest,
)

router = APIRouter(prefix="/api/v1/ide", tags=["ide"])

# Service reference – set during startup
_ide_service = None


def set_ide_service(service):
    global _ide_service
    _ide_service = service


def _svc():
    if _ide_service is None:
        raise HTTPException(500, "IDE not initialized")
    return _ide_service


# ──────────── File System ────────────


@router.get("/files")
async def get_file_tree():
    """شجرة الملفات"""
    svc = _svc()

    def serialize_node(node):
        return {
            "id": node.id,
            "name": node.name,
            "type": node.type,
            "path": node.path,
            "language": node.language,
            "children": [serialize_node(c) for c in node.children],
        }

    tree = svc.fs.get_file_tree()
    return serialize_node(tree)


@router.get("/files/{file_id}")
async def get_file(file_id: str):
    """محتوى ملف"""
    content = _svc().fs.get_file_content(file_id)
    if content is None:
        raise HTTPException(404, "File not found")
    return {"id": file_id, "content": content}


@router.post("/files/{file_id}")
async def save_file(file_id: str, request: Dict):
    """حفظ ملف"""
    success = _svc().fs.save_file(file_id, request.get("content", ""))
    return {"success": success}


# ──────────── AI Copilot ────────────


@router.post("/copilot/suggest")
async def get_code_suggestions(request: CodeSuggestionRequest):
    """اقتراحات كود من AI"""
    suggestions = await _svc().copilot.get_code_suggestions(
        request.code, request.cursor_position, request.language, request.file_path,
    )
    return {
        "suggestions": [
            {
                "label": s.text,
                "detail": s.detail,
                "insertText": s.insert_text,
                "confidence": s.confidence,
            }
            for s in suggestions
        ]
    }


@router.post("/analysis")
async def get_code_analysis(request: CodeAnalysisRequest):
    """تحليل ثابت للكود"""
    return _svc().copilot.get_code_diagnostics(
        code=request.code, language=request.language, file_path=request.file_path,
    )


@router.post("/refactor/suggest")
async def get_refactor_suggestions(request: RefactorSuggestRequest):
    """اقتراحات Refactoring"""
    return _svc().copilot.get_refactor_suggestions(
        code=request.code, language=request.language, file_path=request.file_path,
    )


@router.post("/tests/generate")
async def generate_tests(request: TestGenerateRequest):
    """توليد اختبارات"""
    return await _svc().copilot.generate_tests_for_file(
        code=request.code, language=request.language, file_path=request.file_path,
    )


@router.post("/docs/symbol")
async def get_symbol_documentation(request: SymbolDocumentationRequest):
    """توثيق Symbol"""
    return _svc().copilot.get_symbol_documentation(
        code=request.code, language=request.language,
        file_path=request.file_path, symbol=request.symbol,
    )


# ──────────── Git ────────────


@router.get("/git/status")
async def get_git_status():
    return await _svc().git.get_status()


@router.get("/git/diff")
async def get_git_diff(path: Optional[str] = None):
    return await _svc().git.get_diff(path)


@router.post("/git/commit")
async def create_git_commit(request: GitCommitRequest):
    return await _svc().git.commit(message=request.message, stage_all=request.stage_all)


@router.post("/git/push")
async def push_git_changes(request: GitSyncRequest):
    return await _svc().git.push(remote=request.remote, branch=request.branch)


@router.post("/git/pull")
async def pull_git_changes(request: GitSyncRequest):
    return await _svc().git.pull(remote=request.remote, branch=request.branch)


# ──────────── Debug ────────────


@router.post("/debug/session/start")
async def start_debug_session(request: DebugStartRequest):
    return await _svc().debug.start_session(file_path=request.file_path, breakpoints=request.breakpoints)


@router.post("/debug/breakpoint")
async def set_debug_breakpoint(request: DebugBreakpointRequest):
    return await _svc().debug.set_breakpoint(
        session_id=request.session_id, file_path=request.file_path, line=request.line,
    )


@router.post("/debug/command")
async def execute_debug_command(request: DebugCommandRequest):
    return await _svc().debug.execute(session_id=request.session_id, command=request.command)


@router.post("/debug/session/stop")
async def stop_debug_session(request: DebugStopRequest):
    return await _svc().debug.stop_session(request.session_id)


# ──────────── Terminal ────────────


@router.post("/terminal/execute")
async def execute_terminal(request: TerminalCommandRequest):
    return await _svc().terminal.execute_command(request.session_id, request.command)


@router.post("/terminal/session/start")
async def start_terminal_session(request: TerminalSessionStartRequest):
    return await _svc().terminal.start_session(request.cwd)
