//! Real PTY Terminal Component using xterm.js

import { useEffect, useRef, useState } from "react";
import { Terminal } from "xterm";
import { FitAddon } from "xterm-addon-fit";
import { WebLinksAddon } from "xterm-addon-web-links";
import "xterm/css/xterm.css";
import { Plus, X, Trash2, Copy, Terminal as TerminalIcon } from "lucide-react";
import { terminal } from "../../lib/tauri";

interface TerminalSession {
  id: string;
  name: string;
  shell: string;
  processId?: number;
}

function shellCandidates(): string[] {
  const platform = navigator.platform.toLowerCase();
  if (platform.includes("win")) {
    return ["pwsh", "powershell", "cmd"];
  }
  if (platform.includes("mac")) {
    return ["/bin/zsh", "/bin/bash", "zsh", "bash", "sh"];
  }
  return ["/bin/bash", "/bin/sh", "bash", "sh"];
}

export function RealTerminal() {
  const terminalRef = useRef<HTMLDivElement>(null);
  const xtermRef = useRef<Terminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const [sessions, setSessions] = useState<TerminalSession[]>([{ id: "default", name: "shell 1", shell: shellCandidates()[0] }]);
  const [activeSessionId, setActiveSessionId] = useState("default");

  const sessionsRef = useRef<TerminalSession[]>([]);
  const activeSessionRef = useRef("default");
  const pollersRef = useRef<Record<string, ReturnType<typeof setInterval>>>({});
  const outputBuffersRef = useRef<Record<string, string>>({});

  useEffect(() => {
    sessionsRef.current = sessions;
  }, [sessions]);

  useEffect(() => {
    activeSessionRef.current = activeSessionId;
  }, [activeSessionId]);

  const startPolling = (sessionId: string, processId: number) => {
    if (pollersRef.current[sessionId]) {
      clearInterval(pollersRef.current[sessionId]);
    }

    pollersRef.current[sessionId] = setInterval(async () => {
      try {
        const out = await terminal.readOutput(processId);
        const chunk = `${out.stdout || ""}${out.stderr || ""}`;
        if (!chunk) return;

        if (activeSessionRef.current === sessionId) {
          xtermRef.current?.write(chunk);
        } else {
          outputBuffersRef.current[sessionId] = `${outputBuffersRef.current[sessionId] || ""}${chunk}`;
        }
      } catch {
        const poller = pollersRef.current[sessionId];
        if (poller) {
          clearInterval(poller);
          delete pollersRef.current[sessionId];
        }
      }
    }, 120);
  };

  const stopSessionProcess = async (sessionId: string) => {
    const session = sessionsRef.current.find((s) => s.id === sessionId);
    if (session?.processId) {
      try {
        await terminal.kill(session.processId);
      } catch {
        // ignore
      }
    }

    const poller = pollersRef.current[sessionId];
    if (poller) {
      clearInterval(poller);
      delete pollersRef.current[sessionId];
    }
  };

  const ensureSessionProcess = async (sessionId: string) => {
    const session = sessionsRef.current.find((s) => s.id === sessionId);
    if (!session || session.processId) return;

    const candidates = [session.shell, ...shellCandidates().filter((c) => c !== session.shell)];
    let spawnedId: number | undefined;
    let usedShell = session.shell;

    for (const cmd of candidates) {
      try {
        const spawned = await terminal.spawn(cmd);
        spawnedId = spawned.process_id;
        usedShell = cmd;
        break;
      } catch {
        // try next shell
      }
    }

    if (!spawnedId) {
      xtermRef.current?.writeln("\r\n\x1b[31mFailed to start shell process\x1b[0m\r\n");
      return;
    }

    setSessions((prev) =>
      prev.map((s) =>
        s.id === sessionId
          ? { ...s, processId: spawnedId, shell: usedShell, name: s.name.replace(/^shell/, usedShell) }
          : s
      )
    );

    startPolling(sessionId, spawnedId);
  };

  useEffect(() => {
    if (!terminalRef.current || xtermRef.current) return;

    const term = new Terminal({
      theme: {
        background: "#1e1e1e",
        foreground: "#d4d4d4",
        cursor: "#d4d4d4",
        selectionBackground: "#264f78",
      },
      fontSize: 13,
      fontFamily: 'Menlo, Monaco, monospace',
      cursorBlink: true,
      scrollback: 10000,
    });

    const fitAddon = new FitAddon();
    term.loadAddon(fitAddon);
    term.loadAddon(new WebLinksAddon());
    term.open(terminalRef.current);
    fitAddon.fit();

    xtermRef.current = term;
    fitAddonRef.current = fitAddon;

    term.writeln("\x1b[1;32mBI-IDE Terminal v1.0\x1b[0m");
    term.writeln("\x1b[90mConnected to process backend\x1b[0m\r\n");

    // Handle input
    const dataDisposable = term.onData(async (data) => {
      const active = sessionsRef.current.find((s) => s.id === activeSessionRef.current);
      if (!active?.processId) {
        await ensureSessionProcess(activeSessionRef.current);
      }

      const activeAfterEnsure = sessionsRef.current.find((s) => s.id === activeSessionRef.current);
      if (!activeAfterEnsure?.processId) return;

      try {
        await terminal.writeInput(activeAfterEnsure.processId, data);
      } catch {
        term.writeln("\r\n\x1b[31mProcess input failed\x1b[0m\r\n");
      }
    });

    ensureSessionProcess("default");

    return () => {
      dataDisposable.dispose();
      Object.keys(pollersRef.current).forEach((id) => {
        clearInterval(pollersRef.current[id]);
      });
      pollersRef.current = {};
      sessionsRef.current.forEach((s) => {
        if (s.processId) {
          terminal.kill(s.processId).catch(() => undefined);
        }
      });
      term.dispose();
      xtermRef.current = null;
    };
  }, []);

  const createSession = () => {
    const id = `session-${Date.now()}`;
    const next = sessions.length + 1;
    const shell = shellCandidates()[0];
    setSessions([...sessions, { id, name: `shell ${next}`, shell }]);
    setActiveSessionId(id);
    xtermRef.current?.clear();
    ensureSessionProcess(id);
  };

  const closeSession = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (sessions.length <= 1) return;

    await stopSessionProcess(id);
    const newSessions = sessions.filter(s => s.id !== id);
    setSessions(newSessions);
    if (activeSessionId === id) {
      setActiveSessionId(newSessions[0].id);
      const buffered = outputBuffersRef.current[newSessions[0].id];
      xtermRef.current?.clear();
      if (buffered) {
        xtermRef.current?.write(buffered);
        outputBuffersRef.current[newSessions[0].id] = "";
      }
      ensureSessionProcess(newSessions[0].id);
    }
  };

  const switchSession = (id: string) => {
    setActiveSessionId(id);
    xtermRef.current?.clear();
    const session = sessions.find(s => s.id === id);
    xtermRef.current?.writeln(`\x1b[1;32mSession: ${session?.name} (${session?.shell})\x1b[0m\r\n`);

    const buffered = outputBuffersRef.current[id];
    if (buffered) {
      xtermRef.current?.write(buffered);
      outputBuffersRef.current[id] = "";
    }

    ensureSessionProcess(id);
  };

  useEffect(() => {
    const handleResize = () => fitAddonRef.current?.fit();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div className="h-full flex flex-col bg-[#1e1e1e]">
      {/* Tabs */}
      <div className="flex items-center bg-[#252526] border-b border-[#1e1e1e]">
        <div className="flex-1 flex overflow-x-auto">
          {sessions.map((session) => (
            <button
              key={session.id}
              onClick={() => switchSession(session.id)}
              className={`group flex items-center gap-2 px-3 py-2 text-sm border-r border-[#1e1e1e] min-w-[100px] max-w-[200px]
                ${session.id === activeSessionId ? "bg-[#1e1e1e] text-white" : "bg-[#2d2d2d] text-dark-400"}`}
            >
              <TerminalIcon className="w-3.5 h-3.5" />
              <span className="truncate flex-1">{session.name}</span>
              {sessions.length > 1 && (
                <button onClick={(e) => { void closeSession(session.id, e); }} className="p-0.5 hover:bg-dark-600 rounded opacity-0 group-hover:opacity-100">
                  <X className="w-3 h-3" />
                </button>
              )}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1 px-2">
          <button onClick={createSession} className="p-1.5 hover:bg-dark-700 rounded text-dark-400">
            <Plus className="w-4 h-4" />
          </button>
          <button onClick={() => xtermRef.current?.clear()} className="p-1.5 hover:bg-dark-700 rounded text-dark-400">
            <Trash2 className="w-4 h-4" />
          </button>
          <button onClick={() => navigator.clipboard.writeText(xtermRef.current?.getSelection() || "")} className="p-1.5 hover:bg-dark-700 rounded text-dark-400">
            <Copy className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Terminal */}
      <div ref={terminalRef} className="flex-1 p-2" style={{ background: "#1e1e1e" }} />
    </div>
  );
}
