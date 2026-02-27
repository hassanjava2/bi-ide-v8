import { useState, useRef, useEffect } from "react";
import { 
  Terminal as TerminalIcon, 
  Trash2, 
  Maximize2, 
  Minimize2,
  X,
  Play,
  Square
} from "lucide-react";
import { useStore } from "../lib/store";
import { terminal } from "../lib/tauri";

export function Terminal() {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentProcess, setCurrentProcess] = useState<number | null>(null);
  const terminalRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  
  const { 
    currentWorkspace,
    terminalHistory,
    addTerminalOutput,
    clearTerminal,
  } = useStore();

  // Auto-scroll to bottom
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [terminalHistory, history]);

  // Focus input on mount
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  const executeCommand = async () => {
    if (!input.trim() || !currentWorkspace) return;

    const command = input.trim();
    setHistory((prev) => [...prev, `> ${command}`]);
    setInput("");

    // Check for built-in commands
    if (command === "clear") {
      clearTerminal();
      setHistory([]);
      return;
    }

    setIsRunning(true);

    try {
      // Parse command
      const parts = command.split(" ");
      const cmd = parts[0];
      const args = parts.slice(1);

      const result = await terminal.execute(
        cmd,
        args,
        currentWorkspace.path,
        undefined,
        60000 // 1 minute timeout
      );

      if (result.stdout) {
        setHistory((prev) => [...prev, result.stdout]);
      }
      if (result.stderr) {
        setHistory((prev) => [...prev, `Error: ${result.stderr}`]);
      }
      if (result.exit_code !== 0 && result.exit_code !== undefined) {
        setHistory((prev) => [...prev, `Exit code: ${result.exit_code}`]);
      }
    } catch (error) {
      setHistory((prev) => [...prev, `Error: ${error}`]);
    } finally {
      setIsRunning(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      executeCommand();
    }
  };

  const handleKill = async () => {
    if (currentProcess) {
      await terminal.kill(currentProcess);
      setCurrentProcess(null);
      setIsRunning(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-dark-950">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-dark-800 border-b border-dark-700">
        <div className="flex items-center gap-2">
          <TerminalIcon className="w-4 h-4 text-dark-400" />
          <span className="text-sm text-dark-300">Terminal</span>
          {currentWorkspace && (
            <span className="text-xs text-dark-500">
              ({currentWorkspace.name})
            </span>
          )}
        </div>
        
        <div className="flex items-center gap-1">
          {isRunning && (
            <button
              onClick={handleKill}
              className="p-1.5 hover:bg-red-600 rounded transition-colors"
              title="Kill Process"
            >
              <Square className="w-3 h-3" />
            </button>
          )}
          
          <button
            onClick={() => {
              clearTerminal();
              setHistory([]);
            }}
            className="p-1.5 hover:bg-dark-700 rounded transition-colors"
            title="Clear"
          >
            <Trash2 className="w-3 h-3 text-dark-400" />
          </button>
          
          <button
            onClick={() => {}}
            className="p-1.5 hover:bg-dark-700 rounded transition-colors"
          >
            <Maximize2 className="w-3 h-3 text-dark-400" />
          </button>
        </div>
      </div>

      {/* Terminal Output */}
      <div
        ref={terminalRef}
        className="flex-1 overflow-auto p-3 terminal"
        onClick={() => inputRef.current?.focus()}
      >
        {history.length === 0 ? (
          <div className="text-dark-500 text-sm">
            BI-IDE Terminal
            <br />
            Type a command to get started
            <br />
            <br />
            <span className="text-dark-600">
              Current directory: {currentWorkspace?.path || "None"}
            </span>
          </div>
        ) : (
          history.map((line, i) => (
            <div key={i} className="text-sm whitespace-pre-wrap">
              {line.startsWith("> ") ? (
                <span className="text-primary-400">{line}</span>
              ) : line.startsWith("Error:") ? (
                <span className="text-red-400">{line}</span>
              ) : (
                <span className="text-dark-200">{line}</span>
              )}
            </div>
          ))
        )}
      </div>

      {/* Input */}
      <div className="flex items-center gap-2 px-3 py-2 bg-dark-800 border-t border-dark-700">
        <span className="text-primary-400 text-sm">{isRunning ? "⋯" : ">"}</span>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isRunning}
          placeholder={isRunning ? "Running..." : "Enter command..."}
          className="flex-1 bg-transparent outline-none text-sm text-dark-100 placeholder-dark-500 font-mono"
          autoComplete="off"
          autoCorrect="off"
          spellCheck={false}
        />
        {isRunning && (
          <div className="w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
        )}
      </div>
    </div>
  );
}
