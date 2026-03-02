//! Monaco Editor Integration for BI-IDE Desktop v8
//! Production-grade code editor with syntax highlighting, IntelliSense, and AI features

import { useEffect, useRef, useCallback, useState } from "react";
import Editor, { useMonaco, loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor";
import { useStore } from "../../lib/store";
import { fs } from "../../lib/tauri";
import { debounce } from "../../lib/utils";
import { FileInfo } from "./FileInfo";
import { Breadcrumbs } from "./Breadcrumbs";
import { listen } from "@tauri-apps/api/event";

// Configure Monaco loader
loader.config({ monaco });

// Language mapping
const EXT_TO_LANG: Record<string, string> = {
  // Web
  js: "javascript",
  jsx: "javascript",
  ts: "typescript",
  tsx: "typescript",
  html: "html",
  htm: "html",
  css: "css",
  scss: "scss",
  sass: "sass",
  less: "less",
  json: "json",
  // Backend
  py: "python",
  rs: "rust",
  go: "go",
  java: "java",
  kt: "kotlin",
  rb: "ruby",
  php: "php",
  // Systems
  c: "c",
  cpp: "cpp",
  cc: "cpp",
  cxx: "cpp",
  h: "c",
  hpp: "cpp",
  // Config
  yaml: "yaml",
  yml: "yaml",
  toml: "toml",
  xml: "xml",
  sh: "shell",
  bash: "shell",
  zsh: "shell",
  ps1: "powershell",
  // Data
  sql: "sql",
  md: "markdown",
  mdx: "markdown",
  // Other
  dockerfile: "dockerfile",
  env: "ini",
  gitignore: "ignore",
};

interface MonacoEditorProps {
  className?: string;
}

export function MonacoEditor({ className }: MonacoEditorProps) {
  const monaco = useMonaco();
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [inlineSuggestion, setInlineSuggestion] = useState<string | null>(null);

  const {
    openFiles,
    activeFilePath,
    closeFile,
    setActiveFile,
    updateFileContent,
    markFileSaved,
    settings,
    currentWorkspace,
  } = useStore();

  const activeFile = openFiles.find((f) => f.path === activeFilePath);

  // Get language from file path
  const getLanguage = useCallback((path: string): string => {
    const ext = path.split(".").pop()?.toLowerCase() || "";
    const lang = EXT_TO_LANG[ext];
    if (lang) return lang;
    
    // Special cases
    const basename = path.split("/").pop()?.toLowerCase() || "";
    if (basename === "dockerfile") return "dockerfile";
    if (basename === "makefile") return "makefile";
    if (basename.endsWith(".gitignore")) return "ignore";
    
    return "plaintext";
  }, []);

  // Auto-save with debounce
  const debouncedSave = useCallback(
    debounce(async (path: string, content: string) => {
      try {
        await fs.writeFile(path, content);
        markFileSaved(path);
      } catch (e) {
        console.error("Auto-save failed:", e);
      }
    }, settings.autoSaveDelay),
    [settings.autoSaveDelay, markFileSaved]
  );

  // Handle content change
  const handleEditorChange = useCallback(
    (value: string | undefined) => {
      if (!activeFile || value === undefined) return;
      
      updateFileContent(activeFile.path, value);
      
      if (settings.autoSave) {
        debouncedSave(activeFile.path, value);
      }
    },
    [activeFile, settings.autoSave, updateFileContent, debouncedSave]
  );

  // Editor mount handler
  const handleEditorDidMount = useCallback(
    (editor: monaco.editor.IStandaloneCodeEditor, monacoInstance: typeof monaco) => {
      editorRef.current = editor;
      setIsReady(true);

      // Configure editor
      editor.updateOptions({
        fontSize: settings.fontSize,
        fontFamily: settings.fontFamily,
        wordWrap: settings.wordWrap ? "on" : "off",
        minimap: { enabled: settings.minimapEnabled },
        lineNumbers: settings.lineNumbers ? "on" : "off",
        folding: true,
        renderWhitespace: settings.renderWhitespace ? "all" : "selection",
        scrollBeyondLastLine: false,
        smoothScrolling: true,
        cursorBlinking: "smooth",
        cursorSmoothCaretAnimation: "on",
        bracketPairColorization: { enabled: true },
        guides: {
          bracketPairs: true,
          indentation: true,
        },
        automaticLayout: true,
        tabSize: settings.tabSize,
        insertSpaces: settings.insertSpaces,
        formatOnPaste: true,
        formatOnType: true,
        quickSuggestions: true,
        suggestOnTriggerCharacters: true,
        acceptSuggestionOnEnter: "on",
        snippetSuggestions: "inline",
        wordBasedSuggestions: "currentDocument",
      });

      // Add command palette actions
      if (monacoInstance) {
        editor.addAction({
          id: "bi-ide-save",
          label: "BI-IDE: Save File",
          keybindings: [monacoInstance.KeyMod.CtrlCmd | monacoInstance.KeyCode.KeyS],
          run: async () => {
            if (activeFile) {
              const content = editor.getValue();
              try {
                await fs.writeFile(activeFile.path, content);
                markFileSaved(activeFile.path);
              } catch (err) {
                console.error("Save failed:", err);
              }
            }
          },
        });

        editor.addAction({
          id: "bi-ide-close-tab",
          label: "BI-IDE: Close Tab",
          keybindings: [monacoInstance.KeyMod.CtrlCmd | monacoInstance.KeyCode.KeyW],
          run: () => {
            if (activeFilePath) {
              closeFile(activeFilePath);
            }
          },
        });

        editor.addAction({
          id: "bi-ide-next-tab",
          label: "BI-IDE: Next Tab",
          keybindings: [monacoInstance.KeyMod.CtrlCmd | monacoInstance.KeyCode.Tab],
          run: () => {
            if (openFiles.length > 1) {
              const currentIndex = openFiles.findIndex((f) => f.path === activeFilePath);
              const nextIndex = (currentIndex + 1) % openFiles.length;
              setActiveFile(openFiles[nextIndex].path);
            }
          },
        });
      }

      // Configure TypeScript/JavaScript
      const ts = (monacoInstance as any)?.languages?.typescript;
      if (ts && ts.typescriptDefaults) {
        ts.typescriptDefaults.setDiagnosticsOptions({
          noSemanticValidation: false,
          noSyntaxValidation: false,
        });
        
        ts.typescriptDefaults.setCompilerOptions({
          target: ts.ScriptTarget?.ES2020 || 7,
          allowNonTsExtensions: true,
          moduleResolution: ts.ModuleResolutionKind?.NodeJs || 2,
          noEmit: true,
          esModuleInterop: true,
          jsx: ts.JsxEmit?.React || 4,
          reactNamespace: "React",
          allowJs: true,
          typeRoots: ["node_modules/@types"],
        });
      }

      // Listen for keyboard events for AI completion trigger
      editor.onKeyUp((e) => {
        // Trigger AI completion after typing pause
        const KeyCode = monacoInstance?.KeyCode;
        if (KeyCode && (e.keyCode === KeyCode.Enter || e.keyCode === KeyCode.Space)) {
          // Could trigger AI completion here
        }
      });
    },
    [settings, activeFile, activeFilePath, openFiles, closeFile, setActiveFile, markFileSaved]
  );

  // Update editor options when settings change
  useEffect(() => {
    if (editorRef.current) {
      editorRef.current.updateOptions({
        fontSize: settings.fontSize,
        fontFamily: settings.fontFamily,
        wordWrap: settings.wordWrap ? "on" : "off",
        minimap: { enabled: settings.minimapEnabled },
        lineNumbers: settings.lineNumbers ? "on" : "off",
        tabSize: settings.tabSize,
        insertSpaces: settings.insertSpaces,
      });
    }
  }, [settings]);

  // Focus editor when active file changes
  useEffect(() => {
    if (editorRef.current && activeFile) {
      editorRef.current.focus();
    }
  }, [activeFilePath]);

  // Register AI completion provider
  useEffect(() => {
    if (!monaco) return;

    const disposable = monaco.languages.registerInlineCompletionsProvider("*", {
      provideInlineCompletions: async (model, position, context, token) => {
        // This will be connected to AI service in Phase 4a
        // For now, return empty
        return { items: [] };
      },
    } as any);

    return () => disposable.dispose();
  }, [monaco]);

  // Listen for CommandPalette editor events
  useEffect(() => {
    if (!editorRef.current) return;

    const editor = editorRef.current;

    const unlistenUndo = listen("editor-undo", () => {
      editor.trigger("command-palette", "undo", null);
    });

    const unlistenRedo = listen("editor-redo", () => {
      editor.trigger("command-palette", "redo", null);
    });

    const unlistenCut = listen("editor-cut", () => {
      editor.trigger("command-palette", "editor.action.clipboardCutAction", null);
    });

    const unlistenCopy = listen("editor-copy", () => {
      editor.trigger("command-palette", "editor.action.clipboardCopyAction", null);
    });

    const unlistenPaste = listen("editor-paste", () => {
      editor.trigger("command-palette", "editor.action.clipboardPasteAction", null);
    });

    const unlistenFind = listen("editor-find", () => {
      editor.trigger("command-palette", "actions.find", null);
    });

    const unlistenReplace = listen("editor-replace", () => {
      editor.trigger("command-palette", "editor.action.startFindReplaceAction:select", null);
    });

    const unlistenSave = listen("save-active-file", async () => {
      if (activeFile) {
        const content = editor.getValue();
        try {
          await fs.writeFile(activeFile.path, content);
          markFileSaved(activeFile.path);
        } catch (err) {
          console.error("Save failed:", err);
        }
      }
    });

    const unlistenSaveAll = listen("save-all-files", async () => {
      // Save all open files
      for (const file of openFiles) {
        if (file.isModified) {
          try {
            await fs.writeFile(file.path, file.content);
            markFileSaved(file.path);
          } catch (err) {
            console.error("Save failed for", file.path, err);
          }
        }
      }
    });

    const unlistenExplain = listen("ai-explain-code", async () => {
      const selection = editor.getSelection();
      if (selection && !selection.isEmpty()) {
        const model = editor.getModel();
        if (model) {
          const selectedText = model.getValueInRange(selection);
          console.log("AI Explain requested for:", selectedText.slice(0, 100));
          // In production, this would call ai.explain()
        }
      }
    });

    const unlistenRefactor = listen("ai-refactor-code", async () => {
      const selection = editor.getSelection();
      if (selection && !selection.isEmpty()) {
        const model = editor.getModel();
        if (model) {
          const selectedText = model.getValueInRange(selection);
          console.log("AI Refactor requested for:", selectedText.slice(0, 100));
          // In production, this would call ai.refactor()
        }
      }
    });

    return () => {
      unlistenUndo.then((fn) => fn());
      unlistenRedo.then((fn) => fn());
      unlistenCut.then((fn) => fn());
      unlistenCopy.then((fn) => fn());
      unlistenPaste.then((fn) => fn());
      unlistenFind.then((fn) => fn());
      unlistenReplace.then((fn) => fn());
      unlistenSave.then((fn) => fn());
      unlistenSaveAll.then((fn) => fn());
      unlistenExplain.then((fn) => fn());
      unlistenRefactor.then((fn) => fn());
    };
  }, [activeFile, openFiles, markFileSaved]);

  if (openFiles.length === 0) {
    return (
      <div className={`h-full flex items-center justify-center bg-[#1e1e1e] ${className}`}>
        <div className="text-center">
          <div className="text-6xl mb-6 opacity-20">📁</div>
          <div className="text-dark-400 text-sm mb-2">Select a file to start editing</div>
          <div className="text-dark-500 text-xs">
            <kbd className="px-2 py-1 bg-dark-700 rounded">Ctrl+O</kbd> to open folder
          </div>
          <div className="text-dark-500 text-xs mt-2">
            <kbd className="px-2 py-1 bg-dark-700 rounded">Ctrl+P</kbd> for Quick Open
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`h-full flex flex-col bg-[#1e1e1e] ${className}`}>
      {/* Breadcrumbs */}
      {activeFile && (
        <Breadcrumbs path={activeFile.path} workspace={currentWorkspace?.path} />
      )}
      
      {/* Tabs */}
      <div className="flex bg-[#252526] border-b border-[#1e1e1e] overflow-x-auto">
        {openFiles.map((file) => (
          <FileInfo
            key={file.path}
            file={file}
            isActive={activeFilePath === file.path}
            onClick={() => setActiveFile(file.path)}
            onClose={() => closeFile(file.path)}
          />
        ))}
      </div>

      {/* Monaco Editor */}
      <div className="flex-1 relative">
        {activeFile ? (
          <Editor
            height="100%"
            language={getLanguage(activeFile.path)}
            value={activeFile.content}
            theme={settings.theme === "dark" ? "vs-dark" : "vs"}
            onChange={handleEditorChange}
            onMount={handleEditorDidMount}
            options={{
              readOnly: false,
            }}
            loading={
              <div className="h-full flex items-center justify-center text-dark-400">
                Loading editor...
              </div>
            }
          />
        ) : (
          <div className="h-full flex items-center justify-center text-dark-500">
            Select a tab to view
          </div>
        )}

        {/* Inline AI Suggestion Overlay */}
        {inlineSuggestion && isReady && (
          <div className="absolute bottom-4 right-4 bg-dark-800 border border-primary-500/30 rounded-lg p-3 shadow-lg max-w-md">
            <div className="text-xs text-primary-400 mb-1">AI Suggestion</div>
            <div className="text-sm text-dark-200">{inlineSuggestion}</div>
            <div className="flex gap-2 mt-2">
              <button className="px-2 py-1 bg-primary-600 text-xs rounded hover:bg-primary-500">
                Accept
              </button>
              <button className="px-2 py-1 bg-dark-700 text-xs rounded hover:bg-dark-600">
                Ignore
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Standalone export for dynamic imports
export default MonacoEditor;
