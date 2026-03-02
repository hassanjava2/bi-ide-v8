//! AI Toolbar - Explain, Refactor, Fix actions

import { useState } from "react";
import { Sparkles, MessageSquare, Wand2, Bug, X, Loader2 } from "lucide-react";
import * as monaco from "monaco-editor";
import { useStore } from "../../lib/store";
import { ai } from "../../lib/tauri";

interface AIToolbarProps {
  editor: monaco.editor.IStandaloneCodeEditor | null;
}

export function AIToolbar({ editor }: AIToolbarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<string | null>(null);
  const { currentWorkspace } = useStore();

  const getSelectedText = () => {
    if (!editor) return "";
    const selection = editor.getSelection();
    if (!selection) return "";
    return editor.getModel()?.getValueInRange(selection) || "";
  };

  const handleExplain = async () => {
    const code = getSelectedText();
    if (!code) return;

    setLoading(true);
    setIsOpen(true);
    setResponse(null);

    try {
      const result = await ai.chat([
        {
          role: "system",
          content: "You are a code explanation assistant. Explain the following code clearly and concisely.",
        },
        {
          role: "user",
          content: `Explain this code:\n\n${code}`,
        },
      ]);
      setResponse(result.generated_text || "No explanation available");
    } catch (err) {
      setResponse("Failed to get explanation");
    } finally {
      setLoading(false);
    }
  };

  const handleRefactor = async () => {
    const code = getSelectedText();
    if (!code) return;

    setLoading(true);
    setIsOpen(true);
    setResponse(null);

    try {
      const result = await ai.chat([
        {
          role: "system",
          content: "You are a code refactoring assistant. Suggest improvements for the following code.",
        },
        {
          role: "user",
          content: `Refactor this code:\n\n${code}`,
        },
      ]);
      setResponse(result.generated_text || "No suggestions available");
    } catch (err) {
      setResponse("Failed to get refactoring suggestions");
    } finally {
      setLoading(false);
    }
  };

  const handleFix = async () => {
    const code = getSelectedText();
    if (!code) return;

    setLoading(true);
    setIsOpen(true);
    setResponse(null);

    try {
      const result = await ai.chat([
        {
          role: "system",
          content: "You are a code debugging assistant. Find and fix any issues in the following code.",
        },
        {
          role: "user",
          content: `Fix this code:\n\n${code}`,
        },
      ]);
      setResponse(result.generated_text || "No fixes available");
    } catch (err) {
      setResponse("Failed to get fixes");
    } finally {
      setLoading(false);
    }
  };

  if (!editor) return null;

  return (
    <>
      {/* Floating Toolbar */}
      <div className="absolute top-4 right-4 z-30 flex items-center gap-1 bg-dark-800 rounded-lg shadow-lg border border-dark-700 p-1">
        <button
          onClick={handleExplain}
          className="flex items-center gap-1.5 px-2 py-1.5 hover:bg-dark-700 rounded text-sm"
          title="Explain code"
        >
          <MessageSquare className="w-4 h-4 text-blue-400" />
          <span className="hidden sm:inline">Explain</span>
        </button>
        <button
          onClick={handleRefactor}
          className="flex items-center gap-1.5 px-2 py-1.5 hover:bg-dark-700 rounded text-sm"
          title="Refactor code"
        >
          <Wand2 className="w-4 h-4 text-purple-400" />
          <span className="hidden sm:inline">Refactor</span>
        </button>
        <button
          onClick={handleFix}
          className="flex items-center gap-1.5 px-2 py-1.5 hover:bg-dark-700 rounded text-sm"
          title="Fix code"
        >
          <Bug className="w-4 h-4 text-red-400" />
          <span className="hidden sm:inline">Fix</span>
        </button>
      </div>

      {/* AI Response Panel */}
      {isOpen && (
        <div className="absolute bottom-4 right-4 left-4 z-30 bg-dark-800 rounded-lg shadow-xl border border-dark-600 max-h-64 flex flex-col">
          <div className="flex items-center justify-between p-3 border-b border-dark-700">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-primary-400" />
              <span className="font-medium text-sm">AI Assistant</span>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="p-1 hover:bg-dark-700 rounded"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          
          <div className="flex-1 overflow-auto p-3">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-primary-400" />
              </div>
            ) : (
              <div className="text-sm whitespace-pre-wrap text-dark-200">
                {response}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
