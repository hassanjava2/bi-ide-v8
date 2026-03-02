//! AI Code Completion Provider - Inline suggestions

import { useEffect, useRef, useCallback } from "react";
import * as monaco from "monaco-editor";
import { useStore } from "../../lib/store";
import { ai } from "../../lib/tauri";

interface CompletionContext {
  textBeforeCursor: string;
  textAfterCursor: string;
  language: string;
  filePath: string;
}

export function useAICompletion(editor: monaco.editor.IStandaloneCodeEditor | null) {
  const { settings, currentWorkspace } = useStore();
  const abortControllerRef = useRef<AbortController | null>(null);
  const lastCompletionRef = useRef<string>("");

  const requestCompletion = useCallback(async (
    model: monaco.editor.ITextModel,
    position: monaco.Position
  ): Promise<monaco.languages.InlineCompletions | null> => {
    if (!settings.aiEnabled) return null;

    // Cancel previous request
    abortControllerRef.current?.abort();
    abortControllerRef.current = new AbortController();

    const textBeforeCursor = model.getValueInRange({
      startLineNumber: 1,
      startColumn: 1,
      endLineNumber: position.lineNumber,
      endColumn: position.column,
    });

    const textAfterCursor = model.getValueInRange({
      startLineNumber: position.lineNumber,
      startColumn: position.column,
      endLineNumber: model.getLineCount(),
      endColumn: model.getLineMaxColumn(model.getLineCount()),
    });

    const context: CompletionContext = {
      textBeforeCursor: textBeforeCursor.slice(-2000), // Last 2000 chars
      textAfterCursor: textAfterCursor.slice(0, 500),  // Next 500 chars
      language: model.getLanguageId(),
      filePath: currentWorkspace?.path || "",
    };

    try {
      const result = await ai.getCompletion(context);
      
      if (!result.completion || result.completion === lastCompletionRef.current) {
        return null;
      }

      lastCompletionRef.current = result.completion;

      return {
        items: [
          {
            insertText: result.completion,
            range: new monaco.Range(
              position.lineNumber,
              position.column,
              position.lineNumber,
              position.column
            ),
          },
        ],
      };
    } catch (err) {
      console.error("AI completion failed:", err);
      return null;
    }
  }, [settings.aiEnabled, currentWorkspace?.path]);

  useEffect(() => {
    if (!editor) return;

    // Register inline completion provider
    const provider: monaco.languages.InlineCompletionsProvider = {
      provideInlineCompletions: async (model, position) => {
        return await requestCompletion(model, position) as any;
      },
      disposeInlineCompletions: () => {},
    } as any;
    const disposable = monaco.languages.registerInlineCompletionsProvider(
      { pattern: "**" },
      provider
    );

    return () => disposable.dispose();
  }, [editor, requestCompletion]);
}
