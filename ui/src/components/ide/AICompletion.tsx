import React, { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '@/lib/api';
import { Sparkles, Check, X, RefreshCw, ChevronRight, ChevronLeft } from 'lucide-react';

interface Suggestion {
  id: string;
  text: string;
  type: 'completion' | 'correction' | 'optimization';
  confidence: number;
  description?: string;
}

interface AICompletionProps {
  code: string;
  cursorPosition: number;
  language: string;
  filePath: string;
  onAccept: (suggestion: string) => void;
  onDismiss?: () => void;
}

export const AICompletion: React.FC<AICompletionProps> = ({
  code,
  cursorPosition,
  language,
  filePath,
  onAccept,
  onDismiss
}) => {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [visible, setVisible] = useState(false);
  const [position] = useState({ top: 0, left: 0 });
  const debounceRef = useRef<NodeJS.Timeout | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const fetchSuggestions = useCallback(async () => {
    if (!code || cursorPosition < 0) return;

    setLoading(true);
    try {
      const response = await api.post('/ide/copilot/suggest', {
        code,
        cursor_position: cursorPosition,
        language,
        file_path: filePath
      });

      if (response.data.suggestions && response.data.suggestions.length > 0) {
        setSuggestions(response.data.suggestions);
        setCurrentIndex(0);
        setVisible(true);
      }
    } catch (error) {
      console.error('Failed to fetch AI suggestions:', error);
    } finally {
      setLoading(false);
    }
  }, [code, cursorPosition, language, filePath]);

  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(() => {
      if (code && cursorPosition >= 0) {
        fetchSuggestions();
      }
    }, 500);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [code, cursorPosition, fetchSuggestions]);

  const handleAccept = () => {
    if (suggestions[currentIndex]) {
      onAccept(suggestions[currentIndex].text);
      setVisible(false);
      setSuggestions([]);
    }
  };

  const handleDismiss = () => {
    setVisible(false);
    setSuggestions([]);
    onDismiss?.();
  };

  const handleNext = () => {
    setCurrentIndex(prev => (prev + 1) % suggestions.length);
  };

  const handlePrev = () => {
    setCurrentIndex(prev => (prev - 1 + suggestions.length) % suggestions.length);
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'completion': return <Sparkles className="w-3 h-3 text-blue-400" />;
      case 'correction': return <RefreshCw className="w-3 h-3 text-yellow-400" />;
      case 'optimization': return <Check className="w-3 h-3 text-green-400" />;
      default: return <Sparkles className="w-3 h-3" />;
    }
  };

  const getTypeLabel = (type: string) => {
    switch (type) {
      case 'completion': return 'إكمال';
      case 'correction': return 'تصحيح';
      case 'optimization': return 'تحسين';
      default: return type;
    }
  };

  if (!visible && !loading) return null;

  return (
    <div
      ref={containerRef}
      className="absolute z-50 bg-gray-900 border border-bi-accent/30 rounded-lg shadow-xl min-w-[300px] max-w-[500px]"
      style={{ top: position.top, left: position.left }}
    >
      {loading ? (
        <div className="flex items-center gap-2 p-3">
          <div className="w-4 h-4 border-2 border-bi-accent border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-gray-400">جاري التحليل...</span>
        </div>
      ) : suggestions.length > 0 ? (
        <div className="p-3">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              {getTypeIcon(suggestions[currentIndex].type)}
              <span className="text-xs text-gray-400">
                {getTypeLabel(suggestions[currentIndex].type)}
              </span>
              <span className="text-xs text-gray-500">
                ({Math.round(suggestions[currentIndex].confidence * 100)}%)
              </span>
            </div>
            <div className="flex items-center gap-1">
              {suggestions.length > 1 && (
                <>
                  <button
                    onClick={handlePrev}
                    className="p-1 hover:bg-white/10 rounded"
                  >
                    <ChevronRight className="w-3 h-3" />
                  </button>
                  <span className="text-xs text-gray-400">
                    {currentIndex + 1} / {suggestions.length}
                  </span>
                  <button
                    onClick={handleNext}
                    className="p-1 hover:bg-white/10 rounded"
                  >
                    <ChevronLeft className="w-3 h-3" />
                  </button>
                </>
              )}
              <button
                onClick={handleDismiss}
                className="p-1 hover:bg-red-500/20 rounded text-red-400"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          </div>

          <div className="bg-black/30 rounded p-2 mb-2">
            <pre className="text-sm text-green-400 font-mono whitespace-pre-wrap">
              {suggestions[currentIndex].text}
            </pre>
          </div>

          {suggestions[currentIndex].description && (
            <p className="text-xs text-gray-400 mb-2">
              {suggestions[currentIndex].description}
            </p>
          )}

          <div className="flex gap-2">
            <button
              onClick={handleAccept}
              className="flex-1 flex items-center justify-center gap-1 px-3 py-1.5 bg-bi-accent hover:bg-bi-accent/90 rounded text-sm transition-colors"
            >
              <Check className="w-3 h-3" />
              قبول (Tab)
            </button>
            <button
              onClick={handleDismiss}
              className="px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded text-sm transition-colors"
            >
              رفض (Esc)
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
};

// Inline completion ghost text component
interface InlineCompletionProps {
  suggestion: string;
  visible: boolean;
}

export const InlineCompletion: React.FC<InlineCompletionProps> = ({
  suggestion,
  visible
}) => {
  if (!visible || !suggestion) return null;

  return (
    <span className="text-gray-500 pointer-events-none select-none">
      {suggestion}
    </span>
  );
};

// AI Status indicator component
interface AIStatusProps {
  isLoading?: boolean;
  hasSuggestion?: boolean;
  onClick?: () => void;
}

export const AIStatus: React.FC<AIStatusProps> = ({
  isLoading,
  hasSuggestion,
  onClick
}) => {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-all ${
        isLoading 
          ? 'bg-blue-500/20 text-blue-400' 
          : hasSuggestion 
            ? 'bg-green-500/20 text-green-400' 
            : 'bg-gray-500/20 text-gray-400'
      }`}
    >
      {isLoading ? (
        <>
          <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
          <span>AI يفكر...</span>
        </>
      ) : hasSuggestion ? (
        <>
          <Sparkles className="w-3 h-3" />
          <span>اقتراح متاح</span>
        </>
      ) : (
        <>
          <Sparkles className="w-3 h-3" />
          <span>AI جاهز</span>
        </>
      )}
    </button>
  );
};
