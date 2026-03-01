/**
 * AI Chat - دردشة AI
 * 
 * Unified chat interface connecting to real API.
 * NO mock responses - all messages go through /api/v1/council/message
 */

import { useState, useRef, useEffect, useCallback } from "react";
import { 
  Send, 
  Sparkles, 
  Code,
  Copy,
  Check,
  Trash2,
  Settings,
  MoreVertical,
  Image as ImageIcon,
  Mic,
  StopCircle,
  RefreshCw,
  WifiOff,
  AlertCircle,
  Cpu
} from "lucide-react";
import { sendCouncilMessage, CouncilMessageResponse } from "../../config/api";

// Types
interface Message {
  id: string;
  role: "user" | "assistant" | "system" | "error";
  content: string;
  timestamp: number;
  isStreaming?: boolean;
  codeBlocks?: CodeBlock[];
  source?: string;
  confidence?: number;
  wiseMan?: string;
}

interface CodeBlock {
  language: string;
  code: string;
  startIndex: number;
  endIndex: number;
}

interface ChatState {
  isOnline: boolean;
  lastError: string | null;
  retryCount: number;
  isRetrying: boolean;
}

// Quick actions
const quickActions = [
  { id: "generate", label: "توليد كود", icon: Code, prompt: "اكتب كود لـ..." },
  { id: "complete", label: "إكمال", icon: Sparkles, prompt: "أكمل هذا الكود..." },
  { id: "explain", label: "شرح", icon: Settings, prompt: "اشرح هذا الكود..." },
  { id: "fix", label: "إصلاح", icon: RefreshCw, prompt: "أصلح الأخطاء في هذا الكود..." },
];

// Syntax highlighting colors
const syntaxColors: Record<string, string> = {
  keyword: "#c678dd",
  string: "#98c379",
  comment: "#5c6370",
  function: "#61afef",
  number: "#d19a66",
  operator: "#56b6c2",
  default: "#abb2bf",
};

// Simple syntax highlighter
function highlightCode(code: string, language: string): string {
  const keywords = /\b(const|let|var|function|return|if|else|for|while|class|import|export|from|async|await|try|catch|throw|new|this|typeof|instanceof|def|class)\b/g;
  const strings = /(".*?"|'.*?'|`.*?`)/g;
  const comments = /(\/\/.*$|\/\*[\s\S]*?\*\/)/gm;
  const numbers = /\b\d+\.?\d*\b/g;
  const functions = /\b([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?=\()/g;

  let highlighted = code
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  highlighted = highlighted
    .replace(comments, '<span style="color: ' + syntaxColors.comment + '">$&</span>')
    .replace(strings, '<span style="color: ' + syntaxColors.string + '">$&</span>')
    .replace(keywords, '<span style="color: ' + syntaxColors.keyword + '">$&</span>')
    .replace(numbers, '<span style="color: ' + syntaxColors.number + '">$&</span>')
    .replace(functions, '<span style="color: ' + syntaxColors.function + '">$&</span>');

  return highlighted;
}

// Code block component
function CodeBlockComponent({ code, language }: { code: string; language: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  return (
    <div className="my-3 rounded-lg overflow-hidden bg-[#282c34] border border-dark-700">
      <div className="flex items-center justify-between px-4 py-2 bg-[#21252b] border-b border-dark-700">
        <div className="flex items-center gap-2">
          <Code className="w-4 h-4 text-dark-400" />
          <span className="text-xs text-dark-400 font-mono">{language || "text"}</span>
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 px-2 py-1 text-xs text-dark-400 hover:text-dark-200 transition-colors"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3 text-green-400" />
              <span className="text-green-400">تم النسخ</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3" />
              <span>نسخ</span>
            </>
          )}
        </button>
      </div>
      
      <div className="p-4 overflow-x-auto">
        <pre className="text-sm font-mono leading-relaxed">
          <code 
            dangerouslySetInnerHTML={{ __html: highlightCode(code, language) }}
          />
        </pre>
      </div>
    </div>
  );
}

// Message component
function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";
  const isError = message.role === "error";
  
  const renderContent = () => {
    if (!message.codeBlocks || message.codeBlocks.length === 0) {
      return <p className="whitespace-pre-wrap">{message.content}</p>;
    }

    const parts: React.ReactNode[] = [];
    let lastIndex = 0;

    message.codeBlocks.forEach((block, idx) => {
      if (block.startIndex > lastIndex) {
        parts.push(
          <p key={`text-${idx}`} className="whitespace-pre-wrap mb-2">
            {message.content.slice(lastIndex, block.startIndex)}
          </p>
        );
      }

      parts.push(
        <CodeBlockComponent 
          key={`code-${idx}`}
          code={block.code}
          language={block.language}
        />
      );

      lastIndex = block.endIndex;
    });

    if (lastIndex < message.content.length) {
      parts.push(
        <p key="text-final" className="whitespace-pre-wrap">
          {message.content.slice(lastIndex)}
        </p>
      );
    }

    return parts;
  };

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""} animate-fade-in`}>
      {/* Avatar */}
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
        isUser 
          ? "bg-primary-600" 
          : isError
            ? "bg-red-500"
            : isAssistant 
              ? "bg-gradient-to-br from-purple-500 to-pink-500"
              : "bg-dark-700"
      }`}>
        {isUser ? (
          <span className="text-sm font-bold text-white">أنت</span>
        ) : isError ? (
          <AlertCircle className="w-4 h-4 text-white" />
        ) : isAssistant ? (
          <Sparkles className="w-4 h-4 text-white" />
        ) : (
          <Settings className="w-4 h-4 text-dark-400" />
        )}
      </div>

      {/* Content */}
      <div className={`max-w-[85%] rounded-2xl px-4 py-3 ${
        isUser 
          ? "bg-primary-600 text-white rounded-br-sm"
          : isError
            ? "bg-red-500/20 text-red-200 border border-red-500/30 rounded-bl-sm"
            : "bg-dark-800 text-dark-100 rounded-bl-sm"
      }`}>
        {message.isStreaming && (
          <span className="inline-flex gap-1 mr-1">
            <span className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
            <span className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
            <span className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
          </span>
        )}
        <div className={`text-sm leading-relaxed ${isUser ? "" : "text-dark-200"}`}>
          {renderContent()}
        </div>
        
        {/* Source indicator for assistant messages */}
        {isAssistant && message.source && (
          <div className="flex items-center gap-2 mt-2 pt-2 border-t border-dark-700/50">
            <Cpu className="w-3 h-3 text-dark-500" />
            <span className="text-xs text-dark-500">
              {message.source === 'rtx4090' ? 'RTX 4090' : 
               message.source === 'local-fallback' ? 'نظام محلي' : 
               message.wiseMan || 'المجلس'}
            </span>
            {message.confidence !== undefined && (
              <span className="text-xs text-dark-500">
                • ثقة: {Math.round(message.confidence * 100)}%
              </span>
            )}
          </div>
        )}
        
        <div className={`text-xs mt-2 ${isUser ? "text-primary-200" : "text-dark-500"}`}>
          {new Date(message.timestamp).toLocaleTimeString('ar-SA')}
        </div>
      </div>
    </div>
  );
}

// Main component
export function AIChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "هلا! أنا مساعد BI-IDE الذكي. كيف يمكنني مساعدتك اليوم؟\n\nأستطيع:\n• توليد وإكمال الكود\n• شرح الكود\n• إصلاح الأخطاء\n• الإجابة على أسئلتك البرمجية",
      timestamp: Date.now(),
      source: "system",
      confidence: 1.0,
    },
  ]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [chatState, setChatState] = useState<ChatState>({
    isOnline: true,
    lastError: null,
    retryCount: 0,
    isRetrying: false,
  });
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Extract code blocks from content
  const extractCodeBlocks = (content: string): CodeBlock[] => {
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
    const blocks: CodeBlock[] = [];
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      blocks.push({
        language: match[1] || "text",
        code: match[2].trim(),
        startIndex: match.index,
        endIndex: match.index + match[0].length,
      });
    }

    return blocks;
  };

  // Send message to API
  const handleSend = useCallback(async () => {
    if (!input.trim() || isStreaming) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: Date.now(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsStreaming(true);
    setChatState(prev => ({ ...prev, lastError: null, isRetrying: false }));

    // Create assistant message placeholder
    const assistantMessageId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      timestamp: Date.now(),
      isStreaming: true,
    }]);

    try {
      // Call real API - NO mock responses
      const response = await sendCouncilMessage(userMessage.content, {
        session_id: "desktop-session",
        previous_messages: messages.map(m => ({ 
          role: m.role === 'error' ? 'system' : m.role, 
          content: m.content 
        })),
      });

      // Update message with real response
      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantMessageId) {
          return {
            ...msg,
            content: response.response,
            codeBlocks: extractCodeBlocks(response.response),
            isStreaming: false,
            source: response.source,
            confidence: response.confidence,
            wiseMan: response.wise_man,
          };
        }
        return msg;
      }));

      setChatState(prev => ({ 
        ...prev, 
        isOnline: true, 
        retryCount: 0,
        lastError: null,
      }));

    } catch (error) {
      console.error("Failed to get response:", error);
      
      const errorMessage = error instanceof Error ? error.message : "حدث خطأ غير معروف";
      
      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantMessageId) {
          return {
            ...msg,
            role: "error",
            content: `عذراً، لم أتمكن من الاتصال بالمجلس. ${errorMessage}`,
            isStreaming: false,
          };
        }
        return msg;
      }));

      setChatState(prev => ({ 
        ...prev, 
        isOnline: false, 
        lastError: errorMessage,
        retryCount: prev.retryCount + 1,
      }));
    } finally {
      setIsStreaming(false);
    }
  }, [input, isStreaming, messages]);

  // Retry failed message
  const handleRetry = useCallback(() => {
    if (chatState.retryCount >= 3) {
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: "system",
        content: "⚠️ تجاوزت الحد الأقصى للمحاولات. يرجى التحقق من الاتصال بالإنترنت والمحاولة لاحقاً.",
        timestamp: Date.now(),
      }]);
      setChatState(prev => ({ ...prev, retryCount: 0 }));
      return;
    }
    
    setChatState(prev => ({ ...prev, isRetrying: true }));
    handleSend();
  }, [chatState.retryCount, handleSend]);

  // Quick action handler
  const handleQuickAction = (action: typeof quickActions[0]) => {
    setInput(action.prompt);
    inputRef.current?.focus();
  };

  // Clear chat
  const handleClear = () => {
    setMessages([{
      id: "welcome",
      role: "assistant",
      content: "تم مسح المحادثة. كيف يمكنني مساعدتك؟",
      timestamp: Date.now(),
      source: "system",
    }]);
    setChatState({
      isOnline: true,
      lastError: null,
      retryCount: 0,
      isRetrying: false,
    });
  };

  // Keyboard shortcuts
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="h-full flex flex-col bg-dark-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-dark-700 bg-dark-800/50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="font-semibold text-dark-100">مساعد BI-IDE AI</h2>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${
                chatState.isOnline ? "bg-green-500 animate-pulse" : "bg-red-500"
              }`} />
              <span className="text-xs text-dark-400">
                {chatState.isOnline ? "متصل" : "غير متصل"}
              </span>
              {!chatState.isOnline && (
                <button
                  onClick={handleRetry}
                  className="text-xs text-primary-400 hover:text-primary-300"
                  disabled={chatState.isRetrying}
                >
                  {chatState.isRetrying ? "جاري إعادة المحاولة..." : "إعادة المحاولة"}
                </button>
              )}
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={handleClear}
            className="p-2 text-dark-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
            title="مسح المحادثة"
          >
            <Trash2 className="w-4 h-4" />
          </button>
          <button className="p-2 text-dark-400 hover:text-dark-200 hover:bg-dark-700 rounded-lg transition-colors">
            <Settings className="w-4 h-4" />
          </button>
          <button className="p-2 text-dark-400 hover:text-dark-200 hover:bg-dark-700 rounded-lg transition-colors">
            <MoreVertical className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Offline warning */}
      {!chatState.isOnline && (
        <div className="px-4 py-2 bg-red-500/10 border-b border-red-500/20 flex items-center gap-2">
          <WifiOff className="w-4 h-4 text-red-400" />
          <span className="text-sm text-red-400">
            {chatState.lastError || "لا يوجد اتصال بالخادم"}
          </span>
        </div>
      )}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map(message => (
          <ChatMessage key={message.id} message={message} />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Quick actions */}
      <div className="px-4 py-2 border-t border-dark-700/50">
        <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
          {quickActions.map(action => (
            <button
              key={action.id}
              onClick={() => handleQuickAction(action)}
              className="flex items-center gap-2 px-3 py-1.5 bg-dark-800 hover:bg-dark-700 text-dark-300 text-sm rounded-full transition-colors whitespace-nowrap"
            >
              <action.icon className="w-3.5 h-3.5" />
              {action.label}
            </button>
          ))}
        </div>
      </div>

      {/* Input area */}
      <div className="p-4 border-t border-dark-700 bg-dark-800/50">
        <div className="flex items-end gap-2 bg-dark-800 rounded-2xl border border-dark-700 p-2 focus-within:border-primary-500 focus-within:ring-1 focus-within:ring-primary-500 transition-all">
          <button
            className="p-2 text-dark-400 hover:text-dark-200 rounded-xl hover:bg-dark-700 transition-colors"
          >
            <ImageIcon className="w-5 h-5" />
          </button>
          
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={chatState.isOnline 
              ? "اكتب رسالتك هنا... (Shift+Enter لسطر جديد)" 
              : "غير متصل - انتظر إعادة الاتصال..."
            }
            className="flex-1 bg-transparent text-dark-100 placeholder-dark-500 resize-none max-h-32 py-2 outline-none"
            rows={1}
            style={{ minHeight: "24px" }}
            disabled={!chatState.isOnline && chatState.retryCount === 0}
          />
          
          <button
            onClick={() => setIsRecording(!isRecording)}
            className={`p-2 rounded-xl transition-colors ${
              isRecording 
                ? "text-red-400 bg-red-500/10" 
                : "text-dark-400 hover:text-dark-200 hover:bg-dark-700"
            }`}
          >
            {isRecording ? <StopCircle className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
          </button>
          
          <button
            onClick={handleSend}
            disabled={!input.trim() || isStreaming || (!chatState.isOnline && chatState.retryCount === 0)}
            className="p-2 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl text-white transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
        <div className="text-center mt-2">
          <span className="text-xs text-dark-500">
            AI قد ينتج معلومات غير دقيقة. تحقق من المعلومات المهمة.
          </span>
        </div>
      </div>
    </div>
  );
}

export default AIChat;
