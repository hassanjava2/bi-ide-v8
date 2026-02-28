/**
* دردشة AI - AI Chat
* واجهة دردشة مع الذكاء الاصطناعي مع دعم تدفق الردود وتظليل الكود
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
  RefreshCw
} from "lucide-react";

// أنواع البيانات
interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  isStreaming?: boolean;
  codeBlocks?: CodeBlock[];
}

interface CodeBlock {
  language: string;
  code: string;
  startIndex: number;
  endIndex: number;
}

// الإجراءات السريعة
const quickActions = [
  { id: "generate", label: "توليد كود", icon: Code, prompt: "اكتب كود لـ..." },
  { id: "complete", label: "إكمال", icon: Sparkles, prompt: "أكمل هذا الكود..." },
  { id: "explain", label: "شرح", icon: Settings, prompt: "اشرح هذا الكود..." },
  { id: "fix", label: "إصلاح", icon: RefreshCw, prompt: "أصلح الأخطاء في هذا الكود..." },
];

// ألوان تظليل الكود
const syntaxColors: Record<string, string> = {
  keyword: "#c678dd",
  string: "#98c379",
  comment: "#5c6370",
  function: "#61afef",
  number: "#d19a66",
  operator: "#56b6c2",
  default: "#abb2bf",
};

// محلل بسيط لتظليل الكود
function highlightCode(code: string, language: string): string {
  // هذا محلل بسيط للعرض - في الإنتاج استخدم prismjs أو highlight.js
  const keywords = /\b(const|let|var|function|return|if|else|for|while|class|import|export|from|async|await|try|catch|throw|new|this|typeof|instanceof)\b/g;
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

// مكون كتلة الكود
function CodeBlockComponent({ 
  code, 
  language 
}: { 
  code: string; 
  language: string;
}) {
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
      {/* رأس كتلة الكود */}
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
      
      {/* محتوى الكود */}
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

// مكون رسالة المحادثة
function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";
  
  // تقسيم المحتوى إلى أجزاء نصية وكتل كود
  const renderContent = () => {
    if (!message.codeBlocks || message.codeBlocks.length === 0) {
      return <p className="whitespace-pre-wrap">{message.content}</p>;
    }

    const parts: React.ReactNode[] = [];
    let lastIndex = 0;

    message.codeBlocks.forEach((block, idx) => {
      // النص قبل كتلة الكود
      if (block.startIndex > lastIndex) {
        parts.push(
          <p key={`text-${idx}`} className="whitespace-pre-wrap mb-2">
            {message.content.slice(lastIndex, block.startIndex)}
          </p>
        );
      }

      // كتلة الكود
      parts.push(
        <CodeBlockComponent 
          key={`code-${idx}`}
          code={block.code}
          language={block.language}
        />
      );

      lastIndex = block.endIndex;
    });

    // النص المتبقي بعد آخر كتلة كود
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
      {/* الأفاتار */}
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
        isUser 
          ? "bg-primary-600" 
          : isAssistant 
            ? "bg-gradient-to-br from-purple-500 to-pink-500"
            : "bg-dark-700"
      }`}>
        {isUser ? (
          <span className="text-sm font-bold text-white">أنت</span>
        ) : isAssistant ? (
          <Sparkles className="w-4 h-4 text-white" />
        ) : (
          <Settings className="w-4 h-4 text-dark-400" />
        )}
      </div>

      {/* المحتوى */}
      <div className={`max-w-[85%] rounded-2xl px-4 py-3 ${
        isUser 
          ? "bg-primary-600 text-white rounded-br-sm"
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
        <div className={`text-xs mt-2 ${isUser ? "text-primary-200" : "text-dark-500"}`}>
          {new Date(message.timestamp).toLocaleTimeString('ar-SA')}
        </div>
      </div>
    </div>
  );
}

// المكون الرئيسي
export function AIChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "هلا! أنا مساعد BI-IDE الذكي. كيف يمكنني مساعدتك اليوم؟\n\nأستطيع:\n• توليد وإكمال الكود\n• شرح الكود\n• إصلاح الأخطاء\n• الإجابة على أسئلتك البرمجية",
      timestamp: Date.now(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // التمرير التلقائي
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // استخراج كتل الكود
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

  // إرسال الرسالة
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

    // رسالة مساعد فارغة للتدفق
    const assistantMessageId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      timestamp: Date.now(),
      isStreaming: true,
    }]);

    // محاكاة تدفق الرد
    const responses = [
      "بالتأكيد! إليك الكود المطلوب:\n\n```typescript\nfunction calculateSum(numbers: number[]): number {\n  return numbers.reduce((sum, num) => sum + num, 0);\n}\n```\n\nهذه الدالة تستخدم `reduce` لحساب مجموع الأرقام.",
      "أفهم! إليك مثال على استخدام React hooks:\n\n```tsx\nimport { useState, useEffect } from 'react';\n\nfunction useCounter(initial = 0) {\n  const [count, setCount] = useState(initial);\n  \n  useEffect(() => {\n    console.log(`Count changed to: ${count}`);\n  }, [count]);\n  \n  return { count, increment: () => setCount(c => c + 1) };\n}\n```",
      "يمكنك إصلاح هذا الخطأ بإضافة التحقق من null:\n\n```typescript\nconst userName = user?.profile?.name ?? 'Anonymous';\n```\n\nالـ optional chaining (`?.`) يمنع حدوث errors.",
    ];

    const response = responses[Math.floor(Math.random() * responses.length)];
    const words = response.split(" ");

    for (let i = 0; i < words.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 100));
      
      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantMessageId) {
          const content = words.slice(0, i + 1).join(" ");
          return {
            ...msg,
            content,
            codeBlocks: extractCodeBlocks(content),
            isStreaming: i < words.length - 1,
          };
        }
        return msg;
      }));
    }

    setIsStreaming(false);
  }, [input, isStreaming]);

  // معالجة الإجراء السريع
  const handleQuickAction = (action: typeof quickActions[0]) => {
    setInput(action.prompt);
    inputRef.current?.focus();
  };

  // مسح المحادثة
  const handleClear = () => {
    setMessages([{
      id: "welcome",
      role: "assistant",
      content: "تم مسح المحادثة. كيف يمكنني مساعدتك؟",
      timestamp: Date.now(),
    }]);
  };

  // معالجة اختصارات لوحة المفاتيح
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="h-full flex flex-col bg-dark-900">
      {/* رأس الدردشة */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-dark-700 bg-dark-800/50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="font-semibold text-dark-100">مساعد BI-IDE AI</h2>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-xs text-dark-400">متصل</span>
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

      {/* منطقة الرسائل */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map(message => (
          <ChatMessage key={message.id} message={message} />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* الإجراءات السريعة */}
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

      {/* منطقة الإدخال */}
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
            placeholder="اكتب رسالتك هنا... (Shift+Enter لسطر جديد)"
            className="flex-1 bg-transparent text-dark-100 placeholder-dark-500 resize-none max-h-32 py-2 outline-none"
            rows={1}
            style={{ minHeight: "24px" }}
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
            disabled={!input.trim() || isStreaming}
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
