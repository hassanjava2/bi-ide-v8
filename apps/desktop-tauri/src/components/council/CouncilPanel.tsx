//! Enhanced Council Panel

import { useState, useEffect, useRef } from "react";
import { MessageSquare, Send, Sparkles, Loader2 } from "lucide-react";
import { useStore } from "../../lib/store";
import { council } from "../../lib/tauri";

interface CouncilMessage {
  id: string;
  role: "user" | "wise_man";
  content: string;
  timestamp: number;
}

export function CouncilPanel() {
  const [messages, setMessages] = useState<CouncilMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { currentWorkspace } = useStore();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: CouncilMessage = {
      id: `msg-${Date.now()}`,
      role: "user",
      content: input.trim(),
      timestamp: Date.now(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await council.sendMessage(userMessage.content);
      
      setMessages(prev => [...prev, {
        id: `msg-${Date.now() + 1}`,
        role: "wise_man",
        content: response.response || response.message || "No response",
        timestamp: Date.now(),
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        id: `msg-${Date.now() + 1}`,
        role: "wise_man",
        content: "Error: Could not reach the Council",
        timestamp: Date.now(),
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-dark-800 p-4">
      <div className="flex items-center gap-2 mb-4">
        <span className="text-2xl">🏛️</span>
        <div>
          <h3 className="font-medium">مجلس الحكماء</h3>
          <p className="text-xs text-dark-400">AI Council</p>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 mb-4">
        {messages.length === 0 && (
          <div className="text-center py-8 text-dark-500">
            <Sparkles className="w-8 h-8 mx-auto mb-2 opacity-30" />
            <p className="text-sm">Ask the Council</p>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`p-3 rounded-lg text-sm ${
              msg.role === "user" ? "bg-primary-600/20 ml-8" : "bg-dark-700 mr-8"
            }`}
          >
            {msg.content}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask..."
          disabled={loading}
          className="flex-1 px-3 py-2 bg-dark-900 border border-dark-700 rounded text-sm"
        />
        <button
          onClick={handleSend}
          disabled={loading}
          className="px-3 py-2 bg-primary-600 hover:bg-primary-500 disabled:bg-dark-700 text-white rounded"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
        </button>
      </div>
    </div>
  );
}
