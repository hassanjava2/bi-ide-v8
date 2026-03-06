//! Enhanced Council Panel — 16 Sages 🏛️
//  Connected to: brain_api.py → council_auto_debate.py

import { useState, useEffect, useRef } from "react";
import { MessageSquare, Send, Sparkles, Loader2, Users, Vote, Brain } from "lucide-react";
import { useStore } from "../../lib/store";
import { council } from "../../lib/tauri";

// === 16 Sage definitions ===
const SAGES: Record<string, { name: string; nameEn: string; icon: string; color: string; expertise: string }> = {
  tech_sage: { name: "حكيم التقنية", nameEn: "Tech Sage", icon: "💻", color: "#3B82F6", expertise: "البرمجة" },
  security_sage: { name: "حكيم الأمان", nameEn: "Security Sage", icon: "🔒", color: "#EF4444", expertise: "الأمن السيبراني" },
  infra_sage: { name: "حكيم البنية", nameEn: "Infrastructure Sage", icon: "🏗️", color: "#10B981", expertise: "السيرفرات" },
  data_sage: { name: "حكيم البيانات", nameEn: "Data Sage", icon: "🗄️", color: "#8B5CF6", expertise: "قواعد البيانات" },
  design_sage: { name: "حكيم التصميم", nameEn: "Design Sage", icon: "🎨", color: "#EC4899", expertise: "UI/UX" },
  testing_sage: { name: "حكيم الاختبار", nameEn: "Testing Sage", icon: "🧪", color: "#F59E0B", expertise: "الجودة" },
  physics_sage: { name: "حكيم الفيزياء", nameEn: "Physics Sage", icon: "⚛️", color: "#06B6D4", expertise: "الفيزياء" },
  chemistry_sage: { name: "حكيم الكيمياء", nameEn: "Chemistry Sage", icon: "🧬", color: "#84CC16", expertise: "الكيمياء" },
  economics_sage: { name: "حكيم الاقتصاد", nameEn: "Economics Sage", icon: "💰", color: "#F97316", expertise: "التكاليف" },
  arabic_sage: { name: "حكيم العربية", nameEn: "Arabic Sage", icon: "📚", color: "#14B8A6", expertise: "المعرفة" },
  strategy_sage: { name: "الحكيم الأعلى", nameEn: "Grand Sage", icon: "🧙", color: "#6366F1", expertise: "الاستراتيجية" },
  rebel_sage: { name: "المتمرد", nameEn: "The Rebel", icon: "⚔️", color: "#DC2626", expertise: "النقد" },
  translator_sage: { name: "حكيم الترجمة", nameEn: "Translation Sage", icon: "🌐", color: "#0EA5E9", expertise: "اللغات" },
  materials_sage: { name: "حكيم المواد", nameEn: "Materials Sage", icon: "🔩", color: "#78716C", expertise: "المواد" },
  manufacturing_sage: { name: "حكيم الإنتاج", nameEn: "Manufacturing Sage", icon: "🏭", color: "#A3E635", expertise: "المصانع" },
  captain: { name: "القائد", nameEn: "The Captain", icon: "👑", color: "#FFD700", expertise: "القرار" },
};

interface CouncilMessage {
  id: string;
  role: "user" | "wise_man";
  content: string;
  timestamp: number;
  sageId?: string;
  sageName?: string;
  sageIcon?: string;
  sageColor?: string;
  vote?: string;
  confidence?: number;
}

interface DebateResult {
  topic: string;
  opinions: Array<{
    sage_id: string;
    sage_name: string;
    opinion: string;
    vote: string;
    confidence: number;
    icon: string;
  }>;
  votes: { approve: number; reject: number; neutral: number };
  decision: string;
}

export function CouncilPanel() {
  const [messages, setMessages] = useState<CouncilMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showSages, setShowSages] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMsg: CouncilMessage = {
      id: `msg-${Date.now()}`,
      role: "user",
      content: input.trim(),
      timestamp: Date.now(),
    };

    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      // Try brain_api first
      const response = await fetch("http://localhost:8400/api/council/debate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic: userMsg.content }),
      });

      if (response.ok) {
        const data: DebateResult = await response.json();

        // Add each sage's opinion as a message
        for (const opinion of data.opinions) {
          const sage = SAGES[opinion.sage_id];
          setMessages(prev => [...prev, {
            id: `msg-${Date.now()}-${opinion.sage_id}`,
            role: "wise_man",
            content: opinion.opinion,
            timestamp: Date.now(),
            sageId: opinion.sage_id,
            sageName: sage?.name || opinion.sage_name,
            sageIcon: sage?.icon || opinion.icon,
            sageColor: sage?.color || "#666",
            vote: opinion.vote,
            confidence: opinion.confidence,
          }]);
        }

        // Add final decision
        const voteEmoji = `✅ ${data.votes.approve} | ❌ ${data.votes.reject} | ⚖️ ${data.votes.neutral}`;
        setMessages(prev => [...prev, {
          id: `msg-${Date.now()}-decision`,
          role: "wise_man",
          content: `**${data.decision}**\n${voteEmoji}`,
          timestamp: Date.now(),
          sageName: "النتيجة",
          sageIcon: "🏛️",
          sageColor: "#FFD700",
        }]);
      } else {
        // Fallback to Tauri council
        const response = await council.sendMessage(userMsg.content);
        setMessages(prev => [...prev, {
          id: `msg-${Date.now() + 1}`,
          role: "wise_man",
          content: response.response || response.message || "No response",
          timestamp: Date.now(),
          sageName: "المجلس",
          sageIcon: "🧙",
        }]);
      }
    } catch (err) {
      // Fallback to old Tauri method
      try {
        const response = await council.sendMessage(userMsg.content);
        setMessages(prev => [...prev, {
          id: `msg-${Date.now() + 1}`,
          role: "wise_man",
          content: response.response || response.message || "No response",
          timestamp: Date.now(),
        }]);
      } catch {
        setMessages(prev => [...prev, {
          id: `msg-${Date.now() + 1}`,
          role: "wise_man",
          content: "⚠️ Council offline — start: python3 brain/brain_api.py --serve",
          timestamp: Date.now(),
        }]);
      }
    } finally {
      setLoading(false);
    }
  };

  const voteIcon = (vote?: string) => {
    if (vote === "approve") return "✅";
    if (vote === "reject") return "❌";
    return "⚖️";
  };

  return (
    <div className="h-full flex flex-col bg-dark-800 p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🏛️</span>
          <div>
            <h3 className="font-medium text-sm">مجلس الحكماء</h3>
            <p className="text-xs text-dark-400">16 Sages Council</p>
          </div>
        </div>
        <button
          onClick={() => setShowSages(!showSages)}
          className="p-1.5 hover:bg-dark-700 rounded transition-colors"
          title="Show sages"
        >
          <Users className="w-4 h-4 text-dark-400" />
        </button>
      </div>

      {/* Sages Grid */}
      {showSages && (
        <div className="grid grid-cols-4 gap-1 mb-3 p-2 bg-dark-900 rounded-lg">
          {Object.entries(SAGES).map(([id, sage]) => (
            <div
              key={id}
              className="flex flex-col items-center p-1.5 rounded hover:bg-dark-800 cursor-pointer transition-colors"
              title={`${sage.name} — ${sage.expertise}`}
              style={{ borderBottom: `2px solid ${sage.color}` }}
            >
              <span className="text-lg">{sage.icon}</span>
              <span className="text-[9px] text-dark-400 text-center leading-tight mt-0.5">
                {sage.name}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-2 mb-3">
        {messages.length === 0 && (
          <div className="text-center py-8 text-dark-500">
            <Sparkles className="w-8 h-8 mx-auto mb-2 opacity-30" />
            <p className="text-sm">اطرح موضوعًا على المجلس</p>
            <p className="text-xs text-dark-600 mt-1">16 حكيم يتناقشون ويصوتون</p>
          </div>
        )}

        {messages.map(msg => (
          <div
            key={msg.id}
            className={`p-2.5 rounded-lg text-sm ${msg.role === "user"
                ? "bg-primary-600/20 ml-6"
                : "bg-dark-700 mr-2"
              }`}
          >
            {msg.role === "wise_man" && msg.sageName && (
              <div className="flex items-center gap-1.5 mb-1">
                <span className="text-sm">{msg.sageIcon || "🧙"}</span>
                <span
                  className="text-xs font-medium"
                  style={{ color: msg.sageColor || "#A78BFA" }}
                >
                  {msg.sageName}
                </span>
                {msg.vote && (
                  <span className="text-xs ml-auto">{voteIcon(msg.vote)}</span>
                )}
                {msg.confidence !== undefined && (
                  <span className="text-[10px] text-dark-500">
                    {Math.round(msg.confidence * 100)}%
                  </span>
                )}
              </div>
            )}
            <div className="text-dark-200 whitespace-pre-wrap">{msg.content}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleSend()}
          placeholder="اطرح موضوعاً..."
          disabled={loading}
          className="flex-1 px-3 py-2 bg-dark-900 border border-dark-700 rounded text-sm
                     focus:border-primary-500 focus:outline-none transition-colors"
        />
        <button
          onClick={handleSend}
          disabled={loading}
          className="px-3 py-2 bg-primary-600 hover:bg-primary-500
                     disabled:bg-dark-700 text-white rounded transition-colors"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
        </button>
      </div>
    </div>
  );
}
