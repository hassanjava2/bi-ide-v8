import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { api } from '@/lib/api';

interface Message {
  id: string;
  role: 'user' | 'wise_man';
  content: string;
  wise_man_name?: string;
  timestamp: string;
}

export const CouncilChat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await api.post('/council/message', {
        message: input,
        wise_man: 'all' // or specific wise man
      });

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'wise_man',
        content: response.data.response,
        wise_man_name: response.data.wise_man,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Failed to send message:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[600px] border rounded-lg">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[70%] p-3 rounded-lg ${
                msg.role === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}
            >
              {msg.role === 'wise_man' && (
                <div className="text-xs font-bold mb-1">{msg.wise_man_name}</div>
              )}
              <div>{msg.content}</div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 border-t flex gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="اكتب رسالتك للمجلس..."
          disabled={loading}
        />
        <Button onClick={sendMessage} disabled={loading}>
          {loading ? 'جاري الإرسال...' : 'إرسال'}
        </Button>
      </div>
    </div>
  );
};
