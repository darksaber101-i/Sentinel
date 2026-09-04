"use client";
import { useState, useRef, useEffect } from "react";
import { Bot, Send, User } from "lucide-react";
import { api } from "@/lib/api";

interface Msg { role: "user" | "ai"; text: string; source?: string; }

const SUGGESTIONS = [
  "Why is this model suitable for return prediction?",
  "What is the model's precision?",
  "Which category has the highest return rate?",
  "How many high-risk orders do we have?",
  "Why not use an LLM as the prediction model?",
  "What are the biggest risk factors?",
];

export default function AssistantPage() {
  const [messages, setMessages] = useState<Msg[]>([
    { role: "ai", text: "Hi! I'm the Sentinel assistant. I can answer questions about model performance, risk factors, and order statistics — all grounded in real application data. What would you like to know?" },
  ]);
  const [input, setInput]   = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async (text: string) => {
    if (!text.trim() || loading) return;
    setMessages(m => [...m, { role: "user", text }]);
    setInput("");
    setLoading(true);
    try {
      const r = await api.askAssistant(text);
      setMessages(m => [...m, { role: "ai", text: r.answer, source: r.sources[0] }]);
    } catch {
      setMessages(m => [...m, { role: "ai", text: "Backend unavailable. Make sure the API is running." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-64px)] max-w-3xl">
      <div className="mb-5">
        <h1 className="text-2xl font-bold text-text-primary flex items-center gap-2">
          <Bot className="text-amber" size={22} /> AI Assistant
        </h1>
        <p className="text-text-secondary text-sm mt-1">
          Answers are grounded in real application data — it cannot invent statistics.
        </p>
      </div>

      {/* Suggestions */}
      {messages.length <= 1 && (
        <div className="grid grid-cols-2 gap-2 mb-5">
          {SUGGESTIONS.map(s => (
            <button
              key={s}
              onClick={() => send(s)}
              className="text-left px-4 py-3 bg-card border border-border rounded-xl text-xs text-text-secondary hover:text-text-primary hover:border-amber/30 transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4">
        {messages.map((m, i) => (
          <div key={i} className={`flex gap-3 ${m.role === "user" ? "flex-row-reverse" : ""}`}>
            <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${
              m.role === "ai" ? "bg-amber/10" : "bg-white/5"
            }`}>
              {m.role === "ai" ? <Bot size={14} className="text-amber" /> : <User size={14} className="text-text-secondary" />}
            </div>
            <div className={`max-w-[80%] ${m.role === "user" ? "items-end" : ""}`}>
              <div className={`px-4 py-3 rounded-2xl text-sm ${
                m.role === "ai"
                  ? "bg-card border border-border text-text-primary"
                  : "bg-amber/10 border border-amber/20 text-text-primary"
              }`}>
                {m.text}
              </div>
              {m.source && (
                <p className="text-[10px] text-text-muted mt-1 px-1">Source: {m.source}</p>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex gap-3">
            <div className="w-7 h-7 rounded-full bg-amber/10 flex items-center justify-center">
              <Bot size={14} className="text-amber" />
            </div>
            <div className="bg-card border border-border rounded-2xl px-4 py-3">
              <div className="flex gap-1">
                {[0, 1, 2].map(i => (
                  <span key={i} className="w-1.5 h-1.5 bg-amber-solid rounded-full animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                ))}
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex gap-3">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && !e.shiftKey && send(input)}
          placeholder="Ask about model performance, risk factors, orders…"
          className="flex-1 bg-card border border-border rounded-xl px-4 py-3 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-amber/50"
        />
        <button
          onClick={() => send(input)}
          disabled={loading || !input.trim()}
          className="p-3 bg-amber-solid text-black rounded-xl hover:bg-amber-dark transition-colors disabled:opacity-40"
        >
          <Send size={16} />
        </button>
      </div>
    </div>
  );
}
