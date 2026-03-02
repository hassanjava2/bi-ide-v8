//! AI Model Settings

import { useState } from "react";
import { Cpu, Server, Cloud, Check, AlertCircle } from "lucide-react";
import { useStore } from "../../lib/store";

interface ModelConfig {
  id: string;
  name: string;
  provider: "local" | "remote";
  description: string;
  max_tokens: number;
  latency: string;
}

const MODELS: ModelConfig[] = [
  {
    id: "local-llama",
    name: "Llama 3.2 (Local)",
    provider: "local",
    description: "Runs on your machine - private & offline",
    max_tokens: 4096,
    latency: "~100ms",
  },
  {
    id: "rtx5090-coder",
    name: "DeepSeek Coder (RTX 5090)",
    provider: "remote",
    description: "Running on RTX 5090 workstation",
    max_tokens: 8192,
    latency: "~200ms",
  },
  {
    id: "cloud-gpt4",
    name: "GPT-4 (Cloud)",
    provider: "remote",
    description: "OpenAI GPT-4 via API",
    max_tokens: 8192,
    latency: "~500ms",
  },
];

export function ModelSettings() {
  const { settings, updateSettings } = useStore();
  const [selectedModel, setSelectedModel] = useState(settings.aiModel || "rtx5090-coder");
  const [aiEnabled, setAiEnabled] = useState(settings.aiEnabled ?? true);

  const handleModelChange = (modelId: string) => {
    setSelectedModel(modelId);
    updateSettings({ aiModel: modelId });
  };

  const handleToggleAI = (enabled: boolean) => {
    setAiEnabled(enabled);
    updateSettings({ aiEnabled: enabled });
  };

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-lg font-medium mb-1">AI Model Settings</h2>
        <p className="text-sm text-dark-400">Configure AI assistance for code completion and chat</p>
      </div>

      {/* Enable AI */}
      <div className="flex items-center justify-between p-4 bg-dark-800 rounded border border-dark-700">
        <div>
          <div className="font-medium">Enable AI Assistance</div>
          <div className="text-sm text-dark-400">Code completion, explanation, and refactoring</div>
        </div>
        <label className="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={aiEnabled}
            onChange={(e) => handleToggleAI(e.target.checked)}
            className="sr-only peer"
          />
          <div className="w-11 h-6 bg-dark-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
        </label>
      </div>

      {/* Model Selection */}
      <div className="space-y-3">
        <label className="text-sm font-medium text-dark-400">Select Model</label>
        <div className="space-y-2">
          {MODELS.map((model) => (
            <button
              key={model.id}
              onClick={() => handleModelChange(model.id)}
              disabled={!aiEnabled}
              className={`w-full p-4 rounded border text-left transition-all
                ${selectedModel === model.id 
                  ? "border-primary-500 bg-primary-600/10" 
                  : "border-dark-700 hover:border-dark-600"
                }
                ${!aiEnabled ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5">
                  {model.provider === "local" ? (
                    <Cpu className="w-5 h-5 text-green-400" />
                  ) : (
                    <Cloud className="w-5 h-5 text-blue-400" />
                  )}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{model.name}</span>
                    {selectedModel === model.id && (
                      <Check className="w-4 h-4 text-primary-400" />
                    )}
                  </div>
                  <p className="text-sm text-dark-400 mt-0.5">{model.description}</p>
                  <div className="flex items-center gap-4 mt-2 text-xs text-dark-500">
                    <span>{model.max_tokens.toLocaleString()} tokens</span>
                    <span>Latency: {model.latency}</span>
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Settings */}
      <div className="pt-4 border-t border-dark-700">
        <h3 className="text-sm font-medium mb-3">Advanced Settings</h3>
        
        <div className="space-y-4">
          <div>
            <label className="text-sm text-dark-400 block mb-1">Temperature</label>
            <input
              type="range"
              min="0"
              max="100"
              defaultValue={70}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-dark-500 mt-1">
              <span>Precise</span>
              <span>Creative</span>
            </div>
          </div>

          <div>
            <label className="text-sm text-dark-400 block mb-1">Max Completion Length</label>
            <select className="w-full p-2 bg-dark-800 border border-dark-700 rounded text-sm">
              <option>128 tokens</option>
              <option>256 tokens</option>
              <option selected>512 tokens</option>
              <option>1024 tokens</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <input type="checkbox" id="auto-trigger" defaultChecked className="rounded border-dark-600" />
            <label htmlFor="auto-trigger" className="text-sm">Auto-trigger on pause typing</label>
          </div>
        </div>
      </div>

      {/* Privacy Notice */}
      <div className="p-3 bg-yellow-900/20 border border-yellow-700 rounded flex items-start gap-2">
        <AlertCircle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
        <p className="text-xs text-yellow-200">
          Local models keep your code private. Remote models may process code on external servers.
        </p>
      </div>
    </div>
  );
}
