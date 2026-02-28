/**
 * Local AI Hook - هوك الذكاء الاصطناعي المحلي
 * 
 * يقوم بتحميل النماذج المكممة محلياً وتشغيل الاستدلال بدون خادم
 * مع إدارة النماذج وتتبع التقدم والرجوع إلى الخادم إذا فشل المحلي
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';
import { appDataDir, join } from '@tauri-apps/api/path';

/** حالة تحميل النموذج */
export type ModelLoadStatus = 'idle' | 'checking' | 'downloading' | 'loading' | 'ready' | 'error';

/** حالة الاستدلال */
export type InferenceStatus = 'idle' | 'running' | 'streaming' | 'completed' | 'error';

/** معلومات النموذج */
export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  size: number;
  quantType: 'Q4_K_M' | 'Q5_K_M' | 'Q6_K' | 'Q8_0';
  contextLength: number;
  parameters: string;
  downloadUrl: string;
  checksum: string;
}

/** إعدادات الاستدلال */
export interface InferenceOptions {
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  topK?: number;
  repeatPenalty?: number;
  stopSequences?: string[];
  stream?: boolean;
}

/** تقدم تحميل النموذج */
export interface ModelDownloadProgress {
  downloaded: number;
  total: number;
  percentage: number;
  speed: number; // بايت/ثانية
  eta: number; // ثواني
}

/** خيارات هوك الذكاء الاصطناعي المحلي */
export interface LocalAIOptions {
  /** النموذج الافتراضي */
  defaultModelId?: string;
  /** مسار النماذج المخصص */
  modelsPath?: string;
  /** الرجوع إلى الخادم عند الفشل */
  fallbackToServer?: boolean;
  /** عنوان الخادم للرجوع */
  serverEndpoint?: string;
}

/** نتيجة هوك الذكاء الاصطناعي المحلي */
export interface UseLocalAIResult {
  /** حالة تحميل النموذج */
  loadStatus: ModelLoadStatus;
  /** حالة الاستدلال */
  inferenceStatus: InferenceStatus;
  /** معلومات النموذج الحالي */
  currentModel: ModelInfo | null;
  /** قائمة النماذج المتاحة */
  availableModels: ModelInfo[];
  /** تقدم التحميل */
  downloadProgress: ModelDownloadProgress | null;
  /** نتيجة الاستدلال */
  output: string;
  /** هل يقوم بتدفق النتيجة */
  isStreaming: boolean;
  /** رسالة الخطأ */
  error: string | null;
  /** تحميل نموذج */
  loadModel: (modelId: string) => Promise<void>;
  /** إلغاء تحميل النموذج */
  unloadModel: () => Promise<void>;
  /** تشغيل الاستدلال */
  runInference: (prompt: string, options?: InferenceOptions) => Promise<string>;
  /** إيقاف الاستدلال */
  stopInference: () => void;
  /** تنزيل نموذج جديد */
  downloadModel: (modelInfo: ModelInfo) => Promise<void>;
  /** حذف نموذج */
  deleteModel: (modelId: string) => Promise<void>;
  /** التحقق من وجود نموذج */
  checkModelExists: (modelId: string) => Promise<boolean>;
}

/** النماذج المتاحة افتراضياً */
const DEFAULT_MODELS: ModelInfo[] = [
  {
    id: 'llama-3.1-8b-q4',
    name: 'Llama 3.1 8B Q4',
    description: 'نموذج Meta Llama 3.1 مكمم 4-bit للاستخدام المحلي',
    size: 4.9 * 1024 * 1024 * 1024, // 4.9 GB
    quantType: 'Q4_K_M',
    contextLength: 8192,
    parameters: '8B',
    downloadUrl: 'https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
    checksum: 'sha256:abc123...',
  },
  {
    id: 'codellama-7b-q4',
    name: 'CodeLlama 7B Q4',
    description: 'نموذج CodeLlama مخصص للبرمجة مكمم 4-bit',
    size: 4.2 * 1024 * 1024 * 1024, // 4.2 GB
    quantType: 'Q4_K_M',
    contextLength: 16384,
    parameters: '7B',
    downloadUrl: 'https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/resolve/main/model-Q4_K_M.gguf',
    checksum: 'sha256:def456...',
  },
  {
    id: 'phi-3-mini-q4',
    name: 'Phi-3 Mini Q4',
    description: 'نموذج Microsoft Phi-3 صغير وسريع',
    size: 2.3 * 1024 * 1024 * 1024, // 2.3 GB
    quantType: 'Q4_K_M',
    contextLength: 4096,
    parameters: '3.8B',
    downloadUrl: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
    checksum: 'sha256:ghi789...',
  },
];

/** الإعدادات الافتراضية للاستدلال */
const DEFAULT_INFERENCE_OPTIONS: InferenceOptions = {
  temperature: 0.7,
  maxTokens: 2048,
  topP: 0.9,
  topK: 40,
  repeatPenalty: 1.1,
  stream: true,
};

/**
 * هوك الذكاء الاصطناعي المحلي
 * @param options - خيارات الهوك
 * @returns نتيجة التحكم بالذكاء الاصطناعي المحلي
 */
export function useLocalAI(options: LocalAIOptions = {}): UseLocalAIResult {
  const {
    defaultModelId,
    modelsPath,
    fallbackToServer = true,
    serverEndpoint = 'http://localhost:3000/api/ai',
  } = options;

  const [loadStatus, setLoadStatus] = useState<ModelLoadStatus>('idle');
  const [inferenceStatus, setInferenceStatus] = useState<InferenceStatus>('idle');
  const [currentModel, setCurrentModel] = useState<ModelInfo | null>(null);
  const [availableModels] = useState<ModelInfo[]>(DEFAULT_MODELS);
  const [downloadProgress, setDownloadProgress] = useState<ModelDownloadProgress | null>(null);
  const [output, setOutput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelsDir, setModelsDir] = useState<string>('');
  
  const abortControllerRef = useRef<AbortController | null>(null);
  const unlistenRef = useRef<(() => void) | null>(null);

  /**
   * تهيئة مسار النماذج
   */
  useEffect(() => {
    const initModelsDir = async () => {
      try {
        const appDir = await appDataDir();
        const dir = modelsPath || await join(appDir, 'models');
        setModelsDir(dir);
        
        // استدعاء Tauri لإنشاء المجلد
        await invoke('create_models_directory', { path: dir });
      } catch (err) {
        console.error('فشل تهيئة مجلد النماذج:', err);
      }
    };
    
    initModelsDir();
  }, [modelsPath]);

  /**
   * التحقق من وجود نموذج
   */
  const checkModelExists = useCallback(async (modelId: string): Promise<boolean> => {
    try {
      const exists = await invoke<boolean>('check_model_exists', {
        modelId,
        modelsDir,
      });
      return exists;
    } catch {
      return false;
    }
  }, [modelsDir]);

  /**
   * تحميل نموذج
   */
  const loadModel = useCallback(async (modelId: string) => {
    try {
      setLoadStatus('checking');
      setError(null);

      const model = availableModels.find((m) => m.id === modelId);
      if (!model) {
        throw new Error(`النموذج ${modelId} غير موجود`);
      }

      // التحقق من وجود النموذج
      const exists = await checkModelExists(modelId);
      
      if (!exists) {
        // تنزيل النموذج إذا لم يكن موجوداً
        await downloadModel(model);
      }

      setLoadStatus('loading');

      // تحميل النموذج عبر Tauri
      await invoke('load_model', {
        modelId,
        modelsDir,
        options: {
          n_ctx: model.contextLength,
          n_threads: 4,
        },
      });

      setCurrentModel(model);
      setLoadStatus('ready');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل تحميل النموذج');
      setLoadStatus('error');
    }
  }, [availableModels, modelsDir, checkModelExists]);

  /**
   * إلغاء تحميل النموذج
   */
  const unloadModel = useCallback(async () => {
    try {
      await invoke('unload_model');
      setCurrentModel(null);
      setLoadStatus('idle');
    } catch (err) {
      console.error('فشل إلغاء تحميل النموذج:', err);
    }
  }, []);

  /**
   * تنزيل نموذج
   */
  const downloadModel = useCallback(async (modelInfo: ModelInfo) => {
    try {
      setLoadStatus('downloading');
      setDownloadProgress({
        downloaded: 0,
        total: modelInfo.size,
        percentage: 0,
        speed: 0,
        eta: 0,
      });

      // الاستماع لتقدم التحميل
      const unlisten = await listen<{ downloaded: number; total: number; speed: number }>(
        'model-download-progress',
        (event) => {
          const { downloaded, total, speed } = event.payload;
          const remaining = total - downloaded;
          const eta = speed > 0 ? Math.round(remaining / speed) : 0;
          
          setDownloadProgress({
            downloaded,
            total,
            percentage: Math.round((downloaded / total) * 100),
            speed,
            eta,
          });
        }
      );

      unlistenRef.current = unlisten;

      // بدء التحميل عبر Tauri
      await invoke('download_model', {
        modelId: modelInfo.id,
        url: modelInfo.downloadUrl,
        checksum: modelInfo.checksum,
        modelsDir,
      });

      unlisten();
      setDownloadProgress(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل تنزيل النموذج');
      setLoadStatus('error');
      throw err;
    }
  }, [modelsDir]);

  /**
   * حذف نموذج
   */
  const deleteModel = useCallback(async (modelId: string) => {
    try {
      await invoke('delete_model', { modelId, modelsDir });
      
      if (currentModel?.id === modelId) {
        await unloadModel();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل حذف النموذج');
      throw err;
    }
  }, [modelsDir, currentModel, unloadModel]);

  /**
   * تشغيل الاستدلال
   */
  const runInference = useCallback(async (
    prompt: string,
    inferenceOptions: InferenceOptions = {}
  ): Promise<string> => {
    const options = { ...DEFAULT_INFERENCE_OPTIONS, ...inferenceOptions };
    
    try {
      setInferenceStatus('running');
      setOutput('');
      setError(null);

      // التحقق من وجود نموذج محمل
      if (!currentModel || loadStatus !== 'ready') {
        if (fallbackToServer) {
          // الرجوع إلى الخادم
          return await runServerInference(prompt, options);
        }
        throw new Error('لا يوجد نموذج محمل');
      }

      if (options.stream) {
        setIsStreaming(true);
        setInferenceStatus('streaming');

        // إعداد الاستماع للتدفق
        const unlisten = await listen<{ token: string; done: boolean }>(
          'inference-token',
          (event) => {
            const { token, done } = event.payload;
            
            if (done) {
              setIsStreaming(false);
              setInferenceStatus('completed');
              unlisten();
            } else {
              setOutput((prev) => prev + token);
            }
          }
        );

        // بدء الاستدلال
        await invoke('run_inference_stream', {
          prompt,
          options: {
            temperature: options.temperature,
            max_tokens: options.maxTokens,
            top_p: options.topP,
            top_k: options.topK,
            repeat_penalty: options.repeatPenalty,
            stop: options.stopSequences,
          },
        });

        return output;
      } else {
        // استدلال غير متدفق
        const result = await invoke<string>('run_inference', {
          prompt,
          options: {
            temperature: options.temperature,
            max_tokens: options.maxTokens,
            top_p: options.topP,
            top_k: options.topK,
            repeat_penalty: options.repeatPenalty,
            stop: options.stopSequences,
          },
        });

        setOutput(result);
        setInferenceStatus('completed');
        return result;
      }
    } catch (err) {
      if (fallbackToServer && loadStatus === 'ready') {
        return await runServerInference(prompt, options);
      }
      
      setError(err instanceof Error ? err.message : 'فشل الاستدلال');
      setInferenceStatus('error');
      throw err;
    }
  }, [currentModel, loadStatus, fallbackToServer, serverEndpoint, output]);

  /**
   * استدلال عبر الخادم (fallback)
   */
  const runServerInference = async (
    prompt: string,
    options: InferenceOptions
  ): Promise<string> => {
    abortControllerRef.current = new AbortController();

    const response = await fetch(`${serverEndpoint}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        ...options,
      }),
      signal: abortControllerRef.current.signal,
    });

    if (!response.ok) {
      throw new Error(`فشل الاستدلال عبر الخادم: ${response.statusText}`);
    }

    const data = await response.json();
    return data.text || data.response || '';
  };

  /**
   * إيقاف الاستدلال
   */
  const stopInference = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    invoke('stop_inference').catch(console.error);
    setIsStreaming(false);
    setInferenceStatus('idle');
  }, []);

  // تحميل النموذج الافتراضي
  useEffect(() => {
    if (defaultModelId && modelsDir) {
      loadModel(defaultModelId);
    }
  }, [defaultModelId, modelsDir, loadModel]);

  // تنظيف عند إلغاء التحميل
  useEffect(() => {
    return () => {
      if (unlistenRef.current) {
        unlistenRef.current();
      }
      unloadModel();
    };
  }, [unloadModel]);

  return {
    loadStatus,
    inferenceStatus,
    currentModel,
    availableModels,
    downloadProgress,
    output,
    isStreaming,
    error,
    loadModel,
    unloadModel,
    runInference,
    stopInference,
    downloadModel,
    deleteModel,
    checkModelExists,
  };
}

export default useLocalAI;
