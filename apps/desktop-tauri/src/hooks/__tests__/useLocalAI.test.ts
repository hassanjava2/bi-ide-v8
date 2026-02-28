/**
 * Tests for useLocalAI hook
 * اختبارات هوك الذكاء الاصطناعي المحلي
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { useLocalAI, LocalAIOptions } from '../useLocalAI';
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';

// Mock Tauri APIs
jest.mock('@tauri-apps/api/tauri', () => ({
  invoke: jest.fn(),
}));

jest.mock('@tauri-apps/api/event', () => ({
  listen: jest.fn(),
}));

jest.mock('@tauri-apps/api/path', () => ({
  appDataDir: jest.fn().mockResolvedValue('/app/data'),
  join: jest.fn().mockImplementation((...parts) => parts.join('/')),
}));

describe('useLocalAI', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should initialize with idle status', () => {
    const { result } = renderHook(() => useLocalAI());

    expect(result.current.loadStatus).toBe('idle');
    expect(result.current.inferenceStatus).toBe('idle');
    expect(result.current.currentModel).toBeNull();
    expect(result.current.availableModels).toHaveLength(3);
  });

  it('should check if model exists', async () => {
    (invoke as jest.Mock).mockResolvedValue(true);

    const { result } = renderHook(() => useLocalAI());

    await waitFor(() => expect(result.current.modelsDir).toBeDefined());

    const exists = await act(async () => {
      return result.current.checkModelExists('llama-3.1-8b-q4');
    });

    expect(exists).toBe(true);
    expect(invoke).toHaveBeenCalledWith('check_model_exists', expect.any(Object));
  });

  it('should load model successfully', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true) // check_model_exists
      .mockResolvedValueOnce(undefined); // load_model

    const { result } = renderHook(() => useLocalAI());

    await waitFor(() => expect(result.current.modelsDir).toBeDefined());

    await act(async () => {
      await result.current.loadModel('llama-3.1-8b-q4');
    });

    expect(result.current.loadStatus).toBe('ready');
    expect(result.current.currentModel).not.toBeNull();
    expect(result.current.currentModel?.id).toBe('llama-3.1-8b-q4');
  });

  it('should download model if not exists', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(false) // check_model_exists
      .mockResolvedValueOnce(undefined); // download_model

    (listen as jest.Mock).mockResolvedValue(jest.fn());

    const { result } = renderHook(() => useLocalAI());

    await waitFor(() => expect(result.current.modelsDir).toBeDefined());

    await act(async () => {
      await result.current.loadModel('llama-3.1-8b-q4');
    });

    expect(invoke).toHaveBeenCalledWith('download_model', expect.any(Object));
  });

  it('should run inference successfully', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(true) // check_model_exists
      .mockResolvedValueOnce(undefined) // load_model
      .mockResolvedValueOnce('Generated text response'); // run_inference

    const { result } = renderHook(() => useLocalAI());

    await waitFor(() => expect(result.current.modelsDir).toBeDefined());

    await act(async () => {
      await result.current.loadModel('llama-3.1-8b-q4');
    });

    await act(async () => {
      await result.current.runInference('Hello, AI!');
    });

    expect(result.current.inferenceStatus).toBe('completed');
    expect(result.current.output).toBe('Generated text response');
  });

  it('should stop inference', async () => {
    (invoke as jest.Mock).mockResolvedValue(undefined);

    const { result } = renderHook(() => useLocalAI());

    act(() => {
      result.current.stopInference();
    });

    expect(invoke).toHaveBeenCalledWith('stop_inference');
    expect(result.current.isStreaming).toBe(false);
  });

  it('should unload model', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(undefined) // unload_model
      .mockResolvedValueOnce(true) // check_model_exists
      .mockResolvedValueOnce(undefined); // load_model

    const { result } = renderHook(() => useLocalAI());

    await waitFor(() => expect(result.current.modelsDir).toBeDefined());

    await act(async () => {
      await result.current.loadModel('llama-3.1-8b-q4');
    });

    await act(async () => {
      await result.current.unloadModel();
    });

    expect(invoke).toHaveBeenCalledWith('unload_model');
    expect(result.current.currentModel).toBeNull();
    expect(result.current.loadStatus).toBe('idle');
  });

  it('should delete model', async () => {
    (invoke as jest.Mock).mockResolvedValue(undefined);

    const { result } = renderHook(() => useLocalAI());

    await waitFor(() => expect(result.current.modelsDir).toBeDefined());

    await act(async () => {
      await result.current.deleteModel('llama-3.1-8b-q4');
    });

    expect(invoke).toHaveBeenCalledWith('delete_model', expect.any(Object));
  });

  it('should fallback to server if local fails', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ text: 'Server response' }),
    });

    (invoke as jest.Mock)
      .mockRejectedValueOnce(new Error('Local model failed'))
      .mockRejectedValueOnce(new Error('Local model failed')); // run_inference

    const { result } = renderHook(() => useLocalAI({
      fallbackToServer: true,
      serverEndpoint: 'http://localhost:3000/api/ai',
    }));

    await waitFor(() => expect(result.current.modelsDir).toBeDefined());

    await act(async () => {
      const response = await result.current.runInference('Hello');
      expect(response).toBe('Server response');
    });

    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:3000/api/ai/generate',
      expect.any(Object)
    );
  });

  it('should track download progress', async () => {
    const mockUnlisten = jest.fn();
    (listen as jest.Mock).mockImplementation((event, callback) => {
      if (event === 'model-download-progress') {
        setTimeout(() => {
          callback({
            payload: {
              downloaded: 1024 * 1024 * 100,
              total: 1024 * 1024 * 1000,
              speed: 1024 * 1024 * 5,
            },
          });
        }, 100);
      }
      return Promise.resolve(mockUnlisten);
    });

    (invoke as jest.Mock).mockResolvedValue(undefined);

    const { result } = renderHook(() => useLocalAI());

    await waitFor(() => expect(result.current.modelsDir).toBeDefined());

    const model = result.current.availableModels[0];
    
    act(() => {
      result.current.downloadModel(model);
    });

    await waitFor(() => {
      expect(result.current.downloadProgress).not.toBeNull();
    }, { timeout: 1000 });

    expect(result.current.loadStatus).toBe('downloading');
    expect(result.current.downloadProgress?.percentage).toBe(10);
  });

  it('should handle streaming inference', async () => {
    const mockUnlisten = jest.fn();
    const tokens = ['Hello', ' world', '!'];
    let tokenIndex = 0;

    (listen as jest.Mock).mockImplementation((event, callback) => {
      if (event === 'inference-token') {
        const interval = setInterval(() => {
          if (tokenIndex < tokens.length) {
            callback({
              payload: {
                token: tokens[tokenIndex],
                done: false,
              },
            });
            tokenIndex++;
          } else {
            callback({
              payload: {
                token: '',
                done: true,
              },
            });
            clearInterval(interval);
          }
        }, 50);
      }
      return Promise.resolve(mockUnlisten);
    });

    (invoke as jest.Mock).mockResolvedValue(undefined);

    const { result } = renderHook(() => useLocalAI());

    await waitFor(() => expect(result.current.modelsDir).toBeDefined());

    await act(async () => {
      await result.current.runInference('Hi', { stream: true });
    });

    await waitFor(() => {
      expect(result.current.output).toBe('Hello world!');
    }, { timeout: 1000 });

    expect(result.current.isStreaming).toBe(false);
  });

  it('should use default model on mount', async () => {
    (invoke as jest.Mock)
      .mockResolvedValueOnce(undefined) // create_models_directory
      .mockResolvedValueOnce(true) // check_model_exists
      .mockResolvedValueOnce(undefined); // load_model

    renderHook(() => useLocalAI({ defaultModelId: 'llama-3.1-8b-q4' }));

    await waitFor(() => {
      expect(invoke).toHaveBeenCalledWith('load_model', expect.any(Object));
    });
  });
});
