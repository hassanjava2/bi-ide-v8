/**
 * Hooks Index - فهرس الهوكات
 * 
 * تصدير جميع الهوكات المتقدمة لتطبيق Tauri Desktop
 */

// Auto Update - التحديثات التلقائية
export {
  useAutoUpdate,
  type AutoUpdateOptions,
  type UpdateInfo,
  type UpdateStatus,
  type DownloadProgress,
  type UseAutoUpdateResult,
} from './useAutoUpdate';

// Offline Mode - وضع عدم الاتصال
export {
  useOfflineMode,
  type OfflineModeOptions,
  type ConnectionStatus,
  type QueuedAction,
  type CachedItem,
  type UseOfflineModeResult,
} from './useOfflineMode';

// Local AI - الذكاء الاصطناعي المحلي
export {
  useLocalAI,
  type LocalAIOptions,
  type ModelLoadStatus,
  type InferenceStatus,
  type ModelInfo,
  type InferenceOptions,
  type ModelDownloadProgress,
  type UseLocalAIResult,
} from './useLocalAI';

// File Watcher - مراقبة الملفات
export {
  useFileWatcher,
  type FileWatcherOptions,
  type FileStatus,
  type FileChangeType,
  type FileChangeEvent,
  type WatchedFile,
  type UseFileWatcherResult,
} from './useFileWatcher';

// Git Integration - التكامل مع Git
export {
  useGit,
  type GitOptions,
  type GitRepoStatus,
  type GitFileStatus,
  type BranchInfo,
  type GitFileInfo,
  type CommitInfo,
  type FileDiff,
  type DiffHunk,
  type DiffLine,
  type MergeConflict,
  type ConflictMarker,
  type UseGitResult,
  type BlameLine,
  type StashInfo,
} from './useGit';
