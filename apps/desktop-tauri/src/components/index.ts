/**
 * Components Index - فهرس المكونات
 * 
 * تصدير جميع المكونات المشتركة
 */

// Common Components - المكونات المشتركة
export {
  DragDropProvider,
  useDragDrop,
  DropZone,
  type DragDropState,
  type AllowedFileType,
  type DroppedFile,
  type FileValidationResult,
  type DropResult,
  type DragDropProviderOptions,
  type DragDropContextValue,
  type DragDropProviderProps,
  type DropZoneProps,
} from './common/DragDropProvider';

// Editor Components - مكونات المحرر
export {
  SplitView,
  DraggableTab,
  TabGroup,
  type SplitDirection,
  type PaneSize,
  type PaneInfo,
  type SplitViewOptions,
  type SplitViewProps,
  type DraggableTabProps,
  type TabGroupProps,
} from './editor/SplitView';
