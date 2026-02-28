/**
 * Drag & Drop Provider - مزود السحب والإفلات
 * 
 * يوفر منطقة إفلات الملفات وأحداث بدء/انتهاء السحب والتحقق من الملفات
 * ومؤشر التقدم ودعم الملفات المتعددة
 */

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useRef,
  ReactNode,
  DragEvent,
} from 'react';

/** حالة السحب والإفلات */
export type DragDropState = 'idle' | 'dragging' | 'valid' | 'invalid' | 'dropping' | 'processing' | 'success' | 'error';

/** نوع الملف المسموح به */
export interface AllowedFileType {
  /** امتدادات الملفات المسموحة */
  extensions?: string[];
  /** أنواع MIME المسموحة */
  mimeTypes?: string[];
  /** الحد الأقصى للحجم (بايت) */
  maxSize?: number;
  /** الحد الأدنى للحجم (بايت) */
  minSize?: number;
  /** السماح بالمجلدات */
  allowDirectories?: boolean;
}

/** ملف مُفلات */
export interface DroppedFile {
  /** الملف الأصلي */
  file: File;
  /** المسار النسبي (للملفات داخل مجلدات) */
  path?: string;
  /** الحجم */
  size: number;
  /** النوع */
  type: string;
  /** اسم الملف */
  name: string;
  /** الامتداد */
  extension: string;
  /** هل هو مجلد */
  isDirectory: boolean;
}

/** نتيجة التحقق من الملف */
export interface FileValidationResult {
  /** هل الملف صالح */
  valid: boolean;
  /** سبب الرفض إن وجد */
  reason?: string;
  /** الملف */
  file: DroppedFile;
}

/** نتيجة عملية الإفلات */
export interface DropResult {
  /** الملفات المقبولة */
  acceptedFiles: DroppedFile[];
  /** الملفات المرفوضة */
  rejectedFiles: FileValidationResult[];
  /** إجمالي الملفات */
  totalFiles: number;
}

/** إعدادات المزود */
export interface DragDropProviderOptions {
  /** أنواع الملفات المسموحة */
  allowedTypes?: AllowedFileType;
  /** السماح بالملفات المتعددة */
  multiple?: boolean;
  /** رد الاتصال عند الإفلات */
  onDrop?: (result: DropResult) => void | Promise<void>;
  /** رد الاتصال عند دخول السحب */
  onDragEnter?: (event: DragEvent) => void;
  /** رد الاتصال عند مغادرة السحب */
  onDragLeave?: (event: DragEvent) => void;
  /** رد الاتصال عند السحب فوق */
  onDragOver?: (event: DragEvent) => void;
  /** رد الاتصال عند بدء السحب */
  onDragStart?: (event: DragEvent) => void;
  /** رد الاتصال عند انتهاء السحب */
  onDragEnd?: (event: DragEvent) => void;
  /** رد الاتصال عند التحقق */
  onValidate?: (file: DroppedFile) => boolean | string;
  /** منع السلوك الافتراضي */
  preventDefault?: boolean;
}

/** قيمة السياق */
export interface DragDropContextValue {
  /** الحالة الحالية */
  state: DragDropState;
  /** الملفات المسحوبة حالياً */
  draggedFiles: DroppedFile[];
  /** نسبة التقدم */
  progress: number;
  /** رسالة الخطأ */
  error: string | null;
  /** هل السحب نشط */
  isDragging: boolean;
  /** هل يمكن الإفلات */
  canDrop: boolean;
  /** معالج دخول السحب */
  handleDragEnter: (event: DragEvent) => void;
  /** معالج مغادرة السحب */
  handleDragLeave: (event: DragEvent) => void;
  /** معالج السحب فوق */
  handleDragOver: (event: DragEvent) => void;
  /** معالج الإفلات */
  handleDrop: (event: DragEvent) => void;
  /** معالج بدء السحب */
  handleDragStart: (event: DragEvent, files: DroppedFile[]) => void;
  /** معالج انتهاء السحب */
  handleDragEnd: (event: DragEvent) => void;
  /** إعادة تعيين الحالة */
  reset: () => void;
}

/** خصائص المزود */
export interface DragDropProviderProps extends DragDropProviderOptions {
  /** العناصر الفرعية */
  children: ReactNode;
  /** فئة CSS إضافية */
  className?: string;
  /** نمط CSS إضافي */
  style?: React.CSSProperties;
  /** عنصر المرجعية للمنطقة */
  containerRef?: React.RefObject<HTMLElement>;
}

/** إنشاء السياق */
const DragDropContext = createContext<DragDropContextValue | undefined>(undefined);

/**
 * استخراج امتداد الملف
 */
function getFileExtension(filename: string): string {
  const parts = filename.split('.');
  return parts.length > 1 ? parts.pop()?.toLowerCase() || '' : '';
}

/**
 * التحقق من صلاحية الملف
 */
function validateFile(
  file: DroppedFile,
  allowedTypes?: AllowedFileType,
  customValidator?: (file: DroppedFile) => boolean | string
): FileValidationResult {
  // التحقق المخصص
  if (customValidator) {
    const result = customValidator(file);
    if (result !== true) {
      return {
        valid: false,
        reason: typeof result === 'string' ? result : 'فشل التحقق المخصص',
        file,
      };
    }
  }

  if (!allowedTypes) {
    return { valid: true, file };
  }

  // التحقق من الامتداد
  if (allowedTypes.extensions && allowedTypes.extensions.length > 0) {
    if (!allowedTypes.extensions.includes(file.extension)) {
      return {
        valid: false,
        reason: `الامتداد .${file.extension} غير مسموح به`,
        file,
      };
    }
  }

  // التحقق من نوع MIME
  if (allowedTypes.mimeTypes && allowedTypes.mimeTypes.length > 0) {
    const isValidMime = allowedTypes.mimeTypes.some(type => {
      if (type.endsWith('/*')) {
        return file.type.startsWith(type.replace('/*', '/'));
      }
      return file.type === type;
    });

    if (!isValidMime) {
      return {
        valid: false,
        reason: `نوع الملف ${file.type} غير مسموح به`,
        file,
      };
    }
  }

  // التحقق من الحجم الأقصى
  if (allowedTypes.maxSize && file.size > allowedTypes.maxSize) {
    return {
      valid: false,
      reason: `حجم الملف (${(file.size / 1024 / 1024).toFixed(2)} MB) يتجاوز الحد الأقصى (${(allowedTypes.maxSize / 1024 / 1024).toFixed(2)} MB)`,
      file,
    };
  }

  // التحقق من الحجم الأدنى
  if (allowedTypes.minSize && file.size < allowedTypes.minSize) {
    return {
      valid: false,
      reason: `حجم الملف (${(file.size / 1024).toFixed(2)} KB) أقل من الحد الأدنى (${(allowedTypes.minSize / 1024).toFixed(2)} KB)`,
      file,
    };
  }

  return { valid: true, file };
}

/**
 * قراءة الملفات من DataTransfer
 */
async function readDroppedFiles(
  dataTransfer: DataTransfer,
  allowedTypes?: AllowedFileType
): Promise<DroppedFile[]> {
  const files: DroppedFile[] = [];

  // Chrome و Firefox يدعمان webkitGetAsEntry
  const items = dataTransfer.items;
  
  if (items && items.length > 0) {
    const entries: FileSystemEntry[] = [];
    
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      const entry = item.webkitGetAsEntry?.();
      if (entry) {
        entries.push(entry);
      }
    }

    // قراءة الإدخالات بشكل متكرر
    const readEntry = async (entry: FileSystemEntry, path: string = ''): Promise<void> => {
      if (entry.isFile) {
        const fileEntry = entry as FileSystemFileEntry;
        const file = await new Promise<File>((resolve) => {
          fileEntry.file(resolve);
        });
        
        files.push({
          file,
          path: path ? `${path}/${file.name}` : file.name,
          size: file.size,
          type: file.type,
          name: file.name,
          extension: getFileExtension(file.name),
          isDirectory: false,
        });
      } else if (entry.isDirectory && allowedTypes?.allowDirectories) {
        const dirEntry = entry as FileSystemDirectoryEntry;
        const reader = dirEntry.createReader();
        
        const readEntries = async (): Promise<void> => {
          const entries = await new Promise<FileSystemEntry[]>((resolve) => {
            reader.readEntries(resolve);
          });
          
          if (entries.length > 0) {
            for (const childEntry of entries) {
              await readEntry(childEntry, `${path}/${entry.name}`);
            }
            // قراءة المزيد إذا وجدت
            await readEntries();
          }
        };
        
        await readEntries();
      }
    };

    for (const entry of entries) {
      await readEntry(entry);
    }
  } else {
    // Fallback للمتصفحات التي لا تدعم webkitGetAsEntry
    const fileList = dataTransfer.files;
    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      files.push({
        file,
        size: file.size,
        type: file.type,
        name: file.name,
        extension: getFileExtension(file.name),
        isDirectory: false,
      });
    }
  }

  return files;
}

/**
 * مزود السحب والإفلات
 */
export function DragDropProvider({
  children,
  allowedTypes,
  multiple = true,
  onDrop,
  onDragEnter,
  onDragLeave,
  onDragOver,
  onDragStart,
  onDragEnd,
  onValidate,
  preventDefault = true,
  className,
  style,
  containerRef,
}: DragDropProviderProps): JSX.Element {
  const [state, setState] = useState<DragDropState>('idle');
  const [draggedFiles, setDraggedFiles] = useState<DroppedFile[]>([]);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  
  const dragCounter = useRef(0);
  const processingRef = useRef(false);

  /**
   * إعادة تعيين الحالة
   */
  const reset = useCallback(() => {
    setState('idle');
    setDraggedFiles([]);
    setProgress(0);
    setError(null);
    dragCounter.current = 0;
    processingRef.current = false;
  }, []);

  /**
   * معالج دخول السحب
   */
  const handleDragEnter = useCallback((event: DragEvent) => {
    if (preventDefault) {
      event.preventDefault();
      event.stopPropagation();
    }

    dragCounter.current++;

    if (dragCounter.current === 1) {
      setState('dragging');
      onDragEnter?.(event);
    }

    // التحقق من أنواع الملفات
    if (event.dataTransfer) {
      const types = event.dataTransfer.types;
      const hasFiles = types.includes('Files');
      
      if (hasFiles) {
        // محاولة معاينة الملفات
        readDroppedFiles(event.dataTransfer, allowedTypes)
          .then(files => {
            setDraggedFiles(files);
            const hasValidFiles = files.some(f => 
              validateFile(f, allowedTypes, onValidate).valid
            );
            setState(hasValidFiles ? 'valid' : 'invalid');
          })
          .catch(() => {
            setState('valid'); // افتراضياً نفترض صلاحية
          });
      } else {
        setState('invalid');
      }
    }
  }, [preventDefault, onDragEnter, allowedTypes, onValidate]);

  /**
   * معالج مغادرة السحب
   */
  const handleDragLeave = useCallback((event: DragEvent) => {
    if (preventDefault) {
      event.preventDefault();
      event.stopPropagation();
    }

    dragCounter.current--;

    if (dragCounter.current === 0) {
      setState('idle');
      setDraggedFiles([]);
      onDragLeave?.(event);
    }
  }, [preventDefault, onDragLeave]);

  /**
   * معالج السحب فوق
   */
  const handleDragOver = useCallback((event: DragEvent) => {
    if (preventDefault) {
      event.preventDefault();
      event.stopPropagation();
    }

    // تغيير شكل المؤشر
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = state === 'invalid' ? 'none' : 'copy';
    }

    onDragOver?.(event);
  }, [preventDefault, onDragOver, state]);

  /**
   * معالج الإفلات
   */
  const handleDrop = useCallback(async (event: DragEvent) => {
    if (preventDefault) {
      event.preventDefault();
      event.stopPropagation();
    }

    if (processingRef.current) return;
    processingRef.current = true;

    dragCounter.current = 0;
    setState('dropping');

    try {
      if (!event.dataTransfer) {
        throw new Error('لا توجد بيانات للإفلات');
      }

      // قراءة الملفات
      const files = await readDroppedFiles(event.dataTransfer, allowedTypes);
      
      if (!multiple && files.length > 1) {
        files.splice(1); // الاحتفاظ بالملف الأول فقط
      }

      setDraggedFiles(files);
      setState('processing');

      // التحقق من الملفات
      const acceptedFiles: DroppedFile[] = [];
      const rejectedFiles: FileValidationResult[] = [];

      for (const file of files) {
        const validation = validateFile(file, allowedTypes, onValidate);
        if (validation.valid) {
          acceptedFiles.push(file);
        } else {
          rejectedFiles.push(validation);
        }

        // تحديث التقدم
        setProgress(Math.round(((acceptedFiles.length + rejectedFiles.length) / files.length) * 100));
      }

      const result: DropResult = {
        acceptedFiles,
        rejectedFiles,
        totalFiles: files.length,
      };

      // استدعاء رد الاتصال
      await onDrop?.(result);

      setState(acceptedFiles.length > 0 ? 'success' : 'error');
      
      if (rejectedFiles.length > 0) {
        setError(`${rejectedFiles.length} ملف/ملفات مرفوضة`);
      }

      // إعادة تعيين بعد فترة
      setTimeout(reset, 2000);
    } catch (err) {
      setState('error');
      setError(err instanceof Error ? err.message : 'فشل معالجة الإفلات');
      setTimeout(reset, 3000);
    } finally {
      processingRef.current = false;
    }
  }, [preventDefault, onDrop, allowedTypes, multiple, onValidate, reset]);

  /**
   * معالج بدء السحب
   */
  const handleDragStart = useCallback((event: DragEvent, files: DroppedFile[]) => {
    if (event.dataTransfer) {
      // إعداد بيانات السحب
      event.dataTransfer.effectAllowed = 'copy';
      
      // إضافة أنواع البيانات
      files.forEach(file => {
        if (!file.isDirectory) {
          event.dataTransfer?.items.add(file.file);
        }
      });
    }

    setDraggedFiles(files);
    setState('dragging');
    onDragStart?.(event);
  }, [onDragStart]);

  /**
   * معالج انتهاء السحب
   */
  const handleDragEnd = useCallback((event: DragEvent) => {
    setState('idle');
    setDraggedFiles([]);
    onDragEnd?.(event);
  }, [onDragEnd]);

  const value: DragDropContextValue = {
    state,
    draggedFiles,
    progress,
    error,
    isDragging: state === 'dragging' || state === 'valid' || state === 'invalid',
    canDrop: state === 'valid',
    handleDragEnter,
    handleDragLeave,
    handleDragOver,
    handleDrop,
    handleDragStart,
    handleDragEnd,
    reset,
  };

  return (
    <DragDropContext.Provider value={value}>
      <div
        className={className}
        style={{
          ...style,
          pointerEvents: state === 'processing' ? 'none' : undefined,
        }}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {children}
      </div>
    </DragDropContext.Provider>
  );
}

/**
 * هوك استخدام سياق السحب والإفلات
 */
export function useDragDrop(): DragDropContextValue {
  const context = useContext(DragDropContext);
  
  if (context === undefined) {
    throw new Error('useDragDrop must be used within a DragDropProvider');
  }
  
  return context;
}

/**
 * مكون منطقة الإفلات
 */
export interface DropZoneProps {
  /** العناصر الفرعية */
  children: ReactNode;
  /** أنواع الملفات المسموحة */
  allowedTypes?: AllowedFileType;
  /** رد الاتصال عند الإفلات */
  onDrop?: (result: DropResult) => void | Promise<void>;
  /** فئة CSS */
  className?: string;
  /** نمط CSS */
  style?: React.CSSProperties;
  /** نص التلميح */
  hintText?: string;
  /** نص السحب النشط */
  activeText?: string;
  /** نص الخطأ */
  errorText?: string;
}

export function DropZone({
  children,
  allowedTypes,
  onDrop,
  className,
  style,
  hintText = 'اسحب الملفات هنا أو انقر للاختيار',
  activeText = 'أفلت الملفات هنا',
  errorText = 'هذا النوع من الملفات غير مسموح به',
}: DropZoneProps): JSX.Element {
  const { state, isDragging, canDrop, progress, error, handleDragEnter, handleDrop } = useDragDrop();

  const getStateText = () => {
    if (state === 'invalid') return errorText;
    if (state === 'processing') return `جاري المعالجة... ${progress}%`;
    if (state === 'success') return 'تم بنجاح!';
    if (state === 'error') return error || 'حدث خطأ';
    if (isDragging) return canDrop ? activeText : errorText;
    return hintText;
  };

  const getStateClassName = () => {
    if (state === 'invalid') return 'drag-drop-error';
    if (state === 'valid') return 'drag-drop-valid';
    if (state === 'processing') return 'drag-drop-processing';
    if (state === 'success') return 'drag-drop-success';
    if (isDragging) return 'drag-drop-active';
    return 'drag-drop-idle';
  };

  return (
    <div
      className={`${className || ''} ${getStateClassName()}`}
      style={{
        ...style,
        border: '2px dashed',
        borderColor: state === 'invalid' ? '#ef4444' : state === 'valid' ? '#10b981' : '#d1d5db',
        borderRadius: '0.5rem',
        padding: '2rem',
        textAlign: 'center',
        transition: 'all 0.2s ease',
      }}
      onDragEnter={handleDragEnter}
      onDrop={handleDrop}
    >
      <p>{getStateText()}</p>
      {children}
    </div>
  );
}

export default DragDropContext;
