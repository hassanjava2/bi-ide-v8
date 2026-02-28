/**
 * Split View - عرض منقسم
 * 
 * يوفر محرر منقسم أفقياً/عمودياً مع أجزاء قابلة لتغيير الحجم
 * وخيار تزامن التمرير وملفات مختلفة بجانب بعضها وسحب التبويبات بين الأجزاء
 */

import React, {
  useState,
  useRef,
  useCallback,
  useEffect,
  ReactNode,
  MouseEvent as ReactMouseEvent,
  useMemo,
} from 'react';

/** اتجاه التقسيم */
export type SplitDirection = 'horizontal' | 'vertical';

/** حجم الجزء */
export interface PaneSize {
  /** النسبة المئوية (0-100) */
  percentage: number;
  /** الحجم بالبكسل (اختياري) */
  pixels?: number;
}

/** معلومات الجزء */
export interface PaneInfo {
  /** معرف فريد */
  id: string;
  /** المحتوى */
  content: ReactNode;
  /** عنوان الجزء */
  title?: string;
  /** هل الجزء نشط */
  isActive?: boolean;
  /** هل الجزء مؤمّن */
  isLocked?: boolean;
  /** الحد الأدنى للحجم (%) */
  minSize?: number;
  /** الحد الأقصى للحجم (%) */
  maxSize?: number;
}

/** إعدادات العرض المنقسم */
export interface SplitViewOptions {
  /** الاتجاه */
  direction?: SplitDirection;
  /** الأحجام الأولية */
  initialSizes?: number[];
  /** الحد الأدنى لحكل الجزء (%) */
  minPaneSize?: number;
  /** الحد الأقصى لحكل الجزء (%) */
  maxPaneSize?: number;
  /** عرض الفاصل */
  splitterWidth?: number;
  /** هل التقسيم قابل للتعديل */
  resizable?: boolean;
  /** تزامن التمرير */
  syncScroll?: boolean;
  /** السماح بسحب التبويبات */
  allowTabDrag?: boolean;
  /** رد الاتصال عند تغيير الحجم */
  onResize?: (sizes: number[]) => void;
  /** رد الاتصال عند تغيير الاتجاه */
  onDirectionChange?: (direction: SplitDirection) => void;
  /** رد الاتصال عند نقل تبويب */
  onTabMove?: (tabId: string, fromPane: string, toPane: string, index: number) => void;
}

/** خصائص مكون العرض المنقسم */
export interface SplitViewProps extends SplitViewOptions {
  /** الأجزاء */
  panes: PaneInfo[];
  /** فئة CSS */
  className?: string;
  /** نمط CSS */
  style?: React.CSSProperties;
}

/** حالة السحب */
interface DragState {
  /** هل السحب نشط */
  isDragging: boolean;
  /** معرف الفاصل */
  splitterId: string | null;
  /** موضع البداية */
  startPos: number;
  /** أحجام البداية */
  startSizes: number[];
}

/** حالة سحب التبويب */
interface TabDragState {
  /** هل السحب نشط */
  isDragging: boolean;
  /** معرف التبويب */
  tabId: string | null;
  /** معرف الجزء المصدر */
  sourcePaneId: string | null;
  /** معرف الجزء الهدف */
  targetPaneId: string | null;
  /** الفهرس الهدف */
  targetIndex: number;
}

/**
 * حساب نسبة من موضع الماوس
 */
function calculateRatio(
  position: number,
  containerSize: number,
  splitterWidth: number
): number {
  return Math.max(0, Math.min(100, (position / (containerSize - splitterWidth)) * 100));
}

/**
 * توزيع الأحجام بالتساوي
 */
function distributeEvenly(count: number): number[] {
  return Array(count).fill(100 / count);
}

/**
 * مكون العرض المنقسم
 */
export function SplitView({
  panes,
  direction = 'horizontal',
  initialSizes,
  minPaneSize = 10,
  maxPaneSize = 90,
  splitterWidth = 4,
  resizable = true,
  syncScroll = false,
  allowTabDrag = true,
  onResize,
  onDirectionChange,
  onTabMove,
  className,
  style,
}: SplitViewProps): JSX.Element {
  const [sizes, setSizes] = useState<number[]>(() => {
    if (initialSizes && initialSizes.length === panes.length) {
      return initialSizes;
    }
    return distributeEvenly(panes.length);
  });

  const [dragState, setDragState] = useState<DragState>({
    isDragging: false,
    splitterId: null,
    startPos: 0,
    startSizes: [],
  });

  const [tabDragState, setTabDragState] = useState<TabDragState>({
    isDragging: false,
    tabId: null,
    sourcePaneId: null,
    targetPaneId: null,
    targetIndex: -1,
  });

  const containerRef = useRef<HTMLDivElement>(null);
  const paneRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  // تحديث الأحجام عند تغيير عدد الأجزاء
  useEffect(() => {
    if (sizes.length !== panes.length) {
      setSizes(distributeEvenly(panes.length));
    }
  }, [panes.length, sizes.length]);

  /**
   * بدء سحب الفاصل
   */
  const handleSplitterMouseDown = useCallback((
    event: ReactMouseEvent,
    index: number
  ) => {
    if (!resizable) return;

    event.preventDefault();
    const isHorizontal = direction === 'horizontal';

    setDragState({
      isDragging: true,
      splitterId: `splitter-${index}`,
      startPos: isHorizontal ? event.clientX : event.clientY,
      startSizes: [...sizes],
    });
  }, [direction, resizable, sizes]);

  /**
   * معالجة حركة الماوس
   */
  useEffect(() => {
    if (!dragState.isDragging) return;

    const handleMouseMove = (event: MouseEvent) => {
      if (!containerRef.current) return;

      const isHorizontal = direction === 'horizontal';
      const containerRect = containerRef.current.getBoundingClientRect();
      const containerSize = isHorizontal ? containerRect.width : containerRect.height;

      const currentPos = isHorizontal ? event.clientX : event.clientY;
      const delta = currentPos - dragState.startPos;
      const deltaPercent = (delta / containerSize) * 100;

      // استخراج معرف الفاصل للحصول على الفهرس
      const splitterIndex = parseInt(dragState.splitterId?.split('-')[1] || '0', 10);

      setSizes(prevSizes => {
        const newSizes = [...prevSizes];
        const leftPane = newSizes[splitterIndex];
        const rightPane = newSizes[splitterIndex + 1];

        const newLeftSize = Math.max(
          minPaneSize,
          Math.min(maxPaneSize, dragState.startSizes[splitterIndex] + deltaPercent)
        );
        const sizeDiff = newLeftSize - leftPane;
        const newRightSize = Math.max(
          minPaneSize,
          Math.min(maxPaneSize, rightPane - sizeDiff)
        );

        // تعديل اليسار إذا تغير اليمين بشكل مختلف
        const actualDiff = newRightSize - rightPane;
        newSizes[splitterIndex] = leftPane - actualDiff;
        newSizes[splitterIndex + 1] = newRightSize;

        return newSizes;
      });
    };

    const handleMouseUp = () => {
      setDragState({
        isDragging: false,
        splitterId: null,
        startPos: 0,
        startSizes: [],
      });
      onResize?.(sizes);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [dragState, direction, minPaneSize, maxPaneSize, onResize, sizes]);

  /**
   * تبديل الاتجاه
   */
  const toggleDirection = useCallback(() => {
    const newDirection = direction === 'horizontal' ? 'vertical' : 'horizontal';
    onDirectionChange?.(newDirection);
  }, [direction, onDirectionChange]);

  /**
   * تزامن التمرير
   */
  useEffect(() => {
    if (!syncScroll || panes.length < 2) return;

    const syncScrollHandler = (sourcePane: HTMLDivElement) => {
      const scrollRatio = direction === 'horizontal'
        ? sourcePane.scrollLeft / (sourcePane.scrollWidth - sourcePane.clientWidth)
        : sourcePane.scrollTop / (sourcePane.scrollHeight - sourcePane.clientHeight);

      paneRefs.current.forEach((pane, id) => {
        if (pane !== sourcePane && pane) {
          if (direction === 'horizontal') {
            const maxScroll = pane.scrollWidth - pane.clientWidth;
            pane.scrollLeft = scrollRatio * maxScroll;
          } else {
            const maxScroll = pane.scrollHeight - pane.clientHeight;
            pane.scrollTop = scrollRatio * maxScroll;
          }
        }
      });
    };

    const handlers: Array<(e: Event) => void> = [];

    paneRefs.current.forEach((pane) => {
      if (pane) {
        const handler = () => syncScrollHandler(pane);
        pane.addEventListener('scroll', handler);
        handlers.push(handler);
      }
    });

    return () => {
      paneRefs.current.forEach((pane, index) => {
        if (pane && handlers[index]) {
          pane.removeEventListener('scroll', handlers[index]);
        }
      });
    };
  }, [syncScroll, direction, panes.length]);

  /**
   * بدء سحب التبويب
   */
  const handleTabDragStart = useCallback((
    event: React.DragEvent,
    tabId: string,
    paneId: string
  ) => {
    if (!allowTabDrag) return;

    event.dataTransfer.effectAllowed = 'move';
    event.dataTransfer.setData('text/plain', JSON.stringify({ tabId, paneId }));

    setTabDragState({
      isDragging: true,
      tabId,
      sourcePaneId: paneId,
      targetPaneId: null,
      targetIndex: -1,
    });
  }, [allowTabDrag]);

  /**
   * إفلات التبويب
   */
  const handleTabDrop = useCallback((
    event: React.DragEvent,
    targetPaneId: string,
    targetIndex: number
  ) => {
    event.preventDefault();

    try {
      const data = JSON.parse(event.dataTransfer.getData('text/plain'));
      const { tabId, paneId: sourcePaneId } = data;

      if (sourcePaneId !== targetPaneId || tabDragState.targetIndex !== targetIndex) {
        onTabMove?.(tabId, sourcePaneId, targetPaneId, targetIndex);
      }
    } catch {
      console.error('فشل معالجة إفلات التبويب');
    }

    setTabDragState({
      isDragging: false,
      tabId: null,
      sourcePaneId: null,
      targetPaneId: null,
      targetIndex: -1,
    });
  }, [onTabMove, tabDragState.targetIndex]);

  /**
   * السماح بالإفلات
   */
  const handleTabDragOver = useCallback((
    event: React.DragEvent,
    paneId: string,
    index: number
  ) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';

    setTabDragState(prev => ({
      ...prev,
      targetPaneId: paneId,
      targetIndex: index,
    }));
  }, []);

  // بناء الأنماط
  const containerStyle: React.CSSProperties = useMemo(() => ({
    display: 'flex',
    flexDirection: direction === 'horizontal' ? 'row' : 'column',
    width: '100%',
    height: '100%',
    overflow: 'hidden',
    ...style,
  }), [direction, style]);

  return (
    <div
      ref={containerRef}
      className={`split-view ${className || ''}`}
      style={containerStyle}
      data-direction={direction}
      data-sync-scroll={syncScroll}
    >
      {panes.map((pane, index) => {
        const isLast = index === panes.length - 1;
        const showSplitter = !isLast && resizable;

        return (
          <React.Fragment key={pane.id}>
            {/* الجزء */}
            <div
              ref={(el) => {
                if (el) paneRefs.current.set(pane.id, el);
              }}
              className={`split-pane ${pane.isActive ? 'active' : ''} ${pane.isLocked ? 'locked' : ''}`}
              style={{
                flex: `0 0 ${sizes[index]}%`,
                minWidth: direction === 'horizontal' ? `${minPaneSize}%` : undefined,
                minHeight: direction === 'vertical' ? `${minPaneSize}%` : undefined,
                overflow: 'auto',
                display: 'flex',
                flexDirection: 'column',
              }}
              onDragOver={(e) => handleTabDragOver(e, pane.id, index)}
              onDrop={(e) => handleTabDrop(e, pane.id, index)}
              data-pane-id={pane.id}
            >
              {/* رأس الجزء */}
              {(pane.title || allowTabDrag) && (
                <div
                  className="split-pane-header"
                  style={{
                    padding: '0.5rem 1rem',
                    borderBottom: '1px solid var(--color-border)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                  draggable={allowTabDrag}
                  onDragStart={(e) => handleTabDragStart(e, pane.id, pane.id)}
                >
                  <span className="split-pane-title">{pane.title || `جزء ${index + 1}`}</span>
                  {pane.isLocked && (
                    <span className="split-pane-lock-icon" title="مؤمّن">🔒</span>
                  )}
                </div>
              )}

              {/* محتوى الجزء */}
              <div className="split-pane-content" style={{ flex: 1, overflow: 'auto' }}>
                {pane.content}
              </div>
            </div>

            {/* الفاصل */}
            {showSplitter && (
              <div
                className={`split-splitter ${dragState.isDragging && dragState.splitterId === `splitter-${index}` ? 'dragging' : ''}`}
                style={{
                  flex: `0 0 ${splitterWidth}px`,
                  cursor: direction === 'horizontal' ? 'col-resize' : 'row-resize',
                  backgroundColor: dragState.isDragging && dragState.splitterId === `splitter-${index}`
                    ? 'var(--color-primary)'
                    : 'var(--color-border)',
                  userSelect: 'none',
                }}
                onMouseDown={(e) => handleSplitterMouseDown(e, index)}
                data-splitter-id={`splitter-${index}`}
              />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

/**
 * مكون تبويب قابل للسحب
 */
export interface DraggableTabProps {
  /** معرف التبويب */
  id: string;
  /** عنوان التبويب */
  title: string;
  /** هل التبويب نشط */
  isActive?: boolean;
  /** هل التبويب مُعدّل */
  isModified?: boolean;
  /** رد الاتصال عند النقر */
  onClick?: () => void;
  /** رد الاتصال عند الإغلاق */
  onClose?: () => void;
  /** رد الاتصال عند بدء السحب */
  onDragStart?: (event: React.DragEvent) => void;
  /** رد الاتصال عند الإفلات */
  onDrop?: (event: React.DragEvent) => void;
}

export function DraggableTab({
  id,
  title,
  isActive,
  isModified,
  onClick,
  onClose,
  onDragStart,
  onDrop,
}: DraggableTabProps): JSX.Element {
  return (
    <div
      className={`draggable-tab ${isActive ? 'active' : ''}`}
      style={{
        padding: '0.5rem 1rem',
        cursor: 'pointer',
        backgroundColor: isActive ? 'var(--color-surface)' : 'transparent',
        borderBottom: isActive ? '2px solid var(--color-primary)' : '2px solid transparent',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        userSelect: 'none',
      }}
      draggable
      onClick={onClick}
      onDragStart={onDragStart}
      onDrop={onDrop}
      data-tab-id={id}
    >
      {isModified && (
        <span style={{ color: 'var(--color-primary)' }}>•</span>
      )}
      <span className="tab-title">{title}</span>
      {onClose && (
        <button
          className="tab-close"
          onClick={(e) => {
            e.stopPropagation();
            onClose();
          }}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            padding: '0 0.25rem',
            opacity: 0.6,
          }}
          onMouseEnter={(e) => (e.currentTarget.style.opacity = '1')}
          onMouseLeave={(e) => (e.currentTarget.style.opacity = '0.6')}
        >
          ×
        </button>
      )}
    </div>
  );
}

/**
 * مكون مجموعة تبويبات
 */
export interface TabGroupProps {
  /** التبويبات */
  tabs: DraggableTabProps[];
  /** معرف الجزء */
  paneId: string;
  /** رد الاتصال عند نقل تبويب */
  onTabMove?: (tabId: string, fromIndex: number, toIndex: number) => void;
}

export function TabGroup({ tabs, paneId, onTabMove }: TabGroupProps): JSX.Element {
  const [draggedTab, setDraggedTab] = useState<string | null>(null);

  const handleDragStart = (event: React.DragEvent, tabId: string) => {
    setDraggedTab(tabId);
    event.dataTransfer.setData('text/plain', JSON.stringify({ tabId, paneId }));
  };

  const handleDrop = (event: React.DragEvent, targetIndex: number) => {
    event.preventDefault();
    
    try {
      const data = JSON.parse(event.dataTransfer.getData('text/plain'));
      const { tabId, paneId: sourcePaneId } = data;
      
      if (sourcePaneId === paneId) {
        const fromIndex = tabs.findIndex(t => t.id === tabId);
        if (fromIndex !== -1 && fromIndex !== targetIndex) {
          onTabMove?.(tabId, fromIndex, targetIndex);
        }
      }
    } catch {
      console.error('فشل معالجة إفلات التبويب');
    }
    
    setDraggedTab(null);
  };

  return (
    <div
      className="tab-group"
      style={{
        display: 'flex',
        overflowX: 'auto',
        borderBottom: '1px solid var(--color-border)',
      }}
    >
      {tabs.map((tab, index) => (
        <DraggableTab
          key={tab.id}
          {...tab}
          onDragStart={(e) => handleDragStart(e, tab.id)}
          onDrop={(e) => handleDrop(e, index)}
          style={{
            opacity: draggedTab === tab.id ? 0.5 : 1,
          }}
        />
      ))}
    </div>
  );
}

export default SplitView;
