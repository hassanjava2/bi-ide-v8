/**
 * مستكشف الملفات - File Explorer
 * عرض شجري للملفات مع دعم السحب والإفلات والقائمة السياقية
 */

import { useState, useCallback, useEffect, useRef } from "react";
import { 
  Folder, 
  FolderOpen, 
  File, 
  ChevronRight, 
  ChevronDown,
  MoreVertical,
  Plus,
  Trash2,
  Edit3,
  Copy,
  Scissors,
  RefreshCw,
  Search,
  Filter,
  FileCode,
  FileText,
  Image as ImageIcon,
  FileJson,
  FileType,
  Download,
  Upload
} from "lucide-react";

// أنواع البيانات
interface FileNode {
  id: string;
  name: string;
  path: string;
  type: "file" | "directory";
  size?: number;
  modifiedAt: Date;
  isExpanded?: boolean;
  children?: FileNode[];
}

interface FileOperation {
  type: "copy" | "cut";
  node: FileNode;
}

// ربط امتدادات الملفات بالأيقونات
const fileIcons: Record<string, React.ElementType> = {
  ts: FileCode,
  tsx: FileCode,
  js: FileCode,
  jsx: FileCode,
  json: FileJson,
  md: FileText,
  txt: FileText,
  png: ImageIcon,
  jpg: ImageIcon,
  jpeg: ImageIcon,
  svg: ImageIcon,
  default: FileType,
};

// الحصول على أيقونة الملف
function getFileIcon(filename: string): React.ElementType {
  const ext = filename.split('.').pop()?.toLowerCase() || '';
  return fileIcons[ext] || fileIcons.default;
}

// تنسيق حجم الملف
function formatFileSize(bytes?: number): string {
  if (bytes === undefined) return "";
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

// مكون عنصر الشجرة
interface TreeItemProps {
  node: FileNode;
  depth: number;
  selectedId: string | null;
  onSelect: (node: FileNode) => void;
  onToggle: (node: FileNode) => void;
  onContextMenu: (e: React.MouseEvent, node: FileNode) => void;
  draggedNode: FileNode | null;
  onDragStart: (node: FileNode) => void;
  onDragEnd: () => void;
  onDrop: (targetNode: FileNode) => void;
  searchQuery: string;
}

function TreeItem({ 
  node, 
  depth, 
  selectedId, 
  onSelect, 
  onToggle, 
  onContextMenu,
  draggedNode,
  onDragStart,
  onDragEnd,
  onDrop,
  searchQuery
}: TreeItemProps) {
  const isSelected = selectedId === node.id;
  const isDragged = draggedNode?.id === node.id;
  const isDirectory = node.type === "directory";
  const FileIcon = isDirectory 
    ? (node.isExpanded ? FolderOpen : Folder) 
    : getFileIcon(node.name);

  // تظليل نتائج البحث
  const highlightText = (text: string, query: string) => {
    if (!query) return text;
    const parts = text.split(new RegExp(`(${query})`, "gi"));
    return parts.map((part, i) => 
      part.toLowerCase() === query.toLowerCase() 
        ? <mark key={i} className="bg-yellow-500/30 text-yellow-200 rounded px-0.5">{part}</mark>
        : part
    );
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (isDirectory) {
      e.dataTransfer.dropEffect = "move";
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (isDirectory) {
      onDrop(node);
    }
  };

  return (
    <div>
      <div
        draggable
        onDragStart={() => onDragStart(node)}
        onDragEnd={onDragEnd}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => onSelect(node)}
        onContextMenu={(e) => onContextMenu(e, node)}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        className={`
          group flex items-center gap-2 py-1.5 pr-2 cursor-pointer select-none
          transition-colors duration-150
          ${isSelected ? "bg-primary-600/20 text-primary-400" : "text-dark-300 hover:bg-dark-800"}
          ${isDragged ? "opacity-50" : ""}
        `}
      >
        {/* أيقونة التوسيع */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onToggle(node);
          }}
          className={`
            w-4 h-4 flex items-center justify-center rounded hover:bg-dark-700
            ${isDirectory ? "visible" : "invisible"}
          `}
        >
          {node.isExpanded ? (
            <ChevronDown className="w-3.5 h-3.5 text-dark-500" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 text-dark-500" />
          )}
        </button>

        {/* أيقونة الملف/مجلد */}
        <FileIcon className={`w-4 h-4 flex-shrink-0 ${
          isDirectory 
            ? node.isExpanded ? "text-yellow-400" : "text-yellow-500"
            : "text-dark-400"
        }`} />

        {/* اسم الملف */}
        <span className="flex-1 truncate text-sm">
          {highlightText(node.name, searchQuery)}
        </span>

        {/* حجم الملف */}
        {node.size !== undefined && (
          <span className="text-xs text-dark-500">
            {formatFileSize(node.size)}
          </span>
        )}
      </div>

      {/* الأطفال */}
      {isDirectory && node.isExpanded && node.children && (
        <div>
          {node.children.map(child => (
            <TreeItem
              key={child.id}
              node={child}
              depth={depth + 1}
              selectedId={selectedId}
              onSelect={onSelect}
              onToggle={onToggle}
              onContextMenu={onContextMenu}
              draggedNode={draggedNode}
              onDragStart={onDragStart}
              onDragEnd={onDragEnd}
              onDrop={onDrop}
              searchQuery={searchQuery}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// القائمة السياقية
function ContextMenu({
  x,
  y,
  node,
  onClose,
  onNewFile,
  onNewFolder,
  onRename,
  onDelete,
  onCopy,
  onCut,
  onPaste,
  canPaste,
}: {
  x: number;
  y: number;
  node: FileNode;
  onClose: () => void;
  onNewFile: () => void;
  onNewFolder: () => void;
  onRename: () => void;
  onDelete: () => void;
  onCopy: () => void;
  onCut: () => void;
  onPaste: () => void;
  canPaste: boolean;
}) {
  const menuRef = useRef<HTMLDivElement>(null);
  const isDirectory = node.type === "directory";

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [onClose]);

  const menuItems = [
    isDirectory && { label: "ملف جديد", icon: Plus, action: onNewFile },
    isDirectory && { label: "مجلد جديد", icon: Folder, action: onNewFolder },
    isDirectory && { label: null }, // فاصل
    { label: "نسخ", icon: Copy, action: onCopy },
    { label: "قص", icon: Scissors, action: onCut },
    canPaste && isDirectory && { label: "لصق", icon: Copy, action: onPaste },
    { label: null }, // فاصل
    { label: "إعادة تسمية", icon: Edit3, action: onRename },
    { label: "حذف", icon: Trash2, action: onDelete, danger: true },
  ].filter(Boolean) as Array<{ label: string | null; icon?: React.ElementType; action?: () => void; danger?: boolean }>;

  return (
    <div
      ref={menuRef}
      style={{ top: y, left: x }}
      className="fixed z-50 min-w-48 bg-dark-800 border border-dark-700 rounded-lg shadow-xl py-1"
    >
      {menuItems.map((item, idx) =>
        item.label === null ? (
          <div key={idx} className="my-1 border-t border-dark-700" />
        ) : (
          <button
            key={idx}
            onClick={() => {
              item.action?.();
              onClose();
            }}
            className={`
              w-full flex items-center gap-2 px-3 py-2 text-sm text-left transition-colors
              ${item.danger 
                ? "text-red-400 hover:bg-red-500/10" 
                : "text-dark-200 hover:bg-dark-700"
              }
            `}
          >
            {item.icon && <item.icon className="w-4 h-4" />}
            {item.label}
          </button>
        )
      )}
    </div>
  );
}

// المكون الرئيسي
export function FileExplorer() {
  // حالة الملفات
  const [files, setFiles] = useState<FileNode[]>([
    {
      id: "1",
      name: "src",
      path: "/src",
      type: "directory",
      modifiedAt: new Date(),
      isExpanded: true,
      children: [
        {
          id: "1-1",
          name: "components",
          path: "/src/components",
          type: "directory",
          modifiedAt: new Date(),
          isExpanded: true,
          children: [
            { id: "1-1-1", name: "Header.tsx", path: "/src/components/Header.tsx", type: "file", size: 2048, modifiedAt: new Date() },
            { id: "1-1-2", name: "Sidebar.tsx", path: "/src/components/Sidebar.tsx", type: "file", size: 3584, modifiedAt: new Date() },
          ],
        },
        { id: "1-2", name: "App.tsx", path: "/src/App.tsx", type: "file", size: 1536, modifiedAt: new Date() },
        { id: "1-3", name: "main.tsx", path: "/src/main.tsx", type: "file", size: 512, modifiedAt: new Date() },
      ],
    },
    {
      id: "2",
      name: "public",
      path: "/public",
      type: "directory",
      modifiedAt: new Date(),
      isExpanded: false,
      children: [
        { id: "2-1", name: "index.html", path: "/public/index.html", type: "file", size: 1024, modifiedAt: new Date() },
        { id: "2-2", name: "favicon.ico", path: "/public/favicon.ico", type: "file", size: 256, modifiedAt: new Date() },
      ],
    },
    { id: "3", name: "package.json", path: "/package.json", type: "file", size: 2048, modifiedAt: new Date() },
    { id: "4", name: "tsconfig.json", path: "/tsconfig.json", type: "file", size: 1024, modifiedAt: new Date() },
    { id: "5", name: "README.md", path: "/README.md", type: "file", size: 5120, modifiedAt: new Date() },
  ]);

  // حالة التحديد والبحث
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; node: FileNode } | null>(null);
  const [draggedNode, setDraggedNode] = useState<FileNode | null>(null);
  const [clipboard, setClipboard] = useState<FileOperation | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [newName, setNewName] = useState("");

  // توسيع/طي المجلد
  const toggleFolder = useCallback((node: FileNode) => {
    if (node.type !== "directory") return;
    
    const updateNode = (nodes: FileNode[]): FileNode[] => {
      return nodes.map(n => {
        if (n.id === node.id) {
          return { ...n, isExpanded: !n.isExpanded };
        }
        if (n.children) {
          return { ...n, children: updateNode(n.children) };
        }
        return n;
      });
    };

    setFiles(updateNode(files));
  }, [files]);

  // البحث في الملفات
  const searchFiles = useCallback((nodes: FileNode[], query: string): FileNode[] => {
    if (!query) return nodes;
    
    return nodes.reduce<FileNode[]>((acc, node) => {
      const matches = node.name.toLowerCase().includes(query.toLowerCase());
      const matchingChildren = node.children ? searchFiles(node.children, query) : [];
      
      if (matches || matchingChildren.length > 0) {
        acc.push({
          ...node,
          isExpanded: true,
          children: matchingChildren.length > 0 ? matchingChildren : node.children,
        });
      }
      
      return acc;
    }, []);
  }, []);

  const filteredFiles = searchQuery ? searchFiles(files, searchQuery) : files;

  // معالجات السحب والإفلات
  const handleDragStart = (node: FileNode) => {
    setDraggedNode(node);
  };

  const handleDragEnd = () => {
    setDraggedNode(null);
  };

  const handleDrop = (targetNode: FileNode) => {
    if (!draggedNode || draggedNode.id === targetNode.id) return;
    console.log(`Moving ${draggedNode.name} to ${targetNode.name}`);
    // في الإنتاج: استدعاء API لنقل الملف
  };

  // إنشاء ملف/مجلد جديد
  const handleNewFile = () => {
    if (!contextMenu) return;
    const parentNode = contextMenu.node;
    console.log("Creating new file in", parentNode.path);
  };

  const handleNewFolder = () => {
    if (!contextMenu) return;
    const parentNode = contextMenu.node;
    console.log("Creating new folder in", parentNode.path);
  };

  // نسخ/قص/لصق
  const handleCopy = () => {
    if (!contextMenu) return;
    setClipboard({ type: "copy", node: contextMenu.node });
  };

  const handleCut = () => {
    if (!contextMenu) return;
    setClipboard({ type: "cut", node: contextMenu.node });
  };

  const handlePaste = () => {
    if (!contextMenu || !clipboard) return;
    console.log(`Pasting ${clipboard.node.name} to ${contextMenu.node.path}`);
  };

  // إعادة تسمية
  const handleRename = () => {
    if (!contextMenu) return;
    setRenamingId(contextMenu.node.id);
    setNewName(contextMenu.node.name);
  };

  // حذف
  const handleDelete = () => {
    if (!contextMenu) return;
    const confirmed = window.confirm(`هل أنت متأكد من حذف "${contextMenu.node.name}"؟`);
    if (confirmed) {
      console.log("Deleting", contextMenu.node.path);
    }
  };

  return (
    <div className="h-full flex flex-col bg-dark-900">
      {/* رأس المستكشف */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-dark-700">
        <span className="text-sm font-semibold text-dark-400 uppercase tracking-wide">
          المستكشف
        </span>
        <div className="flex items-center gap-1">
          <button 
            className="p-1.5 text-dark-400 hover:text-dark-200 hover:bg-dark-800 rounded transition-colors"
            title="ملف جديد"
          >
            <Plus className="w-4 h-4" />
          </button>
          <button 
            className="p-1.5 text-dark-400 hover:text-dark-200 hover:bg-dark-800 rounded transition-colors"
            title="تحديث"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button 
            className="p-1.5 text-dark-400 hover:text-dark-200 hover:bg-dark-800 rounded transition-colors"
            title="المزيد"
          >
            <MoreVertical className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* شريط البحث */}
      <div className="px-3 py-2 border-b border-dark-700">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="البحث في الملفات..."
            className="w-full pl-9 pr-8 py-1.5 bg-dark-800 border border-dark-700 rounded text-sm text-dark-200 placeholder-dark-500 focus:outline-none focus:border-primary-500"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-dark-500 hover:text-dark-300"
            >
              ×
            </button>
          )}
        </div>
      </div>

      {/* شريط الأدوات */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-dark-700">
        <button className="p-1 text-dark-400 hover:text-dark-200 hover:bg-dark-800 rounded" title="تصفية">
          <Filter className="w-3.5 h-3.5" />
        </button>
        <button className="p-1 text-dark-400 hover:text-dark-200 hover:bg-dark-800 rounded" title="تنزيل">
          <Download className="w-3.5 h-3.5" />
        </button>
        <button className="p-1 text-dark-400 hover:text-dark-200 hover:bg-dark-800 rounded" title="رفع">
          <Upload className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* قائمة الملفات */}
      <div className="flex-1 overflow-y-auto py-2">
        {filteredFiles.map(node => (
          <TreeItem
            key={node.id}
            node={node}
            depth={0}
            selectedId={selectedId}
            onSelect={(n) => setSelectedId(n.id)}
            onToggle={toggleFolder}
            onContextMenu={(e, n) => {
              e.preventDefault();
              setContextMenu({ x: e.clientX, y: e.clientY, node: n });
            }}
            draggedNode={draggedNode}
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
            onDrop={handleDrop}
            searchQuery={searchQuery}
          />
        ))}
        {filteredFiles.length === 0 && (
          <div className="text-center py-8 text-dark-500 text-sm">
            لا توجد نتائج
          </div>
        )}
      </div>

      {/* القائمة السياقية */}
      {contextMenu && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          node={contextMenu.node}
          onClose={() => setContextMenu(null)}
          onNewFile={handleNewFile}
          onNewFolder={handleNewFolder}
          onRename={handleRename}
          onDelete={handleDelete}
          onCopy={handleCopy}
          onCut={handleCut}
          onPaste={handlePaste}
          canPaste={!!clipboard}
        />
      )}
    </div>
  );
}

export default FileExplorer;
