/**
 * مدير المشاريع - Project Manager
 * إدارة المشاريع، المشاريع الحديثة، القوالب، تكامل Git
 */

import { useState, useEffect } from "react";
import { 
  Folder, 
  FolderOpen,
  Plus,
  Clock,
  Star,
  GitBranch,
  MoreHorizontal,
  Search,
  Filter,
  Grid3X3,
  List,
  ExternalLink,
  Trash2,
  Edit3,
  GitFork,
  History,
  Check,
  X,
  Template,
  Code2,
  Layout,
  Box,
  Sparkles
} from "lucide-react";

// أنواع البيانات
interface Project {
  id: string;
  name: string;
  path: string;
  description?: string;
  lastOpened: Date;
  isStarred: boolean;
  gitBranch?: string;
  gitStatus?: "clean" | "modified" | "untracked";
  type: "react" | "node" | "python" | "rust" | "go" | "other";
  size: number;
  filesCount: number;
}

interface ProjectTemplate {
  id: string;
  name: string;
  description: string;
  icon: React.ElementType;
  color: string;
  tags: string[];
}

// القوالب المتاحة
const templates: ProjectTemplate[] = [
  { 
    id: "react-ts", 
    name: "React + TypeScript", 
    description: "تطبيق React حديث مع TypeScript و Tailwind",
    icon: Code2,
    color: "#61dafb",
    tags: ["React", "TypeScript", "Vite"]
  },
  { 
    id: "next-app", 
    name: "Next.js App", 
    description: "تطبيق Next.js مع App Router",
    icon: Layout,
    color: "#000000",
    tags: ["Next.js", "React", "SSR"]
  },
  { 
    id: "node-api", 
    name: "Node.js API", 
    description: "واجهة برمجة Node.js مع Express",
    icon: Box,
    color: "#339933",
    tags: ["Node.js", "Express", "API"]
  },
  { 
    id: "tauri-app", 
    name: "Tauri Desktop", 
    description: "تطبيق سطح مكتب باستخدام Tauri",
    icon: Box,
    color: "#24c8db",
    tags: ["Tauri", "Rust", "Desktop"]
  },
  { 
    id: "python-ml", 
    name: "Python ML", 
    description: "مشروع تعلم آلي مع Python",
    icon: Sparkles,
    color: "#3776ab",
    tags: ["Python", "ML", "PyTorch"]
  },
  { 
    id: "rust-cli", 
    name: "Rust CLI", 
    description: "أداة سطر أوامر باستخدام Rust",
    icon: Code2,
    color: "#dea584",
    tags: ["Rust", "CLI"]
  },
];

// مشاريع افتراضية
const defaultProjects: Project[] = [
  {
    id: "1",
    name: "bi-ide-v8",
    path: "/Users/bi/Documents/bi-ide-v8",
    description: "بيئة التطوير المتكاملة الذكية",
    lastOpened: new Date(Date.now() - 1000 * 60 * 30), // 30 دقيقة مضت
    isStarred: true,
    gitBranch: "main",
    gitStatus: "modified",
    type: "react",
    size: 2457600,
    filesCount: 1250,
  },
  {
    id: "2",
    name: "ai-training-platform",
    path: "/Users/bi/projects/ai-training",
    description: "منصة تدريب نماذج الذكاء الاصطناعي",
    lastOpened: new Date(Date.now() - 1000 * 60 * 60 * 2), // ساعتان مضتا
    isStarred: true,
    gitBranch: "develop",
    gitStatus: "clean",
    type: "python",
    size: 1024000,
    filesCount: 450,
  },
  {
    id: "3",
    name: "mobile-app",
    path: "/Users/bi/projects/mobile-app",
    description: "تطبيق جوال React Native",
    lastOpened: new Date(Date.now() - 1000 * 60 * 60 * 24), // يوم مضى
    isStarred: false,
    type: "react",
    size: 512000,
    filesCount: 320,
  },
  {
    id: "4",
    name: "backend-api",
    path: "/Users/bi/projects/backend",
    description: "واجهة برمجة الخلفية",
    lastOpened: new Date(Date.now() - 1000 * 60 * 60 * 24 * 3), // 3 أيام مضت
    isStarred: false,
    gitBranch: "feature/auth",
    gitStatus: "untracked",
    type: "node",
    size: 768000,
    filesCount: 180,
  },
  {
    id: "5",
    name: "data-pipeline",
    path: "/Users/bi/projects/data-pipeline",
    description: "خط أنابيب معالجة البيانات",
    lastOpened: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7), // أسبوع مضى
    isStarred: false,
    type: "python",
    size: 1536000,
    filesCount: 95,
  },
];

// مكون بطاقة المشروع
function ProjectCard({ 
  project, 
  viewMode,
  onOpen,
  onToggleStar,
  onDelete
}: { 
  project: Project;
  viewMode: "grid" | "list";
  onOpen: (project: Project) => void;
  onToggleStar: (project: Project) => void;
  onDelete: (project: Project) => void;
}) {
  const typeIcons: Record<string, string> = {
    react: "⚛️",
    node: "🟢",
    python: "🐍",
    rust: "🦀",
    go: "🔵",
    other: "📁",
  };

  const gitStatusColors = {
    clean: "text-green-400",
    modified: "text-yellow-400",
    untracked: "text-red-400",
  };

  const formatLastOpened = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (minutes < 60) return `منذ ${minutes} دقيقة`;
    if (hours < 24) return `منذ ${hours} ساعة`;
    if (days < 7) return `منذ ${days} أيام`;
    return date.toLocaleDateString('ar-SA');
  };

  if (viewMode === "list") {
    return (
      <div 
        onClick={() => onOpen(project)}
        className="group flex items-center gap-4 p-3 bg-dark-800 hover:bg-dark-700 rounded-lg cursor-pointer transition-colors"
      >
        <div className="text-2xl">{typeIcons[project.type]}</div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-dark-100 truncate">{project.name}</span>
            {project.gitBranch && (
              <span className="flex items-center gap-1 text-xs text-dark-400 bg-dark-900 px-2 py-0.5 rounded">
                <GitBranch className="w-3 h-3" />
                {project.gitBranch}
              </span>
            )}
          </div>
          <div className="text-sm text-dark-400 truncate">{project.path}</div>
        </div>

        <div className="flex items-center gap-4 text-sm text-dark-400">
          <span className="flex items-center gap-1">
            <Clock className="w-3.5 h-3.5" />
            {formatLastOpened(project.lastOpened)}
          </span>
          {project.gitStatus && (
            <span className={gitStatusColors[project.gitStatus]}>
              ●
            </span>
          )}
        </div>

        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={(e) => { e.stopPropagation(); onToggleStar(project); }}
            className={`p-1.5 rounded hover:bg-dark-600 transition-colors ${
              project.isStarred ? "text-yellow-400" : "text-dark-400"
            }`}
          >
            <Star className={`w-4 h-4 ${project.isStarred ? "fill-current" : ""}`} />
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onDelete(project); }}
            className="p-1.5 text-dark-400 hover:text-red-400 hover:bg-red-500/10 rounded transition-colors"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div 
      onClick={() => onOpen(project)}
      className="group bg-dark-800 hover:bg-dark-700 rounded-xl p-4 cursor-pointer transition-all hover:shadow-lg"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="text-3xl">{typeIcons[project.type]}</div>
        <div className="flex items-center gap-1">
          <button
            onClick={(e) => { e.stopPropagation(); onToggleStar(project); }}
            className={`p-1.5 rounded hover:bg-dark-600 transition-colors ${
              project.isStarred ? "text-yellow-400" : "text-dark-400"
            }`}
          >
            <Star className={`w-4 h-4 ${project.isStarred ? "fill-current" : ""}`} />
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onDelete(project); }}
            className="p-1.5 text-dark-400 hover:text-red-400 hover:bg-red-500/10 rounded transition-colors opacity-0 group-hover:opacity-100"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      <h3 className="font-semibold text-dark-100 mb-1 truncate">{project.name}</h3>
      {project.description && (
        <p className="text-sm text-dark-400 mb-3 line-clamp-2">{project.description}</p>
      )}

      <div className="flex items-center gap-2 mb-3">
        {project.gitBranch && (
          <span className="flex items-center gap-1 text-xs text-dark-400 bg-dark-900 px-2 py-1 rounded">
            <GitBranch className="w-3 h-3" />
            {project.gitBranch}
          </span>
        )}
        {project.gitStatus && (
          <span className={`text-xs ${gitStatusColors[project.gitStatus]}`}>
            ● {project.gitStatus === "clean" ? "نظيف" : project.gitStatus === "modified" ? "معدل" : "جديد"}
          </span>
        )}
      </div>

      <div className="flex items-center justify-between text-xs text-dark-500 pt-3 border-t border-dark-700">
        <span className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {formatLastOpened(project.lastOpened)}
        </span>
        <span>{project.filesCount.toLocaleString()} ملف</span>
      </div>
    </div>
  );
}

// مكون بطاقة القالب
function TemplateCard({ 
  template, 
  onSelect 
}: { 
  template: ProjectTemplate;
  onSelect: (template: ProjectTemplate) => void;
}) {
  return (
    <button
      onClick={() => onSelect(template)}
      className="group text-left bg-dark-800 hover:bg-dark-700 border border-dark-700 hover:border-primary-500/50 rounded-xl p-5 transition-all"
    >
      <div 
        className="w-12 h-12 rounded-xl flex items-center justify-center mb-4 transition-transform group-hover:scale-110"
        style={{ backgroundColor: `${template.color}20` }}
      >
        <template.icon className="w-6 h-6" style={{ color: template.color }} />
      </div>

      <h3 className="font-semibold text-dark-100 mb-1">{template.name}</h3>
      <p className="text-sm text-dark-400 mb-4">{template.description}</p>

      <div className="flex flex-wrap gap-1.5">
        {template.tags.map(tag => (
          <span 
            key={tag}
            className="text-xs px-2 py-0.5 bg-dark-900 text-dark-400 rounded"
          >
            {tag}
          </span>
        ))}
      </div>
    </button>
  );
}

// مكون إنشاء مشروع جديد
function CreateProjectModal({ 
  isOpen, 
  onClose,
  onCreate
}: { 
  isOpen: boolean;
  onClose: () => void;
  onCreate: (name: string, template: ProjectTemplate | null) => void;
}) {
  const [step, setStep] = useState<"template" | "details">("template");
  const [selectedTemplate, setSelectedTemplate] = useState<ProjectTemplate | null>(null);
  const [projectName, setProjectName] = useState("");

  if (!isOpen) return null;

  const handleCreate = () => {
    if (!projectName.trim()) return;
    onCreate(projectName, selectedTemplate);
    setStep("template");
    setSelectedTemplate(null);
    setProjectName("");
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-dark-800 rounded-2xl w-full max-w-2xl max-h-[80vh] overflow-hidden">
        {/* رأس النافذة */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-dark-700">
          <h2 className="text-lg font-semibold text-dark-100">
            {step === "template" ? "اختر قالب" : "إنشاء مشروع جديد"}
          </h2>
          <button 
            onClick={onClose}
            className="p-2 text-dark-400 hover:text-dark-200 hover:bg-dark-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* المحتوى */}
        <div className="p-6 overflow-y-auto max-h-[60vh]">
          {step === "template" ? (
            <div className="grid grid-cols-2 gap-4">
              <button
                onClick={() => { setSelectedTemplate(null); setStep("details"); }}
                className="group text-left bg-dark-800 hover:bg-dark-700 border border-dark-700 border-dashed hover:border-primary-500/50 rounded-xl p-5 transition-all flex flex-col items-center justify-center min-h-[200px]"
              >
                <Plus className="w-8 h-8 text-dark-400 mb-3 group-hover:text-primary-400 transition-colors" />
                <span className="text-dark-300 group-hover:text-dark-100">بدون قالب</span>
              </button>
              
              {templates.map(template => (
                <TemplateCard
                  key={template.id}
                  template={template}
                  onSelect={(t) => { setSelectedTemplate(t); setStep("details"); }}
                />
              ))}
            </div>
          ) : (
            <div className="space-y-4">
              {selectedTemplate && (
                <div className="flex items-center gap-3 p-3 bg-dark-900 rounded-lg">
                  <div 
                    className="w-10 h-10 rounded-lg flex items-center justify-center"
                    style={{ backgroundColor: `${selectedTemplate.color}20` }}
                  >
                    <selectedTemplate.icon className="w-5 h-5" style={{ color: selectedTemplate.color }} />
                  </div>
                  <div>
                    <div className="text-sm text-dark-400">القالب المختار</div>
                    <div className="font-medium text-dark-100">{selectedTemplate.name}</div>
                  </div>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-dark-200 mb-2">
                  اسم المشروع
                </label>
                <input
                  type="text"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  placeholder="أدخل اسم المشروع..."
                  className="w-full px-4 py-2.5 bg-dark-900 border border-dark-700 rounded-lg text-dark-100 placeholder-dark-500 focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-dark-200 mb-2">
                  المسار
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={`/Users/bi/projects/${projectName.toLowerCase().replace(/\s+/g, '-')}`}
                    readOnly
                    className="flex-1 px-4 py-2.5 bg-dark-900 border border-dark-700 rounded-lg text-dark-400"
                  />
                  <button className="px-4 py-2.5 bg-dark-700 hover:bg-dark-600 rounded-lg text-dark-200 transition-colors">
                    تصفح
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* أزرار التنقل */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-dark-700 bg-dark-800/50">
          {step === "template" ? (
            <button 
              onClick={onClose}
              className="px-4 py-2 text-dark-300 hover:text-dark-100 transition-colors"
            >
              إلغاء
            </button>
          ) : (
            <button 
              onClick={() => setStep("template")}
              className="px-4 py-2 text-dark-300 hover:text-dark-100 transition-colors"
            >
              رجوع
            </button>
          )}
          
          {step === "details" && (
            <button
              onClick={handleCreate}
              disabled={!projectName.trim()}
              className="px-6 py-2 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white font-medium transition-colors"
            >
              إنشاء المشروع
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// المكون الرئيسي
export function ProjectManager() {
  const [projects, setProjects] = useState<Project[]>(defaultProjects);
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [filter, setFilter] = useState<"all" | "starred" | "recent">("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  // تصفية المشاريع
  const filteredProjects = projects.filter(project => {
    // البحث
    if (searchQuery && !project.name.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false;
    }
    
    // الفلتر
    switch (filter) {
      case "starred":
        return project.isStarred;
      case "recent":
        return new Date().getTime() - project.lastOpened.getTime() < 1000 * 60 * 60 * 24 * 7;
      default:
        return true;
    }
  }).sort((a, b) => {
    // الترتيب: المفضلة أولاً، ثم حسب آخر فتح
    if (a.isStarred !== b.isStarred) return b.isStarred ? 1 : -1;
    return b.lastOpened.getTime() - a.lastOpened.getTime();
  });

  // فتح مشروع
  const handleOpenProject = (project: Project) => {
    console.log("Opening project:", project.path);
    // في الإنتاج: فتح المشروع في الـ IDE
  };

  // تبديل المفضلة
  const handleToggleStar = (project: Project) => {
    setProjects(prev => prev.map(p => 
      p.id === project.id ? { ...p, isStarred: !p.isStarred } : p
    ));
  };

  // حذف مشروع
  const handleDeleteProject = (project: Project) => {
    if (confirm(`هل أنت متأكد من حذف "${project.name}" من القائمة؟`)) {
      setProjects(prev => prev.filter(p => p.id !== project.id));
    }
  };

  // إنشاء مشروع جديد
  const handleCreateProject = (name: string, template: ProjectTemplate | null) => {
    const newProject: Project = {
      id: Date.now().toString(),
      name,
      path: `/Users/bi/projects/${name.toLowerCase().replace(/\s+/g, '-')}`,
      lastOpened: new Date(),
      isStarred: false,
      type: template?.id.includes("react") ? "react" : 
            template?.id.includes("node") ? "node" :
            template?.id.includes("python") ? "python" :
            template?.id.includes("rust") ? "rust" : "other",
      size: 0,
      filesCount: 0,
    };
    setProjects(prev => [newProject, ...prev]);
  };

  return (
    <div className="h-full flex flex-col bg-dark-900">
      {/* رأس الصفحة */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-dark-700">
        <div>
          <h1 className="text-xl font-bold text-dark-100">مدير المشاريع</h1>
          <p className="text-sm text-dark-400">إدارة مشاريعك وقوالبك</p>
        </div>
        
        <button
          onClick={() => setIsCreateModalOpen(true)}
          className="flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg text-white font-medium transition-colors"
        >
          <Plus className="w-4 h-4" />
          مشروع جديد
        </button>
      </div>

      {/* شريط التصفية */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-dark-700 gap-4">
        <div className="flex items-center gap-2">
          {[
            { id: "all", label: "الكل", icon: Folder },
            { id: "starred", label: "المفضلة", icon: Star },
            { id: "recent", label: "الحديثة", icon: History },
          ].map(f => (
            <button
              key={f.id}
              onClick={() => setFilter(f.id as typeof filter)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                filter === f.id
                  ? "bg-primary-600 text-white"
                  : "text-dark-400 hover:text-dark-200 hover:bg-dark-800"
              }`}
            >
              <f.icon className="w-4 h-4" />
              {f.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3">
          {/* البحث */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-500" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="البحث في المشاريع..."
              className="w-64 pl-9 pr-4 py-1.5 bg-dark-800 border border-dark-700 rounded-lg text-sm text-dark-200 placeholder-dark-500 focus:outline-none focus:border-primary-500"
            />
          </div>

          {/* تبديل العرض */}
          <div className="flex items-center bg-dark-800 rounded-lg p-1">
            <button
              onClick={() => setViewMode("grid")}
              className={`p-1.5 rounded transition-colors ${
                viewMode === "grid" ? "bg-dark-700 text-dark-100" : "text-dark-400 hover:text-dark-200"
              }`}
            >
              <Grid3X3 className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode("list")}
              className={`p-1.5 rounded transition-colors ${
                viewMode === "list" ? "bg-dark-700 text-dark-100" : "text-dark-400 hover:text-dark-200"
              }`}
            >
              <List className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* قائمة المشاريع */}
      <div className="flex-1 overflow-y-auto p-6">
        {filteredProjects.length > 0 ? (
          <div className={viewMode === "grid" 
            ? "grid grid-cols-3 xl:grid-cols-4 gap-4"
            : "space-y-2"
          }>
            {filteredProjects.map(project => (
              <ProjectCard
                key={project.id}
                project={project}
                viewMode={viewMode}
                onOpen={handleOpenProject}
                onToggleStar={handleToggleStar}
                onDelete={handleDeleteProject}
              />
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <FolderOpen className="w-16 h-16 text-dark-600 mb-4" />
            <h3 className="text-lg font-medium text-dark-300 mb-2">
              لا توجد مشاريع
            </h3>
            <p className="text-dark-500 mb-4">
              {searchQuery ? "لا توجد نتائج مطابقة للبحث" : "ابدأ بإنشاء مشروع جديد"}
            </p>
            {!searchQuery && (
              <button
                onClick={() => setIsCreateModalOpen(true)}
                className="flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg text-white transition-colors"
              >
                <Plus className="w-4 h-4" />
                إنشاء مشروع
              </button>
            )}
          </div>
        )}
      </div>

      {/* نافذة إنشاء المشروع */}
      <CreateProjectModal
        isOpen={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
        onCreate={handleCreateProject}
      />
    </div>
  );
}

export default ProjectManager;
