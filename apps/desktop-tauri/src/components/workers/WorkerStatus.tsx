/**
 * حالة العمال - Worker Status
 * عرض حالة العمال المتصلين مع معلومات النظام
 */

import { useState, useEffect } from "react";
import { 
  Monitor, 
  Server, 
  Laptop, 
  Globe,
  Cpu,
  HardDrive,
  Thermometer,
  Wifi,
  WifiOff,
  Activity,
  Zap
} from "lucide-react";

// أنواع البيانات
interface Worker {
  id: string;
  name: string;
  type: "rtx5090" | "windows" | "mac" | "hostinger";
  status: "online" | "offline" | "training" | "idle";
  cpu: {
    usage: number;
    cores: number;
    model: string;
  };
  ram: {
    used: number;
    total: number;
  };
  gpu?: {
    model: string;
    usage: number;
    vramUsed: number;
    vramTotal: number;
    temperature: number;
  };
  connection: {
    latency: number;
    lastSeen: number;
    bandwidth: number;
  };
  ip: string;
}

// بيانات العمال الافتراضية
const defaultWorkers: Worker[] = [
  {
    id: "worker-1",
    name: "RTX 5090",
    type: "rtx5090",
    status: "training",
    cpu: { usage: 45, cores: 32, model: "AMD Ryzen 9 7950X" },
    ram: { used: 64, total: 128 },
    gpu: { 
      model: "NVIDIA RTX 5090", 
      usage: 98, 
      vramUsed: 28, 
      vramTotal: 32,
      temperature: 72 
    },
    connection: { latency: 12, lastSeen: Date.now(), bandwidth: 1250 },
    ip: "192.168.1.101",
  },
  {
    id: "worker-2",
    name: "Windows Workstation",
    type: "windows",
    status: "idle",
    cpu: { usage: 23, cores: 16, model: "Intel i9-13900K" },
    ram: { used: 24, total: 64 },
    gpu: { 
      model: "NVIDIA RTX 4080", 
      usage: 5, 
      vramUsed: 4, 
      vramTotal: 16,
      temperature: 45 
    },
    connection: { latency: 8, lastSeen: Date.now(), bandwidth: 850 },
    ip: "192.168.1.102",
  },
  {
    id: "worker-3",
    name: "Mac Studio",
    type: "mac",
    status: "online",
    cpu: { usage: 67, cores: 20, model: "Apple M2 Ultra" },
    ram: { used: 48, total: 128 },
    connection: { latency: 3, lastSeen: Date.now(), bandwidth: 2100 },
    ip: "192.168.1.103",
  },
  {
    id: "worker-4",
    name: "Hostinger VPS",
    type: "hostinger",
    status: "offline",
    cpu: { usage: 0, cores: 8, model: "Intel Xeon E5" },
    ram: { used: 0, total: 32 },
    connection: { latency: 150, lastSeen: Date.now() - 3600000, bandwidth: 100 },
    ip: "185.185.185.10",
  },
];

// مكون شريط التقدم
function ProgressBar({ 
  value, 
  max = 100, 
  color = "bg-primary-500",
  label,
  size = "md"
}: { 
  value: number; 
  max?: number; 
  color?: string;
  label?: string;
  size?: "sm" | "md" | "lg";
}) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  const heightClass = size === "sm" ? "h-1.5" : size === "lg" ? "h-4" : "h-2.5";
  
  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between text-xs text-dark-400 mb-1">
          <span>{label}</span>
          <span>{percentage.toFixed(0)}%</span>
        </div>
      )}
      <div className={`w-full bg-dark-700 rounded-full overflow-hidden ${heightClass}`}>
        <div 
          className={`${color} transition-all duration-500 ${heightClass} rounded-full`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

// مؤشر الحالة
function StatusIndicator({ status }: { status: Worker["status"] }) {
  const config = {
    online: { color: "bg-green-500", text: "متصل", animate: false },
    offline: { color: "bg-red-500", text: "غير متصل", animate: false },
    training: { color: "bg-blue-500", text: "يتدرب", animate: true },
    idle: { color: "bg-yellow-500", text: "خامل", animate: false },
  };
  
  const { color, text, animate } = config[status];
  
  return (
    <div className="flex items-center gap-2">
      <span className={`w-3 h-3 rounded-full ${color} ${animate ? "animate-pulse" : ""} shadow-[0_0_8px_currentColor]`} />
      <span className="text-sm font-medium text-dark-200">{text}</span>
    </div>
  );
}

// أيقونة نوع العامل
function WorkerIcon({ type }: { type: Worker["type"] }) {
  const icons = {
    rtx5090: <Zap className="w-6 h-6 text-purple-400" />,
    windows: <Monitor className="w-6 h-6 text-blue-400" />,
    mac: <Laptop className="w-6 h-6 text-gray-400" />,
    hostinger: <Globe className="w-6 h-6 text-orange-400" />,
  };
  
  return (
    <div className="p-3 bg-dark-800 rounded-lg">
      {icons[type]}
    </div>
  );
}

// بطاقة العامل
function WorkerCard({ worker }: { worker: Worker }) {
  const ramPercentage = (worker.ram.used / worker.ram.total) * 100;
  
  // حساب اللون حسب درجة الحرارة
  const getTempColor = (temp: number) => {
    if (temp < 50) return "text-green-400";
    if (temp < 70) return "text-yellow-400";
    return "text-red-400";
  };

  return (
    <div className={`bg-dark-800 rounded-xl p-5 border transition-all duration-300 ${
      worker.status === "offline" 
        ? "border-dark-700 opacity-60" 
        : worker.status === "training"
        ? "border-blue-500/50 shadow-[0_0_20px_rgba(59,130,246,0.1)]"
        : "border-dark-700 hover:border-dark-600"
    }`}>
      {/* الرأس */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <WorkerIcon type={worker.type} />
          <div>
            <h3 className="font-semibold text-dark-100">{worker.name}</h3>
            <div className="flex items-center gap-2 mt-1">
              <StatusIndicator status={worker.status} />
              <span className="text-xs text-dark-500">•</span>
              <span className="text-xs text-dark-500 font-mono">{worker.ip}</span>
            </div>
          </div>
        </div>
        
        {/* مؤشر الاتصال */}
        <div className="flex items-center gap-1">
          {worker.status === "offline" ? (
            <WifiOff className="w-4 h-4 text-red-400" />
          ) : (
            <>
              <Wifi className="w-4 h-4 text-green-400" />
              <span className="text-xs text-dark-400 ml-1">{worker.connection.latency}ms</span>
            </>
          )}
        </div>
      </div>

      {/* معلومات CPU */}
      <div className="space-y-3 mb-4">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-2 text-dark-400">
            <Cpu className="w-4 h-4" />
            <span>CPU</span>
          </div>
          <span className="text-dark-300 font-mono">{worker.cpu.model}</span>
        </div>
        <ProgressBar 
          value={worker.cpu.usage} 
          color={worker.cpu.usage > 80 ? "bg-red-500" : worker.cpu.usage > 50 ? "bg-yellow-500" : "bg-green-500"}
          label={`استخدام CPU: ${worker.cpu.usage}% (${worker.cpu.cores} أنوية)`}
        />
      </div>

      {/* معلومات RAM */}
      <div className="space-y-3 mb-4">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-2 text-dark-400">
            <HardDrive className="w-4 h-4" />
            <span>RAM</span>
          </div>
          <span className="text-dark-300 font-mono">{worker.ram.used} / {worker.ram.total} GB</span>
        </div>
        <ProgressBar 
          value={ramPercentage}
          color={ramPercentage > 80 ? "bg-red-500" : ramPercentage > 60 ? "bg-yellow-500" : "bg-blue-500"}
        />
      </div>

      {/* معلومات GPU (إذا متوفرة) */}
      {worker.gpu && (
        <div className="space-y-3 mb-4 p-3 bg-dark-900/50 rounded-lg">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2 text-dark-400">
              <Activity className="w-4 h-4" />
              <span>GPU: {worker.gpu.model}</span>
            </div>
            <div className="flex items-center gap-3">
              <span className={`text-sm font-bold ${getTempColor(worker.gpu.temperature)}`}>
                <Thermometer className="w-3 h-3 inline mr-1" />
                {worker.gpu.temperature}°C
              </span>
            </div>
          </div>
          <ProgressBar 
            value={worker.gpu.usage}
            color={worker.gpu.usage > 90 ? "bg-red-500" : worker.gpu.usage > 70 ? "bg-yellow-500" : "bg-purple-500"}
            label={`استخدام GPU: ${worker.gpu.usage}%`}
          />
          <div className="flex justify-between text-xs text-dark-500">
            <span>VRAM: {worker.gpu.vramUsed} / {worker.gpu.vramTotal} GB</span>
            <span>{((worker.gpu.vramUsed / worker.gpu.vramTotal) * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}

      {/* معلومات الاتصال */}
      <div className="pt-3 border-t border-dark-700">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-4">
            <span className="text-dark-400">
              Bandwidth: <span className="text-dark-200">{worker.connection.bandwidth} Mbps</span>
            </span>
          </div>
          <span className="text-xs text-dark-500">
            آخر ظهور: {new Date(worker.connection.lastSeen).toLocaleTimeString('ar-SA')}
          </span>
        </div>
      </div>
    </div>
  );
}

// المكون الرئيسي
export function WorkerStatus() {
  const [workers, setWorkers] = useState<Worker[]>(defaultWorkers);
  const [lastUpdate, setLastUpdate] = useState(Date.now());

  // محاكاة تحديثات حية
  useEffect(() => {
    const interval = setInterval(() => {
      setWorkers(prev => prev.map(worker => {
        if (worker.status === "offline") return worker;
        
        return {
          ...worker,
          cpu: {
            ...worker.cpu,
            usage: Math.max(5, Math.min(100, worker.cpu.usage + (Math.random() - 0.5) * 10)),
          },
          ram: {
            ...worker.ram,
            used: Math.max(4, Math.min(worker.ram.total, worker.ram.used + (Math.random() - 0.5) * 4)),
          },
          gpu: worker.gpu ? {
            ...worker.gpu,
            usage: worker.status === "training" 
              ? Math.max(80, Math.min(100, worker.gpu.usage + (Math.random() - 0.5) * 5))
              : Math.max(0, Math.min(20, worker.gpu.usage + (Math.random() - 0.5) * 5)),
            temperature: worker.status === "training"
              ? Math.max(60, Math.min(85, worker.gpu.temperature + (Math.random() - 0.5) * 3))
              : Math.max(35, Math.min(55, worker.gpu.temperature + (Math.random() - 0.5) * 2)),
          } : undefined,
          connection: {
            ...worker.connection,
            latency: Math.max(1, worker.connection.latency + Math.floor((Math.random() - 0.5) * 5)),
            lastSeen: Date.now(),
          },
        };
      }));
      setLastUpdate(Date.now());
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // إحصائيات عامة
  const onlineWorkers = workers.filter(w => w.status !== "offline").length;
  const trainingWorkers = workers.filter(w => w.status === "training").length;
  const totalGPU = workers.reduce((acc, w) => acc + (w.gpu?.vramTotal || 0), 0);
  const avgLatency = workers
    .filter(w => w.status !== "offline")
    .reduce((acc, w) => acc + w.connection.latency, 0) / onlineWorkers || 0;

  return (
    <div className="h-full flex flex-col bg-dark-900 p-4 gap-4 overflow-auto">
      {/* رأس الصفحة */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Server className="w-6 h-6 text-primary-400" />
          <h1 className="text-xl font-bold text-dark-100">حالة العمال</h1>
          <span className="px-3 py-1 bg-dark-800 rounded-full text-sm text-dark-400">
            {onlineWorkers} / {workers.length} متصل
          </span>
        </div>
        <div className="text-xs text-dark-500">
          آخر تحديث: {new Date(lastUpdate).toLocaleTimeString('ar-SA')}
        </div>
      </div>

      {/* الإحصائيات العامة */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-dark-800 rounded-lg p-4">
          <div className="text-sm text-dark-400 mb-1">العمال المتصلون</div>
          <div className="text-2xl font-bold text-green-400">{onlineWorkers}</div>
        </div>
        <div className="bg-dark-800 rounded-lg p-4">
          <div className="text-sm text-dark-400 mb-1">العمال يتدربون</div>
          <div className="text-2xl font-bold text-blue-400">{trainingWorkers}</div>
        </div>
        <div className="bg-dark-800 rounded-lg p-4">
          <div className="text-sm text-dark-400 mb-1">إجمالي VRAM</div>
          <div className="text-2xl font-bold text-purple-400">{totalGPU} GB</div>
        </div>
        <div className="bg-dark-800 rounded-lg p-4">
          <div className="text-sm text-dark-400 mb-1">متوسط Latency</div>
          <div className="text-2xl font-bold text-yellow-400">{avgLatency.toFixed(0)} ms</div>
        </div>
      </div>

      {/* شبكة بطاقات العمال */}
      <div className="grid grid-cols-2 gap-4 flex-1">
        {workers.map(worker => (
          <WorkerCard key={worker.id} worker={worker} />
        ))}
      </div>
    </div>
  );
}

export default WorkerStatus;
