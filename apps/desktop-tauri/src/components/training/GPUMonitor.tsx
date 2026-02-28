/**
 * مراقب GPU - GPU Monitor
 * مراقبة GPU في الوقت الفعلي مع رسوم بيانية ومؤشرات حرارة
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { 
  Cpu, 
  Thermometer, 
  Fan,
  Activity,
  Clock,
  AlertTriangle,
  Zap,
  HardDrive,
  Download,
  Upload
} from "lucide-react";

// أنواع البيانات
interface GPUMetric {
  timestamp: number;
  utilization: number;
  vramUsed: number;
  temperature: number;
  fanSpeed: number;
  powerDraw: number;
  clockSpeed: number;
}

interface GPUInfo {
  id: number;
  name: string;
  vramTotal: number;
  architecture: string;
  driverVersion: string;
  pciBus: string;
}

// مكون الرسم البياني المباشر
function RealTimeChart({ 
  data, 
  width = 600, 
  height = 150,
  color = "#0ea5e9",
  maxValue = 100,
  label
}: { 
  data: { value: number; timestamp: number }[];
  width?: number;
  height?: number;
  color?: string;
  maxValue?: number;
  label?: string;
}) {
  const padding = { top: 20, right: 20, bottom: 30, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // إنشاء نقاط المسار
  const points = data.map((d, i) => {
    const x = padding.left + (i / (data.length - 1 || 1)) * chartWidth;
    const y = padding.top + chartHeight - (d.value / maxValue) * chartHeight;
    return `${x},${y}`;
  }).join(' ');

  const currentValue = data[data.length - 1]?.value || 0;

  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-dark-300">{label}</span>
          <span className="text-lg font-bold" style={{ color }}>{currentValue.toFixed(1)}%</span>
        </div>
      )}
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        {/* شبكة الخلفية */}
        {[0, 0.25, 0.5, 0.75, 1].map(t => (
          <g key={t}>
            <line
              x1={padding.left}
              y1={padding.top + t * chartHeight}
              x2={width - padding.right}
              y2={padding.top + t * chartHeight}
              stroke="#334155"
              strokeWidth={0.5}
              strokeDasharray="4"
            />
            <text
              x={padding.left - 10}
              y={padding.top + t * chartHeight + 4}
              fill="#64748b"
              fontSize={10}
              textAnchor="end"
            >
              {Math.round(maxValue - t * maxValue)}
            </text>
          </g>
        ))}

        {/* المساحة الممتلئة */}
        {data.length > 1 && (
          <polygon
            points={`${padding.left},${padding.top + chartHeight} ${points} ${width - padding.right},${padding.top + chartHeight}`}
            fill={color}
            fillOpacity={0.1}
          />
        )}

        {/* الخط */}
        {data.length > 1 && (
          <polyline
            points={points}
            fill="none"
            stroke={color}
            strokeWidth={2}
          />
        )}

        {/* النقاط */}
        {data.map((d, i) => {
          const x = padding.left + (i / (data.length - 1 || 1)) * chartWidth;
          const y = padding.top + chartHeight - (d.value / maxValue) * chartHeight;
          return (
            <circle
              key={i}
              cx={x}
              cy={y}
              r={i === data.length - 1 ? 4 : 2}
              fill={color}
              opacity={i === data.length - 1 ? 1 : 0.5}
            />
          );
        })}
      </svg>
    </div>
  );
}

// مقياس الحرارة الدائري
function TemperatureGauge({ temperature }: { temperature: number }) {
  const getColor = (temp: number) => {
    if (temp < 60) return "#22c55e";
    if (temp < 75) return "#eab308";
    if (temp < 85) return "#f97316";
    return "#ef4444";
  };

  const getStatus = (temp: number) => {
    if (temp < 60) return { text: "ممتاز", icon: Activity };
    if (temp < 75) return { text: "جيد", icon: Activity };
    if (temp < 85) return { text: "دافئ", icon: AlertTriangle };
    return { text: "ساخن!", icon: AlertTriangle };
  };

  const color = getColor(temperature);
  const status = getStatus(temperature);
  const StatusIcon = status.icon;

  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (temperature / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-36 h-36">
        <svg className="w-full h-full transform -rotate-90">
          {/* الخلفية */}
          <circle
            cx={72}
            cy={72}
            r={radius}
            fill="none"
            stroke="#1e293b"
            strokeWidth={12}
          />
          {/* المؤشر */}
          <circle
            cx={72}
            cy={72}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={12}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className="transition-all duration-500"
            style={{
              filter: `drop-shadow(0 0 10px ${color})`
            }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <Thermometer className="w-6 h-6 mb-1" style={{ color }} />
          <span className="text-3xl font-bold" style={{ color }}>{temperature.toFixed(0)}°C</span>
          <span className="text-xs text-dark-400 mt-1">{status.text}</span>
        </div>
      </div>
    </div>
  );
}

// شريط VRAM
function VRAMBar({ used, total }: { used: number; total: number }) {
  const percentage = (used / total) * 100;
  
  const getColor = (pct: number) => {
    if (pct < 50) return "bg-green-500";
    if (pct < 80) return "bg-yellow-500";
    return "bg-red-500";
  };

  return (
    <div className="bg-dark-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <HardDrive className="w-5 h-5 text-purple-400" />
          <span className="font-medium text-dark-200">VRAM</span>
        </div>
        <span className="text-sm text-dark-400">
          {used.toFixed(1)} / {total} GB
        </span>
      </div>
      
      <div className="relative h-8 bg-dark-900 rounded-lg overflow-hidden">
        <div 
          className={`h-full ${getColor(percentage)} transition-all duration-500 flex items-center justify-end pr-2`}
          style={{ width: `${percentage}%` }}
        >
          <span className="text-xs font-bold text-white">{percentage.toFixed(1)}%</span>
        </div>
      </div>
      
      <div className="flex justify-between text-xs text-dark-500 mt-2">
        <span>0 GB</span>
        <span>{(total / 2).toFixed(0)} GB</span>
        <span>{total} GB</span>
      </div>
    </div>
  );
}

// عرض سرعة المروحة
function FanDisplay({ speed }: { speed: number }) {
  return (
    <div className="bg-dark-800 rounded-lg p-4 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-dark-900 rounded-lg">
          <Fan className="w-6 h-6 text-cyan-400 animate-spin" style={{ animationDuration: `${Math.max(0.2, 2 - speed / 100)}s` }} />
        </div>
        <div>
          <div className="text-sm text-dark-400">سرعة المروحة</div>
          <div className="text-lg font-bold text-dark-100">{speed.toFixed(0)}%</div>
        </div>
      </div>
      <div className="text-right">
        <div className="text-xs text-dark-500">RPM</div>
        <div className="text-sm font-mono text-cyan-400">{Math.round(speed * 35)}</div>
      </div>
    </div>
  );
}

// المكون الرئيسي
export function GPUMonitor() {
  // معلومات GPU
  const [gpuInfo] = useState<GPUInfo>({
    id: 0,
    name: "NVIDIA GeForce RTX 5090",
    vramTotal: 32,
    architecture: "Blackwell",
    driverVersion: "551.23",
    pciBus: "0000:01:00.0",
  });

  // المقاييس الحية
  const [metrics, setMetrics] = useState<GPUMetric[]>([]);
  const maxDataPoints = 60;
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // إضافة قياس جديد
  const addMetric = useCallback(() => {
    setMetrics(prev => {
      const newMetric: GPUMetric = {
        timestamp: Date.now(),
        utilization: 30 + Math.random() * 40 + Math.sin(Date.now() / 1000) * 20,
        vramUsed: 12 + Math.random() * 8,
        temperature: 55 + Math.random() * 15,
        fanSpeed: 40 + Math.random() * 30,
        powerDraw: 250 + Math.random() * 100,
        clockSpeed: 2000 + Math.random() * 500,
      };
      
      const newMetrics = [...prev, newMetric];
      if (newMetrics.length > maxDataPoints) {
        return newMetrics.slice(newMetrics.length - maxDataPoints);
      }
      return newMetrics;
    });
  }, []);

  // بدء المراقبة
  useEffect(() => {
    intervalRef.current = setInterval(addMetric, 1000);
    addMetric(); // قياس أولي
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [addMetric]);

  // البيانات الحالية
  const currentMetric = metrics[metrics.length - 1] || {
    utilization: 0,
    vramUsed: 0,
    temperature: 0,
    fanSpeed: 0,
    powerDraw: 0,
    clockSpeed: 0,
  };

  // بيانات الرسوم البيانية
  const utilizationData = metrics.map(m => ({ value: m.utilization, timestamp: m.timestamp }));
  const vramData = metrics.map(m => ({ value: (m.vramUsed / gpuInfo.vramTotal) * 100, timestamp: m.timestamp }));
  const powerData = metrics.map(m => ({ value: (m.powerDraw / 450) * 100, timestamp: m.timestamp }));

  return (
    <div className="h-full flex flex-col bg-dark-900 p-4 gap-4 overflow-auto">
      {/* رأس الصفحة */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Cpu className="w-6 h-6 text-primary-400" />
          <div>
            <h1 className="text-xl font-bold text-dark-100">مراقب GPU</h1>
            <p className="text-sm text-dark-400">{gpuInfo.name}</p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-sm text-dark-400">
          <span>Driver: {gpuInfo.driverVersion}</span>
          <span>PCI: {gpuInfo.pciBus}</span>
        </div>
      </div>

      {/* المعلومات الرئيسية */}
      <div className="grid grid-cols-3 gap-4">
        {/* مقياس الحرارة */}
        <div className="bg-dark-800 rounded-xl p-6 flex flex-col items-center justify-center">
          <TemperatureGauge temperature={currentMetric.temperature} />
          <div className="mt-4 flex items-center gap-4">
            <div className="text-center">
              <div className="text-xs text-dark-500">الحد الأدنى</div>
              <div className="text-lg font-bold text-green-400">
                {Math.min(...metrics.map(m => m.temperature) || [0]).toFixed(0)}°C
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-dark-500">الحد الأقصى</div>
              <div className="text-lg font-bold text-red-400">
                {Math.max(...metrics.map(m => m.temperature) || [0]).toFixed(0)}°C
              </div>
            </div>
          </div>
        </div>

        {/* VRAM */}
        <div className="flex flex-col gap-4">
          <VRAMBar used={currentMetric.vramUsed} total={gpuInfo.vramTotal} />
          <FanDisplay speed={currentMetric.fanSpeed} />
        </div>

        {/* معلومات إضافية */}
        <div className="bg-dark-800 rounded-xl p-4 space-y-4">
          <h3 className="font-semibold text-dark-200 mb-3">معلومات GPU</h3>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-dark-400">
              <Zap className="w-4 h-4" />
              <span className="text-sm">استهلاك الطاقة</span>
            </div>
            <span className="text-sm font-mono text-yellow-400">
              {currentMetric.powerDraw.toFixed(0)}W / 450W
            </span>
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-dark-400">
              <Clock className="w-4 h-4" />
              <span className="text-sm">سرعة الساعة</span>
            </div>
            <span className="text-sm font-mono text-primary-400">
              {currentMetric.clockSpeed.toFixed(0)} MHz
            </span>
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-dark-400">
              <Download className="w-4 h-4" />
              <span className="text-sm">القراءة</span>
            </div>
            <span className="text-sm font-mono text-green-400">
              {(Math.random() * 50 + 20).toFixed(1)} GB/s
            </span>
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-dark-400">
              <Upload className="w-4 h-4" />
              <span className="text-sm">الكتابة</span>
            </div>
            <span className="text-sm font-mono text-blue-400">
              {(Math.random() * 30 + 10).toFixed(1)} GB/s
            </span>
          </div>
          
          <div className="pt-3 border-t border-dark-700">
            <div className="text-xs text-dark-500 mb-1">الهيكل المعماري</div>
            <div className="text-sm font-medium text-dark-300">{gpuInfo.architecture}</div>
          </div>
        </div>
      </div>

      {/* الرسوم البيانية */}
      <div className="grid grid-cols-3 gap-4 flex-1 min-h-0">
        <div className="bg-dark-800 rounded-lg p-4 flex flex-col">
          <RealTimeChart
            data={utilizationData}
            color="#0ea5e9"
            label="استخدام GPU"
          />
        </div>
        
        <div className="bg-dark-800 rounded-lg p-4 flex flex-col">
          <RealTimeChart
            data={vramData}
            color="#8b5cf6"
            maxValue={100}
            label="استخدام VRAM %"
          />
        </div>
        
        <div className="bg-dark-800 rounded-lg p-4 flex flex-col">
          <RealTimeChart
            data={powerData}
            color="#eab308"
            maxValue={100}
            label="استهلاك الطاقة %"
          />
        </div>
      </div>
    </div>
  );
}

export default GPUMonitor;
