/**
 * لوحة تحكم التدريب - Training Dashboard
 * تعرض حالة التدريب، الرسوم البيانية للدقة والخسارة، واستخدام GPU
 */

import { useState, useEffect, useCallback } from "react";
import { 
  Play, 
  Pause, 
  Square, 
  RotateCcw,
  TrendingUp,
  TrendingDown,
  Activity,
  Cpu,
  Clock,
  Zap
} from "lucide-react";

// أنواع البيانات
interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
  timestamp: number;
}

interface TrainingStatus {
  isRunning: boolean;
  isPaused: boolean;
  currentEpoch: number;
  totalEpochs: number;
  progress: number;
  startTime: number | null;
  estimatedTimeRemaining: number;
}

interface GPUStats {
  utilization: number;
  vramUsed: number;
  vramTotal: number;
  temperature: number;
}

// مكون الرسم البياني المخصص
function LineChart({ 
  data, 
  width = 400, 
  height = 200,
  lines,
  yDomain
}: { 
  data: TrainingMetrics[];
  width?: number;
  height?: number;
  lines: { key: keyof TrainingMetrics; color: string; name: string }[];
  yDomain?: [number, number];
}) {
  if (data.length === 0) return <div className="text-dark-400 text-center py-8">لا توجد بيانات</div>;

  const padding = { top: 20, right: 30, bottom: 40, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const maxEpoch = Math.max(...data.map(d => d.epoch));
  const minEpoch = Math.min(...data.map(d => d.epoch));
  
  const allValues = data.flatMap(d => lines.map(l => d[l.key] as number));
  const minValue = yDomain ? yDomain[0] : Math.min(...allValues);
  const maxValue = yDomain ? yDomain[1] : Math.max(...allValues);
  const valueRange = maxValue - minValue || 1;

  const scaleX = (epoch: number) => 
    padding.left + ((epoch - minEpoch) / (maxEpoch - minEpoch || 1)) * chartWidth;
  
  const scaleY = (value: number) => 
    padding.top + chartHeight - ((value - minValue) / valueRange) * chartHeight;

  // إنشاء مسار الخط
  const createPath = (key: keyof TrainingMetrics) => {
    return data.map((d, i) => 
      `${i === 0 ? 'M' : 'L'} ${scaleX(d.epoch)} ${scaleY(d[key] as number)}`
    ).join(' ');
  };

  return (
    <svg width={width} height={height} className="w-full">
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
            {(maxValue - t * valueRange).toFixed(2)}
          </text>
        </g>
      ))}

      {/* خطوط المحور X */}
      {[0, 0.25, 0.5, 0.75, 1].map(t => {
        const epoch = Math.round(minEpoch + t * (maxEpoch - minEpoch));
        return (
          <g key={t}>
            <line
              x1={scaleX(epoch)}
              y1={padding.top}
              x2={scaleX(epoch)}
              y2={height - padding.bottom}
              stroke="#334155"
              strokeWidth={0.5}
              strokeDasharray="4"
            />
            <text
              x={scaleX(epoch)}
              y={height - padding.bottom + 20}
              fill="#64748b"
              fontSize={10}
              textAnchor="middle"
            >
              {epoch}
            </text>
          </g>
        );
      })}

      {/* خطوط البيانات */}
      {lines.map(line => (
        <path
          key={line.key as string}
          d={createPath(line.key)}
          fill="none"
          stroke={line.color}
          strokeWidth={2}
        />
      ))}

      {/* نقاط البيانات */}
      {lines.map(line => 
        data.map((d, i) => (
          <circle
            key={`${line.key}-${i}`}
            cx={scaleX(d.epoch)}
            cy={scaleY(d[line.key] as number)}
            r={3}
            fill={line.color}
          />
        ))
      )}

      {/* مفتاح الرسم البياني */}
      <g transform={`translate(${width - padding.right - 100}, 10)`}>
        {lines.map((line, i) => (
          <g key={line.key as string} transform={`translate(0, ${i * 20})`}>
            <line x1={0} y1={6} x2={20} y2={6} stroke={line.color} strokeWidth={2} />
            <text x={25} y={10} fill="#94a3b8" fontSize={11}>{line.name}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

// مكون مقياس GPU الدائري
function GPUGauge({ value, label, color }: { value: number; label: string; color: string }) {
  const radius = 45;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (value / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-28 h-28">
        <svg className="w-full h-full transform -rotate-90">
          {/* الخلفية */}
          <circle
            cx={56}
            cy={56}
            r={radius}
            fill="none"
            stroke="#1e293b"
            strokeWidth={8}
          />
          {/* المؤشر */}
          <circle
            cx={56}
            cy={56}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={8}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className="transition-all duration-500"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center flex-col">
          <span className="text-2xl font-bold text-dark-100">{value}%</span>
        </div>
      </div>
      <span className="text-sm text-dark-400 mt-2">{label}</span>
    </div>
  );
}

// مكون حدود الخطأ
class TrainingErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error) {
    console.error("Training Dashboard Error:", error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-6 bg-dark-800 rounded-lg text-center">
          <p className="text-red-400">حدث خطأ في لوحة التدريب</p>
          <button 
            onClick={() => this.setState({ hasError: false })}
            className="mt-4 px-4 py-2 bg-primary-600 rounded text-white"
          >
            إعادة المحاولة
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

import React from "react";

// المكون الرئيسي
export function TrainingDashboard() {
  // حالة التدريب
  const [status, setStatus] = useState<TrainingStatus>({
    isRunning: false,
    isPaused: false,
    currentEpoch: 0,
    totalEpochs: 100,
    progress: 0,
    startTime: null,
    estimatedTimeRemaining: 0,
  });

  // مقاييس التدريب
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  
  // إحصائيات GPU
  const [gpuStats, setGpuStats] = useState<GPUStats>({
    utilization: 0,
    vramUsed: 0,
    vramTotal: 24576, // 24GB
    temperature: 45,
  });

  // WebSocket connection
  const [wsConnected, setWsConnected] = useState(false);

  // محاكاة اتصال WebSocket
  useEffect(() => {
    // في الإنتاج، استبدل هذا باتصال WebSocket حقيقي
    const interval = setInterval(() => {
      setWsConnected(prev => !prev || prev); // محاكاة الاتصال
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // محاكاة تحديثات التدريب في الوقت الفعلي
  useEffect(() => {
    if (!status.isRunning || status.isPaused) return;

    const interval = setInterval(() => {
      setStatus(prev => {
        const newEpoch = prev.currentEpoch + 0.1;
        const newProgress = (newEpoch / prev.totalEpochs) * 100;
        
        return {
          ...prev,
          currentEpoch: newEpoch,
          progress: newProgress,
          estimatedTimeRemaining: Math.max(0, (prev.totalEpochs - newEpoch) * 30),
        };
      });

      // إضافة قياسات جديدة كل 10 epochs
      if (Math.floor(status.currentEpoch) > metrics.length) {
        const newMetric: TrainingMetrics = {
          epoch: Math.floor(status.currentEpoch),
          loss: 2.5 * Math.exp(-status.currentEpoch / 50) + 0.1 + Math.random() * 0.05,
          accuracy: Math.min(0.98, 0.6 + status.currentEpoch / 200 + Math.random() * 0.02),
          valLoss: 2.8 * Math.exp(-status.currentEpoch / 55) + 0.15 + Math.random() * 0.05,
          valAccuracy: Math.min(0.95, 0.55 + status.currentEpoch / 220 + Math.random() * 0.02),
          timestamp: Date.now(),
        };
        
        setMetrics(prev => [...prev, newMetric]);
      }
    }, 100);

    return () => clearInterval(interval);
  }, [status.isRunning, status.isPaused, status.currentEpoch, metrics.length]);

  // محاكاة إحصائيات GPU
  useEffect(() => {
    const interval = setInterval(() => {
      setGpuStats(prev => ({
        ...prev,
        utilization: status.isRunning 
          ? Math.min(100, 70 + Math.random() * 25) 
          : Math.max(5, prev.utilization - 5),
        vramUsed: status.isRunning 
          ? Math.min(prev.vramTotal, 18432 + Math.random() * 2048)
          : Math.max(1024, prev.vramUsed - 512),
        temperature: status.isRunning
          ? Math.min(85, 60 + Math.random() * 15)
          : Math.max(35, prev.temperature - 2),
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, [status.isRunning]);

  // بدء التدريب
  const handleStart = useCallback(() => {
    setStatus(prev => ({
      ...prev,
      isRunning: true,
      isPaused: false,
      startTime: prev.startTime || Date.now(),
    }));
  }, []);

  // إيقاف التدريب مؤقتاً
  const handlePause = useCallback(() => {
    setStatus(prev => ({
      ...prev,
      isPaused: !prev.isPaused,
    }));
  }, []);

  // إيقاف التدريب
  const handleStop = useCallback(() => {
    setStatus({
      isRunning: false,
      isPaused: false,
      currentEpoch: 0,
      totalEpochs: 100,
      progress: 0,
      startTime: null,
      estimatedTimeRemaining: 0,
    });
    setMetrics([]);
  }, []);

  // تنسيق الوقت
  const formatTime = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // حساب اللون حسب الحرارة
  const getTempColor = (temp: number) => {
    if (temp < 60) return "#22c55e";
    if (temp < 75) return "#eab308";
    return "#ef4444";
  };

  return (
    <TrainingErrorBoundary>
      <div className="h-full flex flex-col bg-dark-900 p-4 gap-4 overflow-auto">
        {/* رأس الصفحة */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-primary-400" />
            <h1 className="text-xl font-bold text-dark-100">لوحة تحكم التدريب</h1>
            <span className={`px-2 py-1 rounded-full text-xs ${
              wsConnected ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
            }`}>
              {wsConnected ? "متصل" : "غير متصل"}
            </span>
          </div>
          
          {/* أزرار التحكم */}
          <div className="flex items-center gap-2">
            <button
              onClick={handleStart}
              disabled={status.isRunning && !status.isPaused}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white font-medium transition-colors"
            >
              <Play className="w-4 h-4" />
              بدء
            </button>
            <button
              onClick={handlePause}
              disabled={!status.isRunning}
              className="flex items-center gap-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white font-medium transition-colors"
            >
              <Pause className="w-4 h-4" />
              {status.isPaused ? "استئناف" : "إيقاف مؤقت"}
            </button>
            <button
              onClick={handleStop}
              disabled={!status.isRunning}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white font-medium transition-colors"
            >
              <Square className="w-4 h-4" />
              إيقاف
            </button>
            <button
              onClick={() => setMetrics([])}
              className="flex items-center gap-2 px-4 py-2 bg-dark-700 hover:bg-dark-600 rounded-lg text-dark-200 font-medium transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              مسح
            </button>
          </div>
        </div>

        {/* حالة التدريب */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-dark-800 rounded-lg p-4">
            <div className="flex items-center gap-2 text-dark-400 mb-2">
              <Clock className="w-4 h-4" />
              <span className="text-sm">الـ Epoch الحالي</span>
            </div>
            <div className="text-2xl font-bold text-dark-100">
              {Math.floor(status.currentEpoch)} / {status.totalEpochs}
            </div>
          </div>
          
          <div className="bg-dark-800 rounded-lg p-4">
            <div className="flex items-center gap-2 text-dark-400 mb-2">
              <TrendingUp className="w-4 h-4" />
              <span className="text-sm">نسبة التقدم</span>
            </div>
            <div className="text-2xl font-bold text-primary-400">
              {status.progress.toFixed(1)}%
            </div>
          </div>
          
          <div className="bg-dark-800 rounded-lg p-4">
            <div className="flex items-center gap-2 text-dark-400 mb-2">
              <Zap className="w-4 h-4" />
              <span className="text-sm">الوقت المتبقي</span>
            </div>
            <div className="text-2xl font-bold text-dark-100">
              {formatTime(status.estimatedTimeRemaining)}
            </div>
          </div>
          
          <div className="bg-dark-800 rounded-lg p-4">
            <div className="flex items-center gap-2 text-dark-400 mb-2">
              <Cpu className="w-4 h-4" />
              <span className="text-sm">استخدام GPU</span>
            </div>
            <div className="text-2xl font-bold" style={{ color: getTempColor(gpuStats.temperature) }}>
              {gpuStats.utilization.toFixed(0)}%
            </div>
          </div>
        </div>

        {/* شريط التقدم */}
        <div className="bg-dark-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-dark-400">تقدم التدريب</span>
            <span className="text-sm font-medium text-primary-400">
              Epoch {Math.floor(status.currentEpoch)} / {status.totalEpochs}
            </span>
          </div>
          <div className="h-3 bg-dark-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-primary-600 to-primary-400 transition-all duration-300"
              style={{ width: `${status.progress}%` }}
            />
          </div>
        </div>

        {/* الرسوم البيانية */}
        <div className="grid grid-cols-2 gap-4 flex-1 min-h-0">
          {/* رسم الخسارة */}
          <div className="bg-dark-800 rounded-lg p-4 flex flex-col">
            <div className="flex items-center gap-2 mb-4">
              <TrendingDown className="w-5 h-5 text-red-400" />
              <h3 className="font-semibold text-dark-200">الخسارة (Loss)</h3>
            </div>
            <div className="flex-1 min-h-0">
              <LineChart
                data={metrics}
                lines={[
                  { key: "loss", color: "#ef4444", name: "Training Loss" },
                  { key: "valLoss", color: "#f97316", name: "Validation Loss" },
                ]}
                yDomain={[0, 3]}
              />
            </div>
          </div>

          {/* رسم الدقة */}
          <div className="bg-dark-800 rounded-lg p-4 flex flex-col">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-5 h-5 text-green-400" />
              <h3 className="font-semibold text-dark-200">الدقة (Accuracy)</h3>
            </div>
            <div className="flex-1 min-h-0">
              <LineChart
                data={metrics}
                lines={[
                  { key: "accuracy", color: "#22c55e", name: "Training Acc" },
                  { key: "valAccuracy", color: "#3b82f6", name: "Validation Acc" },
                ]}
                yDomain={[0.5, 1]}
              />
            </div>
          </div>
        </div>

        {/* GPU Stats */}
        <div className="bg-dark-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="w-5 h-5 text-primary-400" />
            <h3 className="font-semibold text-dark-200">استخدام GPU</h3>
          </div>
          <div className="grid grid-cols-4 gap-4">
            <GPUGauge 
              value={gpuStats.utilization} 
              label="الاستخدام" 
              color="#0ea5e9"
            />
            <GPUGauge 
              value={(gpuStats.vramUsed / gpuStats.vramTotal) * 100} 
              label="VRAM" 
              color="#8b5cf6"
            />
            <div className="flex flex-col items-center justify-center">
              <span className="text-3xl font-bold" style={{ color: getTempColor(gpuStats.temperature) }}>
                {gpuStats.temperature.toFixed(0)}°C
              </span>
              <span className="text-sm text-dark-400 mt-2">الحرارة</span>
            </div>
            <div className="flex flex-col items-center justify-center">
              <span className="text-3xl font-bold text-dark-100">
                {(gpuStats.vramUsed / 1024).toFixed(1)} GB
              </span>
              <span className="text-sm text-dark-400 mt-2">VRAM مستخدم</span>
            </div>
          </div>
        </div>
      </div>
    </TrainingErrorBoundary>
  );
}

export default TrainingDashboard;
