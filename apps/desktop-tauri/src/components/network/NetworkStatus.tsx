/**
 * حالة الشبكة - Network Status
 * عرض حالة الاتصال لجميع الأجهزة مع مؤشرات Latency واستخدام النطاق الترددي
 */

import { useState, useEffect, useCallback } from "react";
import { 
  Wifi, 
  WifiOff,
  Server,
  Monitor,
  Laptop,
  Smartphone,
  Globe,
  RefreshCw,
  Zap,
  Activity,
  ArrowUp,
  ArrowDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  Settings,
  MoreHorizontal,
  Shield,
  Router,
  Database
} from "lucide-react";

// أنواع البيانات
interface NetworkDevice {
  id: string;
  name: string;
  type: "server" | "desktop" | "laptop" | "mobile" | "cloud";
  ip: string;
  status: "online" | "offline" | "degraded" | "unknown";
  latency: number;
  packetLoss: number;
  bandwidth: {
    up: number;
    down: number;
  };
  lastSeen: Date;
  region: string;
  connectionType: "ethernet" | "wifi" | "cellular";
}

interface NetworkStats {
  totalDevices: number;
  onlineDevices: number;
  avgLatency: number;
  totalBandwidth: {
    up: number;
    down: number;
  };
}

// بيانات الأجهزة الافتراضية
const defaultDevices: NetworkDevice[] = [
  {
    id: "local",
    name: "هذا الجهاز",
    type: "desktop",
    ip: "127.0.0.1",
    status: "online",
    latency: 0,
    packetLoss: 0,
    bandwidth: { up: 125, down: 250 },
    lastSeen: new Date(),
    region: "Local",
    connectionType: "ethernet",
  },
  {
    id: "rtx5090",
    name: "RTX 5090 Workstation",
    type: "desktop",
    ip: "192.168.1.101",
    status: "online",
    latency: 12,
    packetLoss: 0.1,
    bandwidth: { up: 1000, down: 1000 },
    lastSeen: new Date(),
    region: "LAN",
    connectionType: "ethernet",
  },
  {
    id: "windows-pc",
    name: "Windows Workstation",
    type: "desktop",
    ip: "192.168.1.102",
    status: "online",
    latency: 8,
    packetLoss: 0,
    bandwidth: { up: 500, down: 500 },
    lastSeen: new Date(),
    region: "LAN",
    connectionType: "wifi",
  },
  {
    id: "mac-studio",
    name: "Mac Studio",
    type: "desktop",
    ip: "192.168.1.103",
    status: "online",
    latency: 3,
    packetLoss: 0,
    bandwidth: { up: 1000, down: 1000 },
    lastSeen: new Date(),
    region: "LAN",
    connectionType: "ethernet",
  },
  {
    id: "hostinger",
    name: "Hostinger VPS",
    type: "cloud",
    ip: "185.185.185.10",
    status: "degraded",
    latency: 150,
    packetLoss: 2.5,
    bandwidth: { up: 100, down: 100 },
    lastSeen: new Date(Date.now() - 30000),
    region: "Europe",
    connectionType: "cellular",
  },
  {
    id: "aws-server",
    name: "AWS EC2 Instance",
    type: "server",
    ip: "54.123.45.67",
    status: "online",
    latency: 45,
    packetLoss: 0.2,
    bandwidth: { up: 500, down: 500 },
    lastSeen: new Date(),
    region: "US East",
    connectionType: "ethernet",
  },
  {
    id: "mobile-device",
    name: "iPhone 15 Pro",
    type: "mobile",
    ip: "192.168.1.150",
    status: "offline",
    latency: 0,
    packetLoss: 100,
    bandwidth: { up: 0, down: 0 },
    lastSeen: new Date(Date.now() - 3600000),
    region: "LAN",
    connectionType: "wifi",
  },
];

// مكون أيقونة الجهاز
function DeviceIcon({ type, status }: { type: NetworkDevice["type"]; status: NetworkDevice["status"] }) {
  const icons = {
    server: Server,
    desktop: Monitor,
    laptop: Laptop,
    mobile: Smartphone,
    cloud: Globe,
  };

  const Icon = icons[type];
  const colorClass = status === "online" ? "text-green-400" :
                     status === "degraded" ? "text-yellow-400" :
                     status === "offline" ? "text-red-400" : "text-dark-400";

  return (
    <div className={`p-2 rounded-lg bg-dark-900 ${colorClass}`}>
      <Icon className="w-5 h-5" />
    </div>
  );
}

// مؤشر الحالة
function StatusBadge({ status }: { status: NetworkDevice["status"] }) {
  const config = {
    online: { color: "bg-green-500", text: "متصل", bg: "bg-green-500/10", textColor: "text-green-400" },
    offline: { color: "bg-red-500", text: "غير متصل", bg: "bg-red-500/10", textColor: "text-red-400" },
    degraded: { color: "bg-yellow-500", text: "ضعيف", bg: "bg-yellow-500/10", textColor: "text-yellow-400" },
    unknown: { color: "bg-dark-500", text: "غير معروف", bg: "bg-dark-500/10", textColor: "text-dark-400" },
  };

  const { color, text, bg, textColor } = config[status];

  return (
    <div className={`flex items-center gap-2 px-2.5 py-1 rounded-full ${bg}`}>
      <span className={`w-2 h-2 rounded-full ${color} ${status === "online" ? "animate-pulse" : ""}`} />
      <span className={`text-xs font-medium ${textColor}`}>{text}</span>
    </div>
  );
}

// مكون Latency
function LatencyIndicator({ latency, packetLoss }: { latency: number; packetLoss: number }) {
  const getColor = () => {
    if (latency < 20) return "text-green-400";
    if (latency < 100) return "text-yellow-400";
    return "text-red-400";
  };

  const getBars = () => {
    if (latency < 20) return 4;
    if (latency < 50) return 3;
    if (latency < 100) return 2;
    return 1;
  };

  const bars = getBars();

  return (
    <div className="flex items-center gap-2">
      {/* أشرطة الإشارة */}
      <div className="flex items-end gap-0.5 h-4">
        {[1, 2, 3, 4].map((bar) => (
          <div
            key={bar}
            className={`w-1 rounded-sm ${
              bar <= bars ? getColor() : "bg-dark-700"
            }`}
            style={{ height: `${bar * 25}%` }}
          />
        ))}
      </div>
      
      <div className={`text-sm font-mono ${getColor()}`}>
        {latency > 0 ? `${latency}ms` : "--"}
      </div>
      
      {packetLoss > 0 && (
        <span className="text-xs text-red-400">
          ({packetLoss.toFixed(1)}% فقد)
        </span>
      )}
    </div>
  );
}

// مكون عرض النطاق الترددي
function BandwidthDisplay({ up, down }: { up: number; down: number }) {
  const formatSpeed = (speed: number) => {
    if (speed >= 1000) return `${(speed / 1000).toFixed(1)} Gbps`;
    return `${speed.toFixed(0)} Mbps`;
  };

  return (
    <div className="flex items-center gap-4 text-xs">
      <div className="flex items-center gap-1.5">
        <ArrowUp className="w-3.5 h-3.5 text-green-400" />
        <span className="text-dark-300 font-mono">{formatSpeed(up)}</span>
      </div>
      <div className="flex items-center gap-1.5">
        <ArrowDown className="w-3.5 h-3.5 text-blue-400" />
        <span className="text-dark-300 font-mono">{formatSpeed(down)}</span>
      </div>
    </div>
  );
}

// بطاقة الجهاز
function DeviceCard({ 
  device,
  onRefresh
}: { 
  device: NetworkDevice;
  onRefresh: (id: string) => void;
}) {
  return (
    <div className={`p-4 rounded-xl border transition-all ${
      device.status === "online" 
        ? "bg-dark-800 border-dark-700" 
        : device.status === "degraded"
        ? "bg-yellow-500/5 border-yellow-500/20"
        : "bg-red-500/5 border-red-500/20 opacity-70"
    }`}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <DeviceIcon type={device.type} status={device.status} />
          <div>
            <h3 className="font-medium text-dark-100">{device.name}</h3>
            <div className="flex items-center gap-2 text-xs text-dark-400">
              <span className="font-mono">{device.ip}</span>
              <span>•</span>
              <span>{device.region}</span>
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <StatusBadge status={device.status} />
          <button
            onClick={() => onRefresh(device.id)}
            className="p-1.5 text-dark-400 hover:text-dark-200 hover:bg-dark-700 rounded transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-xs text-dark-500 mb-1">Latency</div>
          <LatencyIndicator latency={device.latency} packetLoss={device.packetLoss} />
        </div>
        
        <div>
          <div className="text-xs text-dark-500 mb-1">النطاق الترددي</div>
          <BandwidthDisplay up={device.bandwidth.up} down={device.bandwidth.down} />
        </div>
      </div>

      <div className="flex items-center justify-between mt-3 pt-3 border-t border-dark-700">
        <div className="flex items-center gap-2 text-xs text-dark-400">
          {device.connectionType === "ethernet" ? (
            <>
              <Database className="w-3.5 h-3.5" />
              <span>سلكي</span>
            </>
          ) : device.connectionType === "wifi" ? (
            <>
              <Wifi className="w-3.5 h-3.5" />
              <span>WiFi</span>
            </>
          ) : (
            <>
              <Zap className="w-3.5 h-3.5" />
              <span>خلوي</span>
            </>
          )}
        </div>
        
        <div className="text-xs text-dark-500">
          آخر ظهور: {new Date(device.lastSeen).toLocaleTimeString('ar-SA')}
        </div>
      </div>
    </div>
  );
}

// المكون الرئيسي
export function NetworkStatus() {
  const [devices, setDevices] = useState<NetworkDevice[]>(defaultDevices);
  const [autoReconnect, setAutoReconnect] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(Date.now());
  const [isRefreshing, setIsRefreshing] = useState(false);

  // إحصائيات الشبكة
  const stats: NetworkStats = {
    totalDevices: devices.length,
    onlineDevices: devices.filter(d => d.status === "online").length,
    avgLatency: devices
      .filter(d => d.status === "online")
      .reduce((acc, d) => acc + d.latency, 0) / devices.filter(d => d.status === "online").length || 0,
    totalBandwidth: {
      up: devices.reduce((acc, d) => acc + d.bandwidth.up, 0),
      down: devices.reduce((acc, d) => acc + d.bandwidth.down, 0),
    },
  };

  // محاكاة تحديثات حية
  useEffect(() => {
    const interval = setInterval(() => {
      setDevices(prev => prev.map(device => {
        if (device.id === "local") return device;
        
        // محاكاة تغيرات عشوائية
        const variation = (Math.random() - 0.5) * 5;
        const newLatency = Math.max(1, device.latency + variation);
        
        return {
          ...device,
          latency: device.status === "online" ? newLatency : device.latency,
          bandwidth: device.status === "online" ? {
            up: Math.max(0, device.bandwidth.up + (Math.random() - 0.5) * 10),
            down: Math.max(0, device.bandwidth.down + (Math.random() - 0.5) * 20),
          } : device.bandwidth,
          lastSeen: device.status !== "offline" ? new Date() : device.lastSeen,
        };
      }));
      
      setLastUpdate(Date.now());
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // تحديث يدوي
  const handleRefresh = useCallback(async (deviceId?: string) => {
    setIsRefreshing(true);
    
    // محاكاة فحص الاتصال
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    if (deviceId) {
      setDevices(prev => prev.map(d => 
        d.id === deviceId 
          ? { ...d, lastSeen: new Date(), latency: Math.max(1, d.latency + (Math.random() - 0.5) * 3) }
          : d
      ));
    }
    
    setIsRefreshing(false);
    setLastUpdate(Date.now());
  }, []);

  // إعادة الاتصال
  const handleReconnect = useCallback(async (deviceId: string) => {
    setDevices(prev => prev.map(d => 
      d.id === deviceId 
        ? { ...d, status: "degraded" as const }
        : d
    ));
    
    // محاكاة محاولة الاتصال
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    setDevices(prev => prev.map(d => 
      d.id === deviceId 
        ? { 
            ...d, 
            status: "online" as const,
            latency: 20 + Math.random() * 30,
            lastSeen: new Date(),
          }
        : d
    ));
  }, []);

  return (
    <div className="h-full flex flex-col bg-dark-900">
      {/* رأس الصفحة */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-dark-700">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary-500/10 rounded-lg">
            <Router className="w-6 h-6 text-primary-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-dark-100">حالة الشبكة</h1>
            <p className="text-sm text-dark-400">
              {stats.onlineDevices} من {stats.totalDevices} أجهزة متصلة
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {/* تبديل الإعادة التلقائية */}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-dark-800 rounded-lg">
            <RefreshCw className={`w-4 h-4 text-dark-400 ${autoReconnect ? "" : "opacity-50"}`} />
            <span className="text-sm text-dark-300">إعادة الاتصال التلقائي</span>
            <button
              onClick={() => setAutoReconnect(!autoReconnect)}
              className={`relative w-10 h-5 rounded-full transition-colors ${
                autoReconnect ? "bg-primary-600" : "bg-dark-600"
              }`}
            >
              <span
                className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                  autoReconnect ? "translate-x-5" : ""
                }`}
              />
            </button>
          </div>
          
          <button
            onClick={() => handleRefresh()}
            disabled={isRefreshing}
            className="p-2 text-dark-400 hover:text-dark-200 hover:bg-dark-800 rounded-lg transition-colors"
          >
            <RefreshCw className={`w-5 h-5 ${isRefreshing ? "animate-spin" : ""}`} />
          </button>
          
          <button className="p-2 text-dark-400 hover:text-dark-200 hover:bg-dark-800 rounded-lg transition-colors">
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* إحصائيات الشبكة */}
      <div className="grid grid-cols-4 gap-4 px-6 py-4 border-b border-dark-700">
        <div className="bg-dark-800 rounded-lg p-4">
          <div className="flex items-center gap-2 text-dark-400 mb-2">
            <CheckCircle className="w-4 h-4 text-green-400" />
            <span className="text-sm">الأجهزة المتصلة</span>
          </div>
          <div className="text-2xl font-bold text-dark-100">
            {stats.onlineDevices} <span className="text-dark-500 text-lg">/ {stats.totalDevices}</span>
          </div>
        </div>
        
        <div className="bg-dark-800 rounded-lg p-4">
          <div className="flex items-center gap-2 text-dark-400 mb-2">
            <Activity className="w-4 h-4 text-primary-400" />
            <span className="text-sm">متوسط Latency</span>
          </div>
          <div className="text-2xl font-bold text-dark-100">
            {stats.avgLatency.toFixed(1)} <span className="text-dark-500 text-lg">ms</span>
          </div>
        </div>
        
        <div className="bg-dark-800 rounded-lg p-4">
          <div className="flex items-center gap-2 text-dark-400 mb-2">
            <ArrowUp className="w-4 h-4 text-green-400" />
            <span className="text-sm">الرفع الإجمالي</span>
          </div>
          <div className="text-2xl font-bold text-dark-100">
            {(stats.totalBandwidth.up / 1000).toFixed(1)} <span className="text-dark-500 text-lg">Gbps</span>
          </div>
        </div>
        
        <div className="bg-dark-800 rounded-lg p-4">
          <div className="flex items-center gap-2 text-dark-400 mb-2">
            <ArrowDown className="w-4 h-4 text-blue-400" />
            <span className="text-sm">التحميل الإجمالي</span>
          </div>
          <div className="text-2xl font-bold text-dark-100">
            {(stats.totalBandwidth.down / 1000).toFixed(1)} <span className="text-dark-500 text-lg">Gbps</span>
          </div>
        </div>
      </div>

      {/* قائمة الأجهزة */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="grid grid-cols-2 gap-4">
          {devices.map(device => (
            <DeviceCard
              key={device.id}
              device={device}
              onRefresh={(id) => handleRefresh(id)}
            />
          ))}
        </div>
        
        {/* آخر تحديث */}
        <div className="text-center mt-6 text-xs text-dark-500">
          آخر تحديث: {new Date(lastUpdate).toLocaleTimeString('ar-SA')}
        </div>
      </div>
    </div>
  );
}

// مكون مصغر للشريط العلوي
export function NetworkStatusBadge({
  onClick,
}: {
  onClick?: () => void;
}) {
  const [status, setStatus] = useState<"online" | "degraded" | "offline">("online");
  
  useEffect(() => {
    // محاكاة فحص الحالة
    const interval = setInterval(() => {
      const rand = Math.random();
      setStatus(rand > 0.9 ? "degraded" : "online");
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const config = {
    online: { icon: Wifi, color: "text-green-400", bg: "bg-green-500/10", border: "border-green-500/20" },
    degraded: { icon: AlertTriangle, color: "text-yellow-400", bg: "bg-yellow-500/10", border: "border-yellow-500/20" },
    offline: { icon: WifiOff, color: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/20" },
  };

  const { icon: Icon, color, bg, border } = config[status];

  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-3 py-1.5 ${bg} border ${border} rounded-lg transition-colors`}
    >
      <Icon className={`w-4 h-4 ${color} ${status === "online" ? "" : "animate-pulse"}`} />
      <span className={`text-sm ${color}`}>
        {status === "online" ? "متصل" : status === "degraded" ? "ضعيف" : "غير متصل"}
      </span>
    </button>
  );
}

export default NetworkStatus;
