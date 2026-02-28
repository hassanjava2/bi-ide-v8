/**
 * لوحة ERP - ERP Dashboard
 * مقاييس الأعمال، المخططات، حالة المخزون، رؤى AI
 */

import { useState, useEffect } from "react";
import { 
  TrendingUp, 
  TrendingDown,
  DollarSign,
  Package,
  Users,
  ShoppingCart,
  BarChart3,
  PieChart,
  Activity,
  Sparkles,
  ArrowUpRight,
  ArrowDownRight,
  Calendar,
  RefreshCw,
  Filter,
  Download,
  AlertTriangle,
  CheckCircle,
  Clock
} from "lucide-react";

// أنواع البيانات
interface Metric {
  id: string;
  label: string;
  value: string;
  change: number;
  changeLabel: string;
  icon: React.ElementType;
  color: string;
}

interface SalesData {
  month: string;
  sales: number;
  profit: number;
  expenses: number;
}

interface InventoryItem {
  id: string;
  name: string;
  sku: string;
  stock: number;
  minStock: number;
  status: "in_stock" | "low_stock" | "out_of_stock";
  lastUpdated: Date;
}

interface AIInsight {
  id: string;
  type: "success" | "warning" | "info";
  title: string;
  description: string;
  action?: string;
}

// مكون البطاقة الإحصائية
function MetricCard({ metric }: { metric: Metric }) {
  const isPositive = metric.change >= 0;
  const Icon = metric.icon;
  
  return (
    <div className="bg-dark-800 rounded-xl p-5 border border-dark-700 hover:border-dark-600 transition-colors">
      <div className="flex items-start justify-between mb-4">
        <div 
          className="p-3 rounded-lg"
          style={{ backgroundColor: `${metric.color}20` }}
        >
          <Icon className="w-6 h-6" style={{ color: metric.color }} />
        </div>
        <div className={`flex items-center gap-1 text-sm font-medium ${
          isPositive ? "text-green-400" : "text-red-400"
        }`}>
          {isPositive ? (
            <ArrowUpRight className="w-4 h-4" />
          ) : (
            <ArrowDownRight className="w-4 h-4" />
          )}
          {Math.abs(metric.change)}%
        </div>
      </div>
      
      <div className="text-2xl font-bold text-dark-100 mb-1">
        {metric.value}
      </div>
      <div className="text-sm text-dark-400">{metric.label}</div>
      <div className="text-xs text-dark-500 mt-2">{metric.changeLabel}</div>
    </div>
  );
}

// مكون الرسم البياني الشريطي
function BarChart({ data }: { data: SalesData[] }) {
  const maxValue = Math.max(...data.map(d => Math.max(d.sales, d.profit, d.expenses)));
  
  return (
    <div className="w-full h-64">
      <svg width="100%" height="100%" viewBox="0 0 800 200" preserveAspectRatio="none">
        {/* خطوط الشبكة */}
        {[0, 0.25, 0.5, 0.75, 1].map(t => (
          <line
            key={t}
            x1={60}
            y1={20 + t * 140}
            x2={780}
            y2={20 + t * 140}
            stroke="#334155"
            strokeWidth={0.5}
            strokeDasharray="4"
          />
        ))}
        
        {/* الأعمدة */}
        {data.map((d, i) => {
          const x = 80 + i * 120;
          const barWidth = 30;
          const salesHeight = (d.sales / maxValue) * 140;
          const profitHeight = (d.profit / maxValue) * 140;
          const expensesHeight = (d.expenses / maxValue) * 140;
          
          return (
            <g key={d.month}>
              {/* المبيعات */}
              <rect
                x={x}
                y={160 - salesHeight}
                width={barWidth}
                height={salesHeight}
                fill="#0ea5e9"
                rx={4}
              />
              {/* الأرباح */}
              <rect
                x={x + 35}
                y={160 - profitHeight}
                width={barWidth}
                height={profitHeight}
                fill="#22c55e"
                rx={4}
              />
              {/* المصاريف */}
              <rect
                x={x + 70}
                y={160 - expensesHeight}
                width={barWidth}
                height={expensesHeight}
                fill="#ef4444"
                rx={4}
              />
              
              {/* تسمية الشهر */}
              <text
                x={x + 50}
                y={185}
                fill="#64748b"
                fontSize={11}
                textAnchor="middle"
              >
                {d.month}
              </text>
            </g>
          );
        })}
        
        {/* المحور Y */}
        {[0, 0.25, 0.5, 0.75, 1].map(t => (
          <text
            key={t}
            x={50}
            y={160 - t * 140 + 4}
            fill="#64748b"
            fontSize={10}
            textAnchor="end"
          >
            {Math.round(maxValue * (1 - t)).toLocaleString()}
          </text>
        ))}
      </svg>
      
      {/* مفتاح الرسم البياني */}
      <div className="flex justify-center gap-6 mt-4">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-primary-500 rounded" />
          <span className="text-xs text-dark-400">المبيعات</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500 rounded" />
          <span className="text-xs text-dark-400">الأرباح</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500 rounded" />
          <span className="text-xs text-dark-400">المصاريف</span>
        </div>
      </div>
    </div>
  );
}

// مكون حالة المخزون
function InventoryStatus({ items }: { items: InventoryItem[] }) {
  const getStatusColor = (status: InventoryItem["status"]) => {
    switch (status) {
      case "in_stock": return "bg-green-500";
      case "low_stock": return "bg-yellow-500";
      case "out_of_stock": return "bg-red-500";
    }
  };

  const getStatusText = (status: InventoryItem["status"]) => {
    switch (status) {
      case "in_stock": return "متوفر";
      case "low_stock": return "منخفض";
      case "out_of_stock": return "نفذ";
    }
  };

  return (
    <div className="space-y-3">
      {items.map(item => (
        <div 
          key={item.id}
          className="flex items-center justify-between p-3 bg-dark-900 rounded-lg hover:bg-dark-700/50 transition-colors"
        >
          <div className="flex items-center gap-3">
            <div className={`w-2 h-10 rounded-full ${getStatusColor(item.status)}`} />
            <div>
              <div className="font-medium text-dark-200">{item.name}</div>
              <div className="text-xs text-dark-500">SKU: {item.sku}</div>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-lg font-bold text-dark-100">{item.stock}</div>
              <div className="text-xs text-dark-500">الحد الأدنى: {item.minStock}</div>
            </div>
            <span className={`px-2 py-1 rounded text-xs font-medium ${
              item.status === "in_stock" ? "bg-green-500/20 text-green-400" :
              item.status === "low_stock" ? "bg-yellow-500/20 text-yellow-400" :
              "bg-red-500/20 text-red-400"
            }`}>
              {getStatusText(item.status)}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

// مكون رؤى AI
function AIInsights({ insights }: { insights: AIInsight[] }) {
  const getIcon = (type: AIInsight["type"]) => {
    switch (type) {
      case "success": return CheckCircle;
      case "warning": return AlertTriangle;
      case "info": return Sparkles;
    }
  };

  const getColor = (type: AIInsight["type"]) => {
    switch (type) {
      case "success": return "text-green-400 bg-green-500/10 border-green-500/20";
      case "warning": return "text-yellow-400 bg-yellow-500/10 border-yellow-500/20";
      case "info": return "text-primary-400 bg-primary-500/10 border-primary-500/20";
    }
  };

  return (
    <div className="space-y-3">
      {insights.map(insight => {
        const Icon = getIcon(insight.type);
        return (
          <div 
            key={insight.id}
            className={`p-4 rounded-lg border ${getColor(insight.type)}`}
          >
            <div className="flex items-start gap-3">
              <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h4 className="font-medium mb-1">{insight.title}</h4>
                <p className="text-sm opacity-80 mb-2">{insight.description}</p>
                {insight.action && (
                  <button className="text-sm font-medium hover:underline">
                    {insight.action} →
                  </button>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// مكون الإجراء السريع
function QuickAction({ 
  icon: Icon, 
  label, 
  onClick 
}: { 
  icon: React.ElementType; 
  label: string; 
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 p-3 bg-dark-800 hover:bg-dark-700 rounded-lg transition-colors text-left group"
    >
      <div className="p-2 bg-dark-900 rounded-lg group-hover:bg-dark-600 transition-colors">
        <Icon className="w-5 h-5 text-primary-400" />
      </div>
      <span className="text-dark-200 font-medium">{label}</span>
    </button>
  );
}

// المكون الرئيسي
export function ERPDashboard() {
  // البيانات
  const [metrics] = useState<Metric[]>([
    {
      id: "1",
      label: "إجمالي المبيعات",
      value: "$124,500",
      change: 12.5,
      changeLabel: "مقارنة بالشهر الماضي",
      icon: DollarSign,
      color: "#0ea5e9",
    },
    {
      id: "2",
      label: "الطلبات",
      value: "1,429",
      change: 8.2,
      changeLabel: "مقارنة بالشهر الماضي",
      icon: ShoppingCart,
      color: "#8b5cf6",
    },
    {
      id: "3",
      label: "العملاء",
      value: "892",
      change: -2.4,
      changeLabel: "مقارنة بالشهر الماضي",
      icon: Users,
      color: "#22c55e",
    },
    {
      id: "4",
      label: "المنتجات",
      value: "456",
      change: 5.1,
      changeLabel: "مقارنة بالشهر الماضي",
      icon: Package,
      color: "#f97316",
    },
  ]);

  const [salesData] = useState<SalesData[]>([
    { month: "يناير", sales: 65000, profit: 25000, expenses: 35000 },
    { month: "فبراير", sales: 72000, profit: 28000, expenses: 38000 },
    { month: "مارس", sales: 68000, profit: 26000, expenses: 36000 },
    { month: "أبريل", sales: 85000, profit: 35000, expenses: 42000 },
    { month: "مايو", sales: 92000, profit: 38000, expenses: 45000 },
    { month: "يونيو", sales: 105000, profit: 45000, expenses: 48000 },
  ]);

  const [inventoryItems] = useState<InventoryItem[]>([
    { id: "1", name: "لابتوب Dell XPS 15", sku: "LAP-DEL-001", stock: 45, minStock: 10, status: "in_stock", lastUpdated: new Date() },
    { id: "2", name: "آيفون 15 Pro", sku: "MOB-APL-001", stock: 8, minStock: 15, status: "low_stock", lastUpdated: new Date() },
    { id: "3", name: "سماعات AirPods", sku: "AUD-APL-001", stock: 120, minStock: 20, status: "in_stock", lastUpdated: new Date() },
    { id: "4", name: "شاحن MagSafe", sku: "ACC-APL-001", stock: 0, minStock: 10, status: "out_of_stock", lastUpdated: new Date() },
    { id: "5", name: "آيباد Pro 12.9", sku: "TAB-APL-001", stock: 25, minStock: 5, status: "in_stock", lastUpdated: new Date() },
  ]);

  const [aiInsights] = useState<AIInsight[]>([
    {
      id: "1",
      type: "success",
      title: "أداء مبيعات ممتاز!",
      description: "المبيعات هذا الشهر تفوق التوقعات بنسبة 15%. استمر في هذا الأداء.",
    },
    {
      id: "2",
      type: "warning",
      title: "مخزون منخفض",
      description: "3 منتجات على وشك النفاد. يُنصح بإعادة الطلب.",
      action: "عرض المنتجات",
    },
    {
      id: "3",
      type: "info",
      title: "فرصة نمو",
      description: "المنتجات الإلكترونية تحقق نمواً بنسبة 25%. فكر في زيادة المخزون.",
    },
  ]);

  const [lastUpdate, setLastUpdate] = useState(Date.now());

  // محاكاة تحديث حي
  useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdate(Date.now());
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-full flex flex-col bg-dark-900 overflow-auto">
      {/* رأس الصفحة */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-dark-700">
        <div>
          <h1 className="text-xl font-bold text-dark-100">لوحة تحكم ERP</h1>
          <p className="text-sm text-dark-400">نظرة عامة على أداء الأعمال</p>
        </div>
        
        <div className="flex items-center gap-3">
          {/* فترة التاريخ */}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-dark-800 rounded-lg text-dark-300 text-sm">
            <Calendar className="w-4 h-4" />
            <span>آخر 30 يوم</span>
          </div>
          
          {/* أزرار التحكم */}
          <button className="p-2 text-dark-400 hover:text-dark-200 hover:bg-dark-800 rounded-lg transition-colors">
            <RefreshCw className="w-4 h-4" />
          </button>
          <button className="p-2 text-dark-400 hover:text-dark-200 hover:bg-dark-800 rounded-lg transition-colors">
            <Filter className="w-4 h-4" />
          </button>
          <button className="p-2 text-dark-400 hover:text-dark-200 hover:bg-dark-800 rounded-lg transition-colors">
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* المحتوى */}
      <div className="flex-1 p-6 space-y-6">
        {/* المقاييس */}
        <div className="grid grid-cols-4 gap-4">
          {metrics.map(metric => (
            <MetricCard key={metric.id} metric={metric} />
          ))}
        </div>

        <div className="grid grid-cols-3 gap-6">
          {/* رسم المبيعات */}
          <div className="col-span-2 bg-dark-800 rounded-xl p-5 border border-dark-700">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-semibold text-dark-100">تحليل المبيعات</h3>
                <p className="text-sm text-dark-400">الأشهر الستة الماضية</p>
              </div>
              <div className="flex items-center gap-2">
                <button className="p-2 text-dark-400 hover:text-dark-200 hover:bg-dark-700 rounded-lg transition-colors">
                  <BarChart3 className="w-4 h-4" />
                </button>
                <button className="p-2 text-dark-400 hover:text-dark-200 hover:bg-dark-700 rounded-lg transition-colors">
                  <PieChart className="w-4 h-4" />
                </button>
              </div>
            </div>
            <BarChart data={salesData} />
          </div>

          {/* الإجراءات السريعة */}
          <div className="bg-dark-800 rounded-xl p-5 border border-dark-700">
            <h3 className="font-semibold text-dark-100 mb-4">إجراءات سريعة</h3>
            <div className="space-y-2">
              <QuickAction 
                icon={ShoppingCart} 
                label="طلب جديد" 
                onClick={() => console.log("New order")}
              />
              <QuickAction 
                icon={Package} 
                label="إضافة منتج" 
                onClick={() => console.log("New product")}
              />
              <QuickAction 
                icon={Users} 
                label="عميل جديد" 
                onClick={() => console.log("New customer")}
              />
              <QuickAction 
                icon={BarChart3} 
                label="تقرير مبيعات" 
                onClick={() => console.log("Sales report")}
              />
              <QuickAction 
                icon={DollarSign} 
                label="فاتورة جديدة" 
                onClick={() => console.log("New invoice")}
              />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          {/* حالة المخزون */}
          <div className="bg-dark-800 rounded-xl p-5 border border-dark-700">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-semibold text-dark-100">حالة المخزون</h3>
                <p className="text-sm text-dark-400">آخر التحديثات</p>
              </div>
              <span className="text-xs text-dark-500">
                آخر تحديث: {new Date(lastUpdate).toLocaleTimeString('ar-SA')}
              </span>
            </div>
            <InventoryStatus items={inventoryItems} />
          </div>

          {/* رؤى AI */}
          <div className="bg-dark-800 rounded-xl p-5 border border-dark-700">
            <div className="flex items-center gap-2 mb-4">
              <Sparkles className="w-5 h-5 text-primary-400" />
              <h3 className="font-semibold text-dark-100">رؤى AI</h3>
            </div>
            <AIInsights insights={aiInsights} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default ERPDashboard;
