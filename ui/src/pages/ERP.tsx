import { memo, useMemo, useCallback, useState } from 'react'
import { Building2, Receipt, Package, Users, DollarSign, TrendingUp, Plus, Search } from 'lucide-react'

// Static data moved outside component
const TABS = [
  { id: 'overview', label: 'نظرة عامة', icon: Building2 },
  { id: 'invoices', label: 'الفواتير', icon: Receipt },
  { id: 'inventory', label: 'المخزون', icon: Package },
  { id: 'employees', label: 'الموظفون', icon: Users },
  { id: 'finance', label: 'المالية', icon: DollarSign },
] as const

const STATS = [
  { label: 'إجمالي المبيعات', value: '125,000 $', change: '+12%', icon: TrendingUp },
  { label: 'الفواتير المعلقة', value: '23', change: '-5%', icon: Receipt },
  { label: 'قيمة المخزون', value: '450,000 $', change: '+3%', icon: Package },
  { label: 'الموظفون النشطون', value: '48', change: '+2', icon: Users },
] as const

const INVOICES = [
  { id: 'INV-001', customer: 'شركة التقنية', amount: 5000, status: 'paid' as const },
  { id: 'INV-002', customer: 'مؤسسة النور', amount: 3200, status: 'pending' as const },
  { id: 'INV-003', customer: 'مكتب المحاماة', amount: 7500, status: 'paid' as const },
]

const INVENTORY_ITEMS = [
  { name: 'لابتوب Dell', sku: 'LAP-001', qty: 3, min: 5 },
  { name: 'طابعة HP', sku: 'PRI-002', qty: 2, min: 5 },
  { name: 'ماوس لاسلكي', sku: 'MOU-003', qty: 8, min: 10 },
]

// Memoized Stat Card component
interface StatCardProps {
  label: string
  value: string
  change: string
  icon: typeof TrendingUp
}

const StatCard = memo(function StatCard({ label, value, change, icon: Icon }: StatCardProps) {
  const isPositive = change.startsWith('+')
  
  return (
    <div className="stat-card">
      <div className="flex items-center justify-between mb-2">
        <Icon className="w-6 h-6 text-bi-accent" />
        <span className={`text-sm ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
          {change}
        </span>
      </div>
      <p className="stat-value">{value}</p>
      <p className="stat-label">{label}</p>
    </div>
  )
}, (prevProps, nextProps) => {
  return prevProps.value === nextProps.value && prevProps.change === nextProps.change
})

// Memoized Tab Button component
interface TabButtonProps {
  id: string
  label: string
  icon: typeof Building2
  isActive: boolean
  onClick: (id: string) => void
}

const TabButton = memo(function TabButton({ id, label, icon: Icon, isActive, onClick }: TabButtonProps) {
  const handleClick = useCallback(() => {
    onClick(id)
  }, [id, onClick])

  return (
    <button
      onClick={handleClick}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap ${
        isActive 
          ? 'bg-bi-accent text-white' 
          : 'bg-white/5 text-gray-400 hover:bg-white/10'
      }`}
    >
      <Icon className="w-4 h-4" />
      {label}
    </button>
  )
}, (prevProps, nextProps) => {
  return prevProps.id === nextProps.id && prevProps.isActive === nextProps.isActive
})

// Memoized Invoice Item component
interface InvoiceItemProps {
  id: string
  customer: string
  amount: number
  status: 'paid' | 'pending'
}

const InvoiceItem = memo(function InvoiceItem({ id, customer, amount, status }: InvoiceItemProps) {
  return (
    <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
      <div>
        <p className="text-white font-medium">{id}</p>
        <p className="text-sm text-gray-400">{customer}</p>
      </div>
      <div className="text-left">
        <p className="text-white">{amount} $</p>
        <span className={`text-xs px-2 py-0.5 rounded ${
          status === 'paid' ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'
        }`}>
          {status === 'paid' ? 'مدفوع' : 'معلق'}
        </span>
      </div>
    </div>
  )
}, (prevProps, nextProps) => {
  return prevProps.id === nextProps.id && prevProps.status === nextProps.status
})

// Memoized Inventory Item component
interface InventoryItemProps {
  name: string
  sku: string
  qty: number
  min: number
}

const InventoryItem = memo(function InventoryItem({ name, sku, qty, min }: InventoryItemProps) {
  return (
    <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
      <div>
        <p className="text-white font-medium">{name}</p>
        <p className="text-sm text-gray-400">{sku}</p>
      </div>
      <div className="text-left">
        <p className="text-red-400">{qty} وحدة</p>
        <p className="text-xs text-gray-500">الحد: {min}</p>
      </div>
    </div>
  )
}, (prevProps, nextProps) => {
  return prevProps.sku === nextProps.sku && prevProps.qty === nextProps.qty
})

// Memoized Header component
const ERPHeader = memo(function ERPHeader() {
  return (
    <div className="glass-panel p-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-green-600 to-green-800 flex items-center justify-center">
            <Building2 className="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">نظام ERP</h1>
            <p className="text-gray-400">إدارة الموارد المؤسسية</p>
          </div>
        </div>
        <button className="btn-primary flex items-center gap-2">
          <Plus className="w-5 h-5" />
          عملية جديدة
        </button>
      </div>
    </div>
  )
})

// Memoized Overview Section component
const OverviewSection = memo(function OverviewSection() {
  const invoicesList = useMemo(() => {
    return INVOICES.map((inv) => (
      <InvoiceItem
        key={inv.id}
        id={inv.id}
        customer={inv.customer}
        amount={inv.amount}
        status={inv.status}
      />
    ))
  }, [])

  const inventoryList = useMemo(() => {
    return INVENTORY_ITEMS.map((item) => (
      <InventoryItem
        key={item.sku}
        name={item.name}
        sku={item.sku}
        qty={item.qty}
        min={item.min}
      />
    ))
  }, [])

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="glass-card p-4">
        <h3 className="text-white font-medium mb-4">آخر الفواتير</h3>
        <div className="space-y-3">
          {invoicesList}
        </div>
      </div>

      <div className="glass-card p-4">
        <h3 className="text-white font-medium mb-4">مخزون منخفض</h3>
        <div className="space-y-3">
          {inventoryList}
        </div>
      </div>
    </div>
  )
})

function ERP() {
  const [activeTab, setActiveTab] = useState('overview')
  const [searchQuery, setSearchQuery] = useState('')

  // Memoized tab click handler
  const handleTabClick = useCallback((id: string) => {
    setActiveTab(id)
  }, [])

  // Memoized search handler
  const handleSearchChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value)
  }, [])

  // Memoized stat cards list
  const statCardsList = useMemo(() => {
    return STATS.map((stat, index) => (
      <StatCard
        key={index}
        label={stat.label}
        value={stat.value}
        change={stat.change}
        icon={stat.icon}
      />
    ))
  }, [])

  // Memoized tab buttons list
  const tabButtonsList = useMemo(() => {
    return TABS.map((tab) => (
      <TabButton
        key={tab.id}
        id={tab.id}
        label={tab.label}
        icon={tab.icon}
        isActive={activeTab === tab.id}
        onClick={handleTabClick}
      />
    ))
  }, [activeTab, handleTabClick])

  // Memoized active tab label
  const activeTabLabel = useMemo(() => {
    return TABS.find(t => t.id === activeTab)?.label
  }, [activeTab])

  return (
    <div className="space-y-6">
      <ERPHeader />

      {/* التبويبات */}
      <div className="flex gap-2 overflow-x-auto">
        {tabButtonsList}
      </div>

      {/* الإحصائيات */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {statCardsList}
      </div>

      {/* المحتوى */}
      <div className="glass-panel p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold">{activeTabLabel}</h2>
          <div className="relative">
            <Search className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
            <input 
              type="text" 
              placeholder="بحث..." 
              value={searchQuery}
              onChange={handleSearchChange}
              className="input-field pr-10 w-64" 
            />
          </div>
        </div>

        {activeTab === 'overview' && <OverviewSection />}

        {activeTab !== 'overview' && (
          <div className="text-center py-12 text-gray-400">
            <p>قسم {activeTabLabel} قيد التطوير</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default memo(ERP)
