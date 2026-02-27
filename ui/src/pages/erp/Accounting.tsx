import { useState, useEffect } from 'react'
import { 
  Calculator, 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  FileText,
  Plus,
  Search,
  Filter,
  Download
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'

interface Account {
  id: string
  code: string
  name: string
  name_ar: string
  type: 'asset' | 'liability' | 'equity' | 'revenue' | 'expense'
  balance: number
}

interface Transaction {
  id: string
  date: string
  description: string
  debit: number
  credit: number
  reference: string
}

export default function AccountingPage() {
  const [accounts, setAccounts] = useState<Account[]>([])
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [activeTab, setActiveTab] = useState<'accounts' | 'transactions' | 'reports'>('accounts')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchData()
  }, [activeTab])

  const fetchData = async () => {
    setLoading(true)
    try {
      if (activeTab === 'accounts') {
        const response = await api.get('/erp/accounts')
        setAccounts(response.data)
      } else if (activeTab === 'transactions') {
        const response = await api.get('/erp/transactions')
        setTransactions(response.data)
      }
    } catch (error) {
      console.error('Error fetching data:', error)
    }
    setLoading(false)
  }

  const totalAssets = accounts
    .filter(a => a.type === 'asset')
    .reduce((sum, a) => sum + a.balance, 0)

  const totalLiabilities = accounts
    .filter(a => a.type === 'liability')
    .reduce((sum, a) => sum + a.balance, 0)

  const totalEquity = accounts
    .filter(a => a.type === 'equity')
    .reduce((sum, a) => sum + a.balance, 0)

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">المحاسبة</h1>
          <p className="text-gray-400">نظام المحاسبة المالية المزدوجة</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline">
            <Download className="w-4 h-4 mr-2" />
            تصدير
          </Button>
          <Button>
            <Plus className="w-4 h-4 mr-2" />
            قيد جديد
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">الأصول</p>
                <p className="text-2xl font-bold text-green-400">
                  ${totalAssets.toLocaleString()}
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">الخصوم</p>
                <p className="text-2xl font-bold text-red-400">
                  ${totalLiabilities.toLocaleString()}
                </p>
              </div>
              <TrendingDown className="w-8 h-8 text-red-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">حقوق الملكية</p>
                <p className="text-2xl font-bold text-blue-400">
                  ${totalEquity.toLocaleString()}
                </p>
              </div>
              <DollarSign className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">ميزان المراجعة</p>
                <p className={`text-2xl font-bold ${
                  Math.abs(totalAssets - totalLiabilities - totalEquity) < 0.01 
                    ? 'text-green-400' 
                    : 'text-red-400'
                }`}>
                  {Math.abs(totalAssets - totalLiabilities - totalEquity) < 0.01 
                    ? 'متوازن' 
                    : 'غير متوازن'}
                </p>
              </div>
              <Calculator className="w-8 h-8 text-bi-accent" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-white/10">
        {[
          { id: 'accounts', label: 'شجرة الحسابات', icon: FileText },
          { id: 'transactions', label: 'القيود المحاسبية', icon: Calculator },
          { id: 'reports', label: 'التقارير', icon: TrendingUp },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center gap-2 px-4 py-2 border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-bi-accent text-white'
                : 'border-transparent text-gray-400 hover:text-white'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin w-8 h-8 border-2 border-bi-accent border-t-transparent rounded-full" />
        </div>
      ) : (
        <>
          {activeTab === 'accounts' && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>شجرة الحسابات</CardTitle>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      <Search className="w-4 h-4 mr-2" />
                      بحث
                    </Button>
                    <Button variant="outline" size="sm">
                      <Filter className="w-4 h-4 mr-2" />
                      تصفية
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="text-right py-2">الكود</th>
                      <th className="text-right py-2">الاسم</th>
                      <th className="text-right py-2">النوع</th>
                      <th className="text-right py-2">الرصيد</th>
                    </tr>
                  </thead>
                  <tbody>
                    {accounts.map((account) => (
                      <tr key={account.id} className="border-b border-white/5">
                        <td className="py-2 font-mono">{account.code}</td>
                        <td className="py-2">
                          <div>{account.name}</div>
                          <div className="text-sm text-gray-400">{account.name_ar}</div>
                        </td>
                        <td className="py-2">
                          <span className={`px-2 py-1 rounded text-xs ${
                            account.type === 'asset' ? 'bg-green-500/20 text-green-400' :
                            account.type === 'liability' ? 'bg-red-500/20 text-red-400' :
                            account.type === 'equity' ? 'bg-blue-500/20 text-blue-400' :
                            account.type === 'revenue' ? 'bg-purple-500/20 text-purple-400' :
                            'bg-orange-500/20 text-orange-400'
                          }`}>
                            {account.type}
                          </span>
                        </td>
                        <td className={`py-2 font-mono ${
                          account.balance >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          ${account.balance.toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}

          {activeTab === 'transactions' && (
            <Card>
              <CardHeader>
                <CardTitle>القيود المحاسبية</CardTitle>
              </CardHeader>
              <CardContent>
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="text-right py-2">التاريخ</th>
                      <th className="text-right py-2">الوصف</th>
                      <th className="text-right py-2">مدين</th>
                      <th className="text-right py-2">دائن</th>
                      <th className="text-right py-2">المرجع</th>
                    </tr>
                  </thead>
                  <tbody>
                    {transactions.map((tx) => (
                      <tr key={tx.id} className="border-b border-white/5">
                        <td className="py-2">{new Date(tx.date).toLocaleDateString('ar-SA')}</td>
                        <td className="py-2">{tx.description}</td>
                        <td className="py-2 text-green-400">{tx.debit > 0 ? `$${tx.debit}` : '-'}</td>
                        <td className="py-2 text-red-400">{tx.credit > 0 ? `$${tx.credit}` : '-'}</td>
                        <td className="py-2 font-mono text-sm">{tx.reference}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}

          {activeTab === 'reports' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>الميزانية العمومية</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between py-2 border-b border-white/10">
                      <span>الأصول</span>
                      <span className="font-mono">${totalAssets.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between py-2 border-b border-white/10">
                      <span>الخصوم</span>
                      <span className="font-mono">${totalLiabilities.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between py-2 border-b border-white/10">
                      <span>حقوق الملكية</span>
                      <span className="font-mono">${totalEquity.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between py-2 font-bold">
                      <span>المجموع</span>
                      <span className={`font-mono ${
                        Math.abs(totalAssets - totalLiabilities - totalEquity) < 0.01
                          ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {Math.abs(totalAssets - totalLiabilities - totalEquity) < 0.01
                          ? '✓ متوازن' : '✗ خطأ'}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>قائمة الدخل</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between py-2 border-b border-white/10">
                      <span>الإيرادات</span>
                      <span className="font-mono text-green-400">$0</span>
                    </div>
                    <div className="flex justify-between py-2 border-b border-white/10">
                      <span>المصروفات</span>
                      <span className="font-mono text-red-400">$0</span>
                    </div>
                    <div className="flex justify-between py-2 font-bold">
                      <span>صافي الدخل</span>
                      <span className="font-mono">$0</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </>
      )}
    </div>
  )
}
