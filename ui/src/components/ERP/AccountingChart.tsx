import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { api } from '@/lib/api';
import { ChevronRight, ChevronDown, Folder, FileText, Plus } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface Account {
  id: string;
  code: string;
  name: string;
  type: 'asset' | 'liability' | 'equity' | 'revenue' | 'expense';
  balance: number;
  children?: Account[];
}

interface AccountingChartProps {
  accounts?: Account[];
}

const AccountTypeLabels: Record<string, string> = {
  asset: 'أصول',
  liability: 'خصوم',
  equity: 'حقوق ملكية',
  revenue: 'إيرادات',
  expense: 'مصروفات'
};

const AccountTypeColors: Record<string, string> = {
  asset: 'text-green-400',
  liability: 'text-red-400',
  equity: 'text-blue-400',
  revenue: 'text-purple-400',
  expense: 'text-orange-400'
};

const TreeItem: React.FC<{
  account: Account;
  depth: number;
  expanded: Set<string>;
  onToggle: (id: string) => void;
}> = ({ account, depth, expanded, onToggle }) => {
  const isExpanded = expanded.has(account.id);
  const hasChildren = account.children && account.children.length > 0;

  return (
    <div>
      <div
        className="flex items-center justify-between p-2 hover:bg-white/5 rounded cursor-pointer"
        style={{ paddingRight: `${depth * 20 + 8}px` }}
        onClick={() => hasChildren && onToggle(account.id)}
      >
        <div className="flex items-center gap-2">
          {hasChildren ? (
            isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />
          ) : (
            <FileText className="w-4 h-4 text-gray-400" />
          )}
          <Folder className={`w-4 h-4 ${AccountTypeColors[account.type]}`} />
          <span className="text-sm">{account.code} - {account.name}</span>
        </div>
        <span className={`text-sm font-medium ${account.balance >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          ${Math.abs(account.balance).toLocaleString()}
        </span>
      </div>
      {hasChildren && isExpanded && (
        <div>
          {account.children!.map(child => (
            <TreeItem
              key={child.id}
              account={child}
              depth={depth + 1}
              expanded={expanded}
              onToggle={onToggle}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export const AccountingChart: React.FC<AccountingChartProps> = ({ accounts: initialAccounts }) => {
  const [accounts, setAccounts] = useState<Account[]>(initialAccounts || []);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(!initialAccounts);

  useEffect(() => {
    if (!initialAccounts) {
      fetchAccounts();
    }
  }, [initialAccounts]);

  const fetchAccounts = async () => {
    try {
      const response = await api.get('/erp/accounts');
      setAccounts(response.data.accounts || []);
    } catch (error) {
      console.error('Failed to fetch accounts:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleToggle = (id: string) => {
    setExpanded(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  const totalBalance = accounts.reduce((sum, acc) => sum + acc.balance, 0);

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>شجرة الحسابات</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-center text-gray-400 py-4">جاري التحميل...</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>شجرة الحسابات</CardTitle>
          <Button size="sm" variant="ghost">
            <Plus className="w-4 h-4 mr-1" />
            حساب جديد
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="mb-4 p-3 bg-white/5 rounded-lg">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">الرصيد الإجمالي</span>
            <span className={`text-lg font-bold ${totalBalance >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ${Math.abs(totalBalance).toLocaleString()}
            </span>
          </div>
        </div>

        <div className="space-y-1 max-h-96 overflow-y-auto">
          {accounts.map(account => (
            <TreeItem
              key={account.id}
              account={account}
              depth={0}
              expanded={expanded}
              onToggle={handleToggle}
            />
          ))}
          {accounts.length === 0 && (
            <p className="text-center text-gray-400 py-4">لا توجد حسابات</p>
          )}
        </div>

        <div className="mt-4 pt-4 border-t border-white/10">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
            {Object.entries(AccountTypeLabels).map(([type, label]) => (
              <div key={type} className="flex items-center gap-1">
                <Folder className={`w-3 h-3 ${AccountTypeColors[type]}`} />
                <span className="text-gray-400">{label}</span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
