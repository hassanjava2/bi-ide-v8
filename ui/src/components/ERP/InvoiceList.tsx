import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { api } from '@/lib/api';
import { Plus, Search, CheckCircle, Clock, DollarSign } from 'lucide-react';

interface Invoice {
  id: string;
  invoice_number: string;
  customer_name: string;
  amount: number;
  status: 'pending' | 'paid' | 'overdue';
  created_at: string;
  due_date: string;
}

interface InvoiceListProps {
  invoices?: Invoice[];
  onRefresh?: () => void;
}

export const InvoiceList: React.FC<InvoiceListProps> = ({ 
  invoices: initialInvoices = [],
  onRefresh 
}) => {
  const [invoices, setInvoices] = useState<Invoice[]>(initialInvoices);
  const [searchQuery, setSearchQuery] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [newInvoice, setNewInvoice] = useState({
    customer_name: '',
    amount: '',
    due_date: ''
  });

  const fetchInvoices = useCallback(async () => {
    try {
      const response = await api.get('/erp/invoices');
      setInvoices(response.data.invoices || []);
    } catch (error) {
      console.error('Failed to fetch invoices:', error);
    }
  }, []);

  const handleMarkPaid = async (invoiceId: string) => {
    try {
      await api.post(`/erp/invoices/${invoiceId}/pay`, {});
      setInvoices(prev => prev.map(inv => 
        inv.id === invoiceId ? { ...inv, status: 'paid' } : inv
      ));
      onRefresh?.();
    } catch (error) {
      console.error('Failed to mark invoice as paid:', error);
    }
  };

  const handleCreateInvoice = async () => {
    if (!newInvoice.customer_name || !newInvoice.amount) return;
    
    try {
      await api.post('/erp/invoices', {
        customer_name: newInvoice.customer_name,
        amount: parseFloat(newInvoice.amount),
        due_date: newInvoice.due_date
      });
      setIsCreating(false);
      setNewInvoice({ customer_name: '', amount: '', due_date: '' });
      fetchInvoices();
      onRefresh?.();
    } catch (error) {
      console.error('Failed to create invoice:', error);
    }
  };

  const filteredInvoices = invoices.filter(inv => 
    inv.customer_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    inv.invoice_number.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'paid': return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'pending': return <Clock className="w-4 h-4 text-yellow-400" />;
      case 'overdue': return <DollarSign className="w-4 h-4 text-red-400" />;
      default: return null;
    }
  };

  const getStatusClass = (status: string) => {
    switch (status) {
      case 'paid': return 'bg-green-500/20 text-green-400';
      case 'pending': return 'bg-yellow-500/20 text-yellow-400';
      case 'overdue': return 'bg-red-500/20 text-red-400';
      default: return 'bg-gray-500/20 text-gray-400';
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>الفواتير</CardTitle>
          <Button onClick={() => setIsCreating(true)} size="sm">
            <Plus className="w-4 h-4 mr-1" />
            فاتورة جديدة
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="mb-4 relative">
          <Search className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <Input
            placeholder="بحث في الفواتير..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pr-10"
          />
        </div>

        {isCreating && (
          <div className="mb-4 p-4 bg-white/5 rounded-lg space-y-3">
            <h4 className="text-sm font-medium">فاتورة جديدة</h4>
            <Input
              placeholder="اسم العميل"
              value={newInvoice.customer_name}
              onChange={(e) => setNewInvoice({ ...newInvoice, customer_name: e.target.value })}
            />
            <Input
              placeholder="المبلغ"
              type="number"
              value={newInvoice.amount}
              onChange={(e) => setNewInvoice({ ...newInvoice, amount: e.target.value })}
            />
            <Input
              placeholder="تاريخ الاستحقاق"
              type="date"
              value={newInvoice.due_date}
              onChange={(e) => setNewInvoice({ ...newInvoice, due_date: e.target.value })}
            />
            <div className="flex gap-2">
              <Button onClick={handleCreateInvoice} size="sm">إنشاء</Button>
              <Button variant="ghost" onClick={() => setIsCreating(false)} size="sm">إلغاء</Button>
            </div>
          </div>
        )}

        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredInvoices.map((invoice) => (
            <div
              key={invoice.id}
              className="flex items-center justify-between p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors"
            >
              <div className="flex items-center gap-3">
                {getStatusIcon(invoice.status)}
                <div>
                  <p className="font-medium text-sm">{invoice.invoice_number}</p>
                  <p className="text-xs text-gray-400">{invoice.customer_name}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="text-left">
                  <p className="font-medium">${invoice.amount.toLocaleString()}</p>
                  <span className={`text-xs px-2 py-0.5 rounded ${getStatusClass(invoice.status)}`}>
                    {invoice.status === 'paid' ? 'مدفوع' : 
                     invoice.status === 'pending' ? 'معلق' : 'متأخر'}
                  </span>
                </div>
                {invoice.status !== 'paid' && (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => handleMarkPaid(invoice.id)}
                  >
                    دفع
                  </Button>
                )}
              </div>
            </div>
          ))}
          {filteredInvoices.length === 0 && (
            <p className="text-center text-gray-400 py-4">لا توجد فواتير</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
