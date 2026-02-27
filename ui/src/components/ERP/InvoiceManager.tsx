import React, { useState } from 'react';
import { useInvoices } from '@/hooks/useERP-legacy';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { FileText, CheckCircle, DollarSign } from 'lucide-react';

interface InvoiceManagerProps {
  status?: string;
}

export const InvoiceManager: React.FC<InvoiceManagerProps> = ({ status }) => {
  const { invoices, loading, error, markAsPaid, refresh } = useInvoices(status);
  const [processingId, setProcessingId] = useState<string | null>(null);

  const handleMarkAsPaid = async (invoiceId: string) => {
    setProcessingId(invoiceId);
    await markAsPaid(invoiceId);
    setProcessingId(null);
  };

  if (loading) {
    return (
      <div className="flex justify-center p-8">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
        <p className="text-red-600">Error: {error}</p>
        <Button onClick={refresh} variant="outline" className="mt-2">
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold">الفواتير</h2>
        <Button onClick={refresh} variant="outline" size="sm">
          تحديث
        </Button>
      </div>

      {invoices.length === 0 ? (
        <div className="text-center p-8 text-gray-500">
          <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>لا توجد فواتير</p>
        </div>
      ) : (
        <div className="grid gap-4">
          {invoices.map((invoice) => (
            <Card key={invoice.id} className="hover:shadow-md transition-shadow">
              <CardContent className="p-4">
                <div className="flex justify-between items-start">
                  <div className="flex items-start gap-3">
                    <div className={`p-2 rounded-lg ${
                      invoice.status === 'paid' ? 'bg-green-100 text-green-600' :
                      invoice.status === 'overdue' ? 'bg-red-100 text-red-600' :
                      'bg-yellow-100 text-yellow-600'
                    }`}>
                      <FileText className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-semibold">{invoice.invoice_number}</h3>
                      <p className="text-sm text-gray-500">{invoice.customer_name}</p>
                      <p className="text-xs text-gray-400">
                        {new Date(invoice.created_at).toLocaleDateString('ar-SA')}
                      </p>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="flex items-center gap-1 text-lg font-bold">
                      <DollarSign className="w-5 h-5" />
                      {invoice.amount.toLocaleString()}
                    </div>
                    <span className={`inline-block px-2 py-1 text-xs rounded-full ${
                      invoice.status === 'paid' ? 'bg-green-100 text-green-700' :
                      invoice.status === 'overdue' ? 'bg-red-100 text-red-700' :
                      'bg-yellow-100 text-yellow-700'
                    }`}>
                      {invoice.status === 'paid' ? 'مدفوعة' :
                       invoice.status === 'overdue' ? 'متأخرة' : 'معلقة'}
                    </span>
                  </div>
                </div>

                {invoice.status !== 'paid' && (
                  <div className="mt-4 pt-4 border-t flex justify-end">
                    <Button
                      onClick={() => handleMarkAsPaid(invoice.id)}
                      disabled={processingId === invoice.id}
                      size="sm"
                      className="gap-2"
                    >
                      {processingId === invoice.id ? (
                        <LoadingSpinner size="sm" />
                      ) : (
                        <>
                          <CheckCircle className="w-4 h-4" />
                          تم التحصيل
                        </>
                      )}
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default InvoiceManager;
