import React, { useState } from 'react';
import { useInventory } from '@/hooks/useERP-legacy';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { Package, AlertTriangle, RefreshCw, Plus, Minus } from 'lucide-react';

export const InventoryManager: React.FC = () => {
  const { items, lowStock, loading, error, adjustStock, refresh } = useInventory();
  const [adjustingId, setAdjustingId] = useState<string | null>(null);
  const [adjustmentQty, setAdjustmentQty] = useState<Record<string, string>>({});

  const handleAdjust = async (productId: string) => {
    const qty = parseInt(adjustmentQty[productId] || '0');
    if (qty === 0) return;

    setAdjustingId(productId);
    await adjustStock(productId, qty, 'Manual adjustment');
    setAdjustingId(null);
    setAdjustmentQty((prev) => ({ ...prev, [productId]: '' }));
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
    <div className="space-y-6">
      {/* Low Stock Alert */}
      {lowStock.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-amber-800 mb-2">
            <AlertTriangle className="w-5 h-5" />
            <h3 className="font-semibold">تنبيه: مخزون منخفض</h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {lowStock.map((item) => (
              <span key={item.id} className="text-sm bg-amber-100 text-amber-700 px-2 py-1 rounded">
                {item.name} ({item.quantity} متبقي)
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold">المخزون</h2>
        <Button onClick={refresh} variant="outline" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          تحديث
        </Button>
      </div>

      {items.length === 0 ? (
        <div className="text-center p-8 text-gray-500">
          <Package className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>لا توجد منتجات في المخزون</p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {items.map((item) => (
            <Card key={item.id} className={item.quantity <= (item.reorder_level || item.reorderPoint || 0) ? 'border-amber-300' : ''}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${
                      item.quantity <= (item.reorder_level || item.reorderPoint || 0) ? 'bg-amber-100 text-amber-600' : 'bg-blue-100 text-blue-600'
                    }`}>
                      <Package className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-semibold">{item.name}</h3>
                      <p className="text-sm text-gray-500">SKU: {item.sku}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold">{item.quantity}</div>
                    <p className="text-xs text-gray-400">وحدة</p>
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t">
                  <div className="flex gap-2">
                    <Input
                      type="number"
                      placeholder="الكمية"
                      value={adjustmentQty[item.id] || ''}
                      onChange={(e) => setAdjustmentQty((prev) => ({ ...prev, [item.id]: e.target.value }))}
                      className="w-24"
                    />
                    <Button
                      onClick={() => handleAdjust(item.id)}
                      disabled={adjustingId === item.id || !adjustmentQty[item.id]}
                      size="sm"
                      variant="outline"
                    >
                      {adjustingId === item.id ? (
                        <LoadingSpinner size="sm" />
                      ) : (
                        <>
                          <Plus className="w-4 h-4" />
                          <Minus className="w-4 h-4" />
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default InventoryManager;
