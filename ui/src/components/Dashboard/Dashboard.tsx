import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useERPDashboard } from '@/hooks/useERP-legacy';

interface StatCardProps {
  title: string;
  value: number | string;
  subtitle: string;
  valueClassName?: string;
}

const StatCard: React.FC<StatCardProps> = ({ title, value, subtitle, valueClassName }) => (
  <Card>
    <CardHeader>
      <CardTitle>{title}</CardTitle>
    </CardHeader>
    <CardContent>
      <div className={`text-2xl font-bold ${valueClassName || ''}`}>{value}</div>
      <p className="text-muted-foreground text-sm">{subtitle}</p>
    </CardContent>
  </Card>
);

const LoadingSkeleton: React.FC = () => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
    {[1, 2, 3, 4].map((i) => (
      <Card key={i} className="animate-pulse">
        <CardHeader>
          <div className="h-6 bg-gray-200 rounded w-24"></div>
        </CardHeader>
        <CardContent>
          <div className="h-8 bg-gray-200 rounded w-16 mb-2"></div>
          <div className="h-4 bg-gray-200 rounded w-32"></div>
        </CardContent>
      </Card>
    ))}
  </div>
);

const ErrorMessage: React.FC<{ error: string }> = ({ error }) => (
  <div className="flex items-center justify-center min-h-[200px]">
    <div className="text-center">
      <div className="text-red-500 text-4xl mb-2">⚠️</div>
      <h3 className="text-lg font-semibold text-red-600 mb-1">Error Loading Dashboard</h3>
      <p className="text-gray-600 text-sm">{error}</p>
    </div>
  </div>
);

export const Dashboard: React.FC = () => {
  const { data, loading, error } = useERPDashboard();

  if (loading) return <LoadingSkeleton />;
  if (error) return <ErrorMessage error={error} />;

  const paidCount = data?.invoices.by_status?.paid?.count || 0;
  const pendingCount = data?.invoices.by_status?.pending?.count || 0;
  const overdueCount = data?.invoices.by_status?.overdue?.count || 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <StatCard
        title="الفواتير"
        value={data?.invoices.total_count || 0}
        subtitle={`${paidCount} مدفوعة | ${pendingCount} معلقة | ${overdueCount} متأخرة`}
      />

      <StatCard
        title="المخزون"
        value={data?.inventory.total_products || 0}
        subtitle={`${data?.inventory.low_stock_count || 0} منخفض التوفر`}
      />

      <StatCard
        title="الموظفين"
        value={data?.employees.total || 0}
        subtitle={`${data?.employees.active || 0} نشط`}
      />

      <StatCard
        title="AI Status"
        value={data?.ai_status.connected ? 'متصل' : 'غير متصل'}
        subtitle={data?.ai_status.model_loaded ? 'النموذج محمل' : 'لا يوجد نموذج'}
        valueClassName={data?.ai_status.connected ? 'text-green-500' : 'text-red-500'}
      />
    </div>
  );
};

export default Dashboard;
