import React from 'react';
import { useEmployees } from '@/hooks/useERP-legacy';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { Users, RefreshCw, Briefcase, Mail } from 'lucide-react';

export const EmployeeManager: React.FC = () => {
  const { employees, loading, error, refresh } = useEmployees();

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
        <h2 className="text-xl font-semibold">الموظفين</h2>
        <Button onClick={refresh} variant="outline" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          تحديث
        </Button>
      </div>

      {employees.length === 0 ? (
        <div className="text-center p-8 text-gray-500">
          <Users className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>لا يوجد موظفين</p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {employees.map((employee) => (
            <Card key={employee.id}>
              <CardContent className="p-4">
                <div className="flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-green-100 text-green-600">
                    <Users className="w-5 h-5" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold">
                      {employee.first_name} {employee.last_name}
                    </h3>
                    <div className="flex items-center gap-2 text-sm text-gray-500 mt-1">
                      <Briefcase className="w-4 h-4" />
                      <span>{employee.department}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <Mail className="w-4 h-4" />
                      <span>{employee.email}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className={`inline-block px-2 py-1 text-xs rounded-full ${
                      employee.is_active ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'
                    }`}>
                      {employee.is_active ? 'نشط' : 'غير نشط'}
                    </span>
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

export default EmployeeManager;
