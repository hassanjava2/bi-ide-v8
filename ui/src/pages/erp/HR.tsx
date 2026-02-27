import React, { useState } from 'react';
import { Users, Calendar, DollarSign, UserPlus, Clock } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui';
import { useERP } from '../../hooks/useERP';

interface Employee {
  id: string;
  name: string;
  position: string;
  department: string;
  email: string;
  salary: number;
  hireDate: string;
  status: 'active' | 'on_leave' | 'terminated';
}

interface PayrollRecord {
  id: string;
  employeeName: string;
  period: string;
  baseSalary: number;
  overtime: number;
  deductions: number;
  netPay: number;
}

export const HR: React.FC = () => {
  const { erpData, createEmployee } = useERP();
  const [activeTab, setActiveTab] = useState<'employees' | 'payroll' | 'attendance'>('employees');
  const [showNewEmployee, setShowNewEmployee] = useState(false);
  const [newEmployee, setNewEmployee] = useState({
    name: '',
    position: '',
    department: '',
    email: '',
    salary: 0
  });

  const hrData = erpData;
  const stats = [
    { title: 'Total Employees', value: hrData?.totalEmployees || 0, icon: Users, color: 'blue' },
    { title: 'Active Now', value: hrData?.activeEmployees || 0, icon: Clock, color: 'green' },
    { title: 'On Leave', value: hrData?.onLeave || 0, icon: Calendar, color: 'yellow' },
    { title: 'Monthly Payroll', value: `$${((hrData as any)?.monthlyPayroll || 0).toLocaleString()}`, icon: DollarSign, color: 'purple' }
  ];

  const departments = [
    { name: 'Engineering', count: 12, color: 'bg-blue-100 text-blue-800' },
    { name: 'Sales', count: 8, color: 'bg-green-100 text-green-800' },
    { name: 'Marketing', count: 5, color: 'bg-purple-100 text-purple-800' },
    { name: 'HR', count: 3, color: 'bg-yellow-100 text-yellow-800' },
    { name: 'Finance', count: 4, color: 'bg-red-100 text-red-800' }
  ];

  const handleCreateEmployee = async () => {
    await createEmployee(newEmployee);
    setShowNewEmployee(false);
    setNewEmployee({ name: '', position: '', department: '', email: '', salary: 0 });
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Human Resources</h1>
          <p className="text-gray-600 mt-1">Manage employees, payroll, and attendance</p>
        </div>
        <button
          onClick={() => setShowNewEmployee(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <UserPlus size={20} />
          Add Employee
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <Card key={index}>
            <CardContent className="flex items-center p-6">
              <div className={`p-3 rounded-lg bg-${stat.color}-100`}>
                <stat.icon className={`text-${stat.color}-600`} size={24} />
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-600">{stat.title}</p>
                <p className="text-2xl font-bold">{stat.value}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex gap-8">
          {(['employees', 'payroll', 'attendance'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-4 px-1 border-b-2 font-medium capitalize ${
                activeTab === tab
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'employees' && (
        <div className="space-y-6">
          {/* Departments */}
          <Card>
            <CardHeader>
              <CardTitle>Departments</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-3">
                {departments.map((dept) => (
                  <div key={dept.name} className={`px-4 py-2 rounded-full ${dept.color}`}>
                    {dept.name}: {dept.count}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Employees Table */}
          <Card>
            <CardHeader>
              <CardTitle>Employee Directory</CardTitle>
            </CardHeader>
            <CardContent>
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Name</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Position</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Department</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Email</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Salary</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {hrData?.employees?.map((employee: Employee) => (
                    <tr key={employee.id}>
                      <td className="px-4 py-3 font-medium">{employee.name}</td>
                      <td className="px-4 py-3 text-gray-600">{employee.position}</td>
                      <td className="px-4 py-3">
                        <span className="px-2 py-1 bg-gray-100 rounded text-sm">
                          {employee.department}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-gray-600">{employee.email}</td>
                      <td className="px-4 py-3">${employee.salary.toLocaleString()}</td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-1 rounded text-sm ${
                          employee.status === 'active' ? 'bg-green-100 text-green-800' :
                          employee.status === 'on_leave' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {employee.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === 'payroll' && (
        <Card>
          <CardHeader>
            <CardTitle>Payroll Records</CardTitle>
          </CardHeader>
          <CardContent>
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Employee</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Period</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Base Salary</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Overtime</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Deductions</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Net Pay</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {hrData?.payroll?.map((record: PayrollRecord) => (
                  <tr key={record.id}>
                    <td className="px-4 py-3 font-medium">{record.employeeName}</td>
                    <td className="px-4 py-3 text-gray-600">{record.period}</td>
                    <td className="px-4 py-3">${record.baseSalary.toLocaleString()}</td>
                    <td className="px-4 py-3 text-green-600">+${record.overtime}</td>
                    <td className="px-4 py-3 text-red-600">-${record.deductions}</td>
                    <td className="px-4 py-3 font-bold">${record.netPay.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      {activeTab === 'attendance' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Today's Attendance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-center p-4 bg-green-50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    <span>Present</span>
                  </div>
                  <span className="font-bold">{hrData?.attendance?.present || 0}</span>
                </div>
                <div className="flex justify-between items-center p-4 bg-red-50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <span>Absent</span>
                  </div>
                  <span className="font-bold">{hrData?.attendance?.absent || 0}</span>
                </div>
                <div className="flex justify-between items-center p-4 bg-yellow-50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    <span>Late</span>
                  </div>
                  <span className="font-bold">{hrData?.attendance?.late || 0}</span>
                </div>
                <div className="flex justify-between items-center p-4 bg-blue-50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                    <span>On Leave</span>
                  </div>
                  <span className="font-bold">{hrData?.attendance?.onLeave || 0}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Performance Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">On-Time Rate</span>
                    <span className="text-sm font-bold">95%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{ width: '95%' }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">Leave Utilization</span>
                    <span className="text-sm font-bold">68%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-blue-500 h-2 rounded-full" style={{ width: '68%' }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">Overtime Hours</span>
                    <span className="text-sm font-bold">124h</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-purple-500 h-2 rounded-full" style={{ width: '45%' }}></div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* New Employee Modal */}
      {showNewEmployee && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-xl font-bold mb-4">Add New Employee</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input
                  type="text"
                  value={newEmployee.name}
                  onChange={(e) => setNewEmployee({ ...newEmployee, name: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Position</label>
                <input
                  type="text"
                  value={newEmployee.position}
                  onChange={(e) => setNewEmployee({ ...newEmployee, position: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Department</label>
                <select
                  value={newEmployee.department}
                  onChange={(e) => setNewEmployee({ ...newEmployee, department: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                >
                  <option value="">Select Department</option>
                  {departments.map(d => <option key={d.name} value={d.name}>{d.name}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <input
                  type="email"
                  value={newEmployee.email}
                  onChange={(e) => setNewEmployee({ ...newEmployee, email: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Salary</label>
                <input
                  type="number"
                  value={newEmployee.salary}
                  onChange={(e) => setNewEmployee({ ...newEmployee, salary: Number(e.target.value) })}
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={handleCreateEmployee}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Create
              </button>
              <button
                onClick={() => setShowNewEmployee(false)}
                className="flex-1 px-4 py-2 border rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HR;
