import { useState, useCallback } from 'react';

// Types
export interface Employee {
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

export interface Customer {
  id: string;
  name: string;
  company: string;
  email: string;
  phone: string;
  status: 'lead' | 'prospect' | 'customer' | 'churned';
  ltv: number;
  lastContact: string;
  rating: number;
}

export interface Deal {
  id: string;
  customerName: string;
  title: string;
  value: number;
  stage: string;
  probability: number;
  expectedClose: string;
}

export interface Product {
  id: string;
  name: string;
  sku: string;
  category: string;
  quantity: number;
  reorderPoint: number;
  reorder_level?: number;  // alias for backward compatibility
  unitPrice: number;
  warehouse: string;
  status: 'in_stock' | 'low_stock' | 'out_of_stock';
}

interface ERPData {
  employees?: Employee[];
  payroll?: PayrollRecord[];
  customers?: Customer[];
  deals?: Deal[];
  products?: Product[];
  totalEmployees?: number;
  activeEmployees?: number;
  onLeave?: number;
  totalCustomers?: number;
  activeDeals?: number;
  pipelineValue?: number;
  avgLTV?: number;
  totalProducts?: number;
  inStock?: number;
  lowStock?: number;
  outOfStock?: number;
  attendance?: {
    present: number;
    absent: number;
    late: number;
    onLeave: number;
  };
}

export const useERP = () => {
  const [erpData, setErpData] = useState<ERPData>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // HR
  const fetchHRData = useCallback(async () => {
    setLoading(true);
    try {
      // Mock data for now - replace with actual API calls
      const mockEmployees: Employee[] = [
        { id: '1', name: 'John Doe', position: 'Software Engineer', department: 'Engineering', email: 'john@company.com', salary: 85000, hireDate: '2023-01-15', status: 'active' },
        { id: '2', name: 'Jane Smith', position: 'Product Manager', department: 'Product', email: 'jane@company.com', salary: 95000, hireDate: '2022-08-01', status: 'active' },
        { id: '3', name: 'Bob Johnson', position: 'Designer', department: 'Design', email: 'bob@company.com', salary: 75000, hireDate: '2023-03-10', status: 'on_leave' }
      ];
      
      const mockPayroll: PayrollRecord[] = [
        { id: '1', employeeName: 'John Doe', period: '2024-01', baseSalary: 7083, overtime: 500, deductions: 1200, netPay: 6383 },
        { id: '2', employeeName: 'Jane Smith', period: '2024-01', baseSalary: 7917, overtime: 0, deductions: 1583, netPay: 6334 }
      ];

      setErpData(prev => ({
        ...prev,
        employees: mockEmployees,
        payroll: mockPayroll,
        totalEmployees: mockEmployees.length,
        activeEmployees: mockEmployees.filter(e => e.status === 'active').length,
        onLeave: mockEmployees.filter(e => e.status === 'on_leave').length,
        monthlyPayroll: mockPayroll.reduce((sum, p) => sum + p.netPay, 0),
        attendance: { present: 28, absent: 2, late: 3, onLeave: 2 }
      }));
    } catch (err) {
      setError('Failed to fetch HR data');
    } finally {
      setLoading(false);
    }
  }, []);

  const createEmployee = useCallback(async (data: any) => {
    try {
      const newEmployee: Employee = {
        id: Date.now().toString(),
        ...data,
        hireDate: new Date().toISOString().split('T')[0],
        status: 'active'
      };
      setErpData(prev => ({
        ...prev,
        employees: [...(prev.employees || []), newEmployee]
      }));
      return newEmployee;
    } catch (err) {
      setError('Failed to create employee');
      throw err;
    }
  }, []);

  // CRM
  const fetchCRMData = useCallback(async () => {
    setLoading(true);
    try {
      const mockCustomers: Customer[] = [
        { id: '1', name: 'Acme Corp', company: 'Acme Inc', email: 'contact@acme.com', phone: '+1-555-0123', status: 'customer', ltv: 125000, lastContact: '2024-01-15', rating: 5 },
        { id: '2', name: 'TechStart', company: 'TechStart LLC', email: 'info@techstart.com', phone: '+1-555-0456', status: 'prospect', ltv: 45000, lastContact: '2024-01-10', rating: 4 },
        { id: '3', name: 'GlobalBiz', company: 'GlobalBiz Co', email: 'sales@globalbiz.com', phone: '+1-555-0789', status: 'lead', ltv: 0, lastContact: '2024-01-20', rating: 3 }
      ];

      const mockDeals: Deal[] = [
        { id: '1', customerName: 'Acme Corp', title: 'Enterprise License', value: 50000, stage: 'negotiation', probability: 80, expectedClose: '2024-02-15' },
        { id: '2', customerName: 'TechStart', title: 'Starter Package', value: 15000, stage: 'proposal', probability: 60, expectedClose: '2024-02-01' },
        { id: '3', customerName: 'GlobalBiz', title: 'Consulting Project', value: 25000, stage: 'qualified', probability: 40, expectedClose: '2024-03-01' }
      ];

      setErpData(prev => ({
        ...prev,
        customers: mockCustomers,
        deals: mockDeals,
        totalCustomers: mockCustomers.length,
        activeDeals: mockDeals.filter(d => !['closed_won', 'closed_lost'].includes(d.stage)).length,
        pipelineValue: mockDeals.reduce((sum, d) => sum + d.value, 0),
        avgLTV: mockCustomers.reduce((sum, c) => sum + c.ltv, 0) / mockCustomers.length
      }));
    } catch (err) {
      setError('Failed to fetch CRM data');
    } finally {
      setLoading(false);
    }
  }, []);

  const createCustomer = useCallback(async (data: any) => {
    try {
      const newCustomer: Customer = {
        id: Date.now().toString(),
        ...data,
        ltv: 0,
        lastContact: new Date().toISOString().split('T')[0],
        rating: 0
      };
      setErpData(prev => ({
        ...prev,
        customers: [...(prev.customers || []), newCustomer]
      }));
      return newCustomer;
    } catch (err) {
      setError('Failed to create customer');
      throw err;
    }
  }, []);

  // Inventory
  const fetchInventory = useCallback(async () => {
    setLoading(true);
    try {
      const mockProducts: Product[] = [
        { id: '1', name: 'Laptop Dell XPS 15', sku: 'LAP-DEL-001', category: 'Electronics', quantity: 45, reorderPoint: 10, unitPrice: 1299, warehouse: 'Main Warehouse', status: 'in_stock' },
        { id: '2', name: 'Monitor 27" 4K', sku: 'MON-4K-001', category: 'Electronics', quantity: 8, reorderPoint: 15, unitPrice: 499, warehouse: 'Main Warehouse', status: 'low_stock' },
        { id: '3', name: 'Wireless Mouse', sku: 'ACC-MOU-001', category: 'Accessories', quantity: 0, reorderPoint: 20, unitPrice: 29, warehouse: 'East Branch', status: 'out_of_stock' }
      ];

      setErpData(prev => ({
        ...prev,
        products: mockProducts,
        totalProducts: mockProducts.length,
        inStock: mockProducts.filter(p => p.status === 'in_stock').length,
        lowStock: mockProducts.filter(p => p.status === 'low_stock').length,
        outOfStock: mockProducts.filter(p => p.status === 'out_of_stock').length
      }));
    } catch (err) {
      setError('Failed to fetch inventory');
    } finally {
      setLoading(false);
    }
  }, []);

  const createProduct = useCallback(async (data: any) => {
    try {
      const newProduct: Product = {
        id: Date.now().toString(),
        ...data,
        status: data.quantity === 0 ? 'out_of_stock' : data.quantity <= data.reorderPoint ? 'low_stock' : 'in_stock'
      };
      setErpData(prev => ({
        ...prev,
        products: [...(prev.products || []), newProduct]
      }));
      return newProduct;
    } catch (err) {
      setError('Failed to create product');
      throw err;
    }
  }, []);

  return {
    erpData,
    loading,
    error,
    fetchHRData,
    createEmployee,
    fetchCRMData,
    createCustomer,
    fetchInventory,
    createProduct
  };
};

export default useERP;
