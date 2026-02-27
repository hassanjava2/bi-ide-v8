// Legacy exports for backward compatibility
import { useERP as useERPHook } from './useERP';

// Re-export the main hook
export const useERP = useERPHook;

// Stub hooks that use the main hook internally
export function useERPDashboard() {
  const { erpData, loading, error } = useERPHook();
  return {
    data: (erpData as any)?.dashboard,
    loading,
    error,
    refresh: () => {}
  };
}

export function useInvoices(_status?: string) {
  return {
    invoices: [] as any[],
    loading: false,
    error: null,
    markAsPaid: async (_id: string) => {},
    refresh: () => {},
    fetchInvoices: async () => {},
    createInvoice: async () => ({}),
    updateInvoice: async () => ({}),
    deleteInvoice: async () => {}
  };
}

export function useInventory() {
  const { erpData, loading, error, fetchInventory, createProduct } = useERPHook();
  return {
    items: erpData?.products || [],
    loading,
    error,
    lowStock: (erpData?.products || []).filter((p: any) => p.status === 'low_stock'),
    adjustStock: async (_productId: string, _qty: number, _reason: string) => {},
    refresh: fetchInventory,
    fetchInventory,
    createItem: createProduct,
    updateItem: async () => ({}),
    deleteItem: async () => {}
  };
}

export function useEmployees() {
  const { erpData, loading, error, fetchHRData, createEmployee } = useERPHook();
  return {
    employees: (erpData?.employees || []).map((e: any) => ({
      ...e,
      first_name: e.name?.split(' ')[0] || '',
      last_name: e.name?.split(' ')[1] || '',
      is_active: e.status === 'active'
    })),
    loading,
    error,
    refresh: fetchHRData,
    fetchEmployees: fetchHRData,
    createEmployee,
    updateEmployee: async () => ({}),
    deleteEmployee: async () => {}
  };
}

export function useAccounts() {
  return {
    accounts: [],
    loading: false,
    error: null,
    fetchAccounts: async () => {},
    createAccount: async () => ({}),
    updateAccount: async () => ({}),
    deleteAccount: async () => {}
  };
}

// Types
export interface DashboardData {
  totalRevenue: number;
  activeCustomers: number;
  pendingInvoices: number;
  lowStockItems: number;
}

export interface Invoice {
  id: string;
  number: string;
  customer: string;
  amount: number;
  status: string;
  date: string;
}

export interface InventoryItem {
  id: string;
  name: string;
  sku: string;
  quantity: number;
  price: number;
  reorder_level?: number;
  reorderPoint?: number;
}

export interface Employee {
  id: string;
  name: string;
  first_name?: string;
  last_name?: string;
  email: string;
  position: string;
  department: string;
  is_active?: boolean;
}

export interface Account {
  id: string;
  name: string;
  code: string;
  type: string;
  balance: number;
}
