/**
 * useERPDashboard Hook
 * 
 * Fetches and caches ERP data from various /api/v1/erp/* endpoints
 * Features:
 * - Dashboard overview
 * - Invoice management
 * - Inventory tracking
 * - Employee management
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../../services/api'
import type { Invoice, InventoryItem, Employee } from '../../types'

// Query keys
const ERP_KEYS = {
  all: ['erp'] as const,
  dashboard: () => [...ERP_KEYS.all, 'dashboard'] as const,
  invoices: (status?: string) => [...ERP_KEYS.all, 'invoices', { status }] as const,
  inventory: () => [...ERP_KEYS.all, 'inventory'] as const,
  employees: () => [...ERP_KEYS.all, 'employees'] as const,
  payroll: () => [...ERP_KEYS.all, 'payroll'] as const,
  financialReport: (period: string) => [...ERP_KEYS.all, 'financial', period] as const,
  aiInsights: () => [...ERP_KEYS.all, 'ai-insights'] as const,
}

// Types
interface DashboardStats {
  total_sales: number
  total_sales_change: string
  pending_invoices: number
  pending_invoices_change: string
  inventory_value: number
  inventory_value_change: string
  active_employees: number
  active_employees_change: string
}

interface ERPDashboard {
  stats: DashboardStats
  recent_invoices: Invoice[]
  low_stock_items: InventoryItem[]
}

interface InvoicesResponse {
  invoices: Invoice[]
  total: number
}

interface InventoryResponse {
  items: InventoryItem[]
  total_value: number
}

interface EmployeesResponse {
  employees: Employee[]
  total: number
}

interface PayrollResponse {
  entries: Array<{
    employee_id: string
    employee_name: string
    base_salary: number
    bonuses: number
    deductions: number
    net_pay: number
  }>
  total_payroll: number
}

interface FinancialReportResponse {
  period: string
  revenue: number
  expenses: number
  profit: number
  profit_margin: number
}

interface AIInsightsResponse {
  insights: Array<{
    type: 'warning' | 'opportunity' | 'info'
    title: string
    description: string
    actionable: boolean
    suggested_action?: string
  }>
}

interface CreateInvoiceRequest {
  customer_id: string
  customer_name: string
  amount: number
  due_date: string
  items?: Array<{
    description: string
    quantity: number
    unit_price: number
  }>
}

// Fetch functions
const fetchERPDashboard = async (): Promise<ERPDashboard> => {
  return api.getERPDashboard()
}

const fetchInvoices = async (status?: string): Promise<InvoicesResponse> => {
  return api.getInvoices(status)
}

const fetchInventory = async (): Promise<InventoryResponse> => {
  return api.getInventory()
}

const fetchEmployees = async (): Promise<EmployeesResponse> => {
  return api.getEmployees()
}

const fetchPayroll = async (): Promise<PayrollResponse> => {
  return api.getPayroll()
}

const fetchFinancialReport = async (period: string): Promise<FinancialReportResponse> => {
  return api.getFinancialReport(period)
}

const fetchAIInsights = async (): Promise<AIInsightsResponse> => {
  return api.getERP_AI_Insights()
}

// Mutation functions
const createInvoice = async (data: CreateInvoiceRequest): Promise<Invoice> => {
  return api.createInvoice(data)
}

const markInvoiceAsPaid = async (invoiceId: string): Promise<void> => {
  return api.markInvoicePaid(invoiceId)
}

/**
 * Hook for fetching ERP dashboard overview
 * 
 * @example
 * ```tsx
 * const { data, isLoading } = useERPDashboard()
 * ```
 */
export function useERPDashboard(options: { enabled?: boolean } = {}) {
  return useQuery<ERPDashboard>({
    queryKey: ERP_KEYS.dashboard(),
    queryFn: fetchERPDashboard,
    staleTime: 1000 * 60, // 1 minute
    enabled: options.enabled ?? true,
  })
}

interface UseInvoicesOptions {
  status?: 'pending' | 'paid' | 'overdue'
  enabled?: boolean
}

/**
 * Hook for fetching invoices with optional status filter
 * 
 * @example
 * ```tsx
 * // All invoices
 * const { data } = useInvoices()
 * 
 * // Pending invoices only
 * const { data } = useInvoices({ status: 'pending' })
 * ```
 */
export function useInvoices(options: UseInvoicesOptions = {}) {
  const { status, enabled = true } = options

  return useQuery<InvoicesResponse>({
    queryKey: ERP_KEYS.invoices(status),
    queryFn: () => fetchInvoices(status),
    staleTime: 1000 * 30,
    enabled,
  })
}

/**
 * Hook for fetching inventory
 * 
 * @example
 * ```tsx
 * const { data, isLoading } = useInventory()
 * ```
 */
export function useInventory(options: { enabled?: boolean } = {}) {
  return useQuery<InventoryResponse>({
    queryKey: ERP_KEYS.inventory(),
    queryFn: fetchInventory,
    staleTime: 1000 * 60,
    enabled: options.enabled ?? true,
  })
}

/**
 * Hook for fetching employees
 * 
 * @example
 * ```tsx
 * const { data, isLoading } = useEmployees()
 * ```
 */
export function useEmployees(options: { enabled?: boolean } = {}) {
  return useQuery<EmployeesResponse>({
    queryKey: ERP_KEYS.employees(),
    queryFn: fetchEmployees,
    staleTime: 1000 * 60 * 5, // 5 minutes
    enabled: options.enabled ?? true,
  })
}

/**
 * Hook for fetching payroll data
 * 
 * @example
 * ```tsx
 * const { data, isLoading } = usePayroll()
 * ```
 */
export function usePayroll(options: { enabled?: boolean } = {}) {
  return useQuery<PayrollResponse>({
    queryKey: ERP_KEYS.payroll(),
    queryFn: fetchPayroll,
    staleTime: 1000 * 60 * 5,
    enabled: options.enabled ?? true,
  })
}

/**
 * Hook for fetching financial reports
 * 
 * @example
 * ```tsx
 * const { data } = useFinancialReport({ period: 'month' })
 * ```
 */
export function useFinancialReport(period: string = 'month', options: { enabled?: boolean } = {}) {
  return useQuery<FinancialReportResponse>({
    queryKey: ERP_KEYS.financialReport(period),
    queryFn: () => fetchFinancialReport(period),
    staleTime: 1000 * 60 * 5,
    enabled: options.enabled ?? true,
  })
}

/**
 * Hook for fetching AI insights
 * 
 * @example
 * ```tsx
 * const { data, isLoading } = useAIInsights()
 * ```
 */
export function useAIInsights(options: { enabled?: boolean } = {}) {
  return useQuery<AIInsightsResponse>({
    queryKey: ERP_KEYS.aiInsights(),
    queryFn: fetchAIInsights,
    staleTime: 1000 * 60 * 5,
    enabled: options.enabled ?? true,
  })
}

/**
 * Hook for creating a new invoice
 * 
 * @example
 * ```tsx
 * const mutation = useCreateInvoice()
 * 
 * mutation.mutate({
 *   customer_id: 'C001',
 *   customer_name: 'Acme Corp',
 *   amount: 5000,
 *   due_date: '2024-12-31'
 * })
 * ```
 */
export function useCreateInvoice() {
  const queryClient = useQueryClient()

  return useMutation<Invoice, Error, CreateInvoiceRequest>({
    mutationFn: createInvoice,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ERP_KEYS.invoices() })
      queryClient.invalidateQueries({ queryKey: ERP_KEYS.dashboard() })
    },
  })
}

/**
 * Hook for marking an invoice as paid
 * 
 * @example
 * ```tsx
 * const mutation = useMarkInvoicePaid()
 * 
 * mutation.mutate('INV-001')
 * ```
 */
export function useMarkInvoicePaid() {
  const queryClient = useQueryClient()

  return useMutation<void, Error, string>({
    mutationFn: markInvoiceAsPaid,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ERP_KEYS.invoices() })
      queryClient.invalidateQueries({ queryKey: ERP_KEYS.dashboard() })
    },
  })
}

/**
 * Hook for refreshing all ERP data
 * 
 * @example
 * ```tsx
 * const { refreshAll, isRefreshing } = useRefreshERP()
 * ```
 */
export function useRefreshERP() {
  const queryClient = useQueryClient()

  const refreshAll = async () => {
    await queryClient.invalidateQueries({ queryKey: ERP_KEYS.all })
  }

  const isRefreshing = queryClient.isFetching({ queryKey: ERP_KEYS.all }) > 0

  return { refreshAll, isRefreshing }
}

export { ERP_KEYS }
export type {
  DashboardStats,
  ERPDashboard,
  InvoicesResponse,
  InventoryResponse,
  EmployeesResponse,
  PayrollResponse,
  FinancialReportResponse,
  AIInsightsResponse,
  CreateInvoiceRequest,
}
export default useERPDashboard
