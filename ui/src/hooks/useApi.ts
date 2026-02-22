import { useState, useEffect, useCallback } from 'react'
import { api } from '../services/api'
import { useLiveData } from './useLiveData'

// Hook that uses WebSocket for live data instead of polling
export function useSystemStatus() {
  const { systemStatus, isConnected, isFallback, error: wsError, refresh, reconnect } = useLiveData()
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Set loading to false once we have initial data
    if (systemStatus !== null) {
      setLoading(false)
    }
  }, [systemStatus])

  // Manual refetch - uses API if WebSocket is not connected
  const refetch = useCallback(async () => {
    setLoading(true)
    await refresh()
    setLoading(false)
  }, [refresh])

  return { 
    status: systemStatus, 
    loading, 
    error: wsError, 
    refetch,
    isConnected,
    isFallback,
    reconnect
  }
}

export function useCommand() {
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<any>(null)

  const execute = useCallback(async (command: string, alertLevel: string = 'GREEN') => {
    try {
      setLoading(true)
      const data = await api.executeCommand(command, alertLevel)
      setResult(data)
      return data
    } catch (err: any) {
      setError(err)
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  return { execute, result, loading, error }
}

export function useInvoices() {
  const [invoices, setInvoices] = useState([])
  const [loading, setLoading] = useState(true)

  const fetchInvoices = useCallback(async () => {
    const data = await api.getInvoices()
    setInvoices(data.invoices || [])
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchInvoices()
  }, [fetchInvoices])

  return { invoices, loading, refetch: fetchInvoices }
}

export function useInventory(lowStock: boolean = false) {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)

  const fetchInventory = useCallback(async () => {
    const data = await api.getInventory()
    setItems(data.items || [])
    setLoading(false)
  }, [lowStock])

  useEffect(() => {
    fetchInventory()
  }, [fetchInventory])

  return { items, loading, refetch: fetchInventory }
}

export function usePosts() {
  // Community API not yet implemented - return empty
  return { posts: [], loading: false, refetch: () => {}, createPost: () => {} }
}
