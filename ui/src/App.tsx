import { Routes, Route, Navigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import Layout from './components/Layout'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Council from './pages/Council'
import ERP from './pages/ERP'
import Community from './pages/Community'
import IDE from './pages/IDE'
import Training from './pages/Training'
import Settings from './pages/Settings'
import MetaControl from './pages/MetaControl'
import { ProtectedRoute } from './components/Auth/ProtectedRoute'
import { useAuth } from './hooks/useAuth'
import { useLiveData } from './hooks/useLiveData'

function App() {
  const { isAuthenticated, loading, logout } = useAuth()
  const [isAppReady, setIsAppReady] = useState(false)

  // Use live data hook for system status via WebSocket
  const { systemStatus, isConnected, isFallback } = useLiveData()

  useEffect(() => {
    // Wait for auth check to complete
    if (!loading) {
      setIsAppReady(true)
    }
  }, [loading])

  if (!isAppReady) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-bi-accent border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">جاري تحميل النظام...</p>
        </div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return <Login onLogin={() => {}} />
  }

  return (
    <Layout 
      systemStatus={systemStatus} 
      connectionStatus={{ isConnected, isFallback }}
      onLogout={logout}
    >
      <Routes>
        <Route path="/" element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        } />
        <Route path="/council" element={
          <ProtectedRoute>
            <Council />
          </ProtectedRoute>
        } />
        <Route path="/erp" element={
          <ProtectedRoute>
            <ERP />
          </ProtectedRoute>
        } />
        <Route path="/community" element={
          <ProtectedRoute>
            <Community />
          </ProtectedRoute>
        } />
        <Route path="/ide" element={
          <ProtectedRoute>
            <IDE />
          </ProtectedRoute>
        } />
        <Route path="/training" element={
          <ProtectedRoute>
            <Training />
          </ProtectedRoute>
        } />
        <Route path="/settings" element={
          <ProtectedRoute>
            <Settings />
          </ProtectedRoute>
        } />
        <Route path="/meta" element={
          <ProtectedRoute requiredRole="admin">
            <MetaControl />
          </ProtectedRoute>
        } />
        <Route path="/login" element={<Navigate to="/" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}

export default App
