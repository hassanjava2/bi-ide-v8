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
import { isAuthenticated as checkAuth, clearAuth } from './services/api'
import { useLiveData } from './hooks/useLiveData'

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  // Use live data hook for system status via WebSocket
  const { systemStatus, isConnected, isFallback } = useLiveData()

  useEffect(() => {
    // التحقق من تسجيل الدخول عبر JWT
    setIsLoggedIn(checkAuth())
    setIsLoading(false)
  }, [])

  const handleLogin = () => {
    // Token is already stored by the login API call
    setIsLoggedIn(true)
  }

  const handleLogout = () => {
    clearAuth()
    setIsLoggedIn(false)
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-bi-accent border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-400">جاري تحميل النظام...</p>
        </div>
      </div>
    )
  }

  if (!isLoggedIn) {
    return <Login onLogin={handleLogin} />
  }

  return (
    <Layout 
      systemStatus={systemStatus} 
      connectionStatus={{ isConnected, isFallback }}
      onLogout={handleLogout}
    >
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/council" element={<Council />} />
        <Route path="/erp" element={<ERP />} />
        <Route path="/community" element={<Community />} />
        <Route path="/ide" element={<IDE />} />
        <Route path="/training" element={<Training />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/meta" element={<MetaControl />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}

export default App
