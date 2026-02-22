import { ReactNode } from 'react'
import Sidebar from './Sidebar'
import Header from './Header'
import type { SystemStatus } from '../types'

interface ConnectionStatus {
  isConnected: boolean
  isFallback: boolean
}

interface LayoutProps {
  children: ReactNode
  systemStatus: SystemStatus | null
  connectionStatus?: ConnectionStatus
  onLogout: () => void
}

export default function Layout({ children, systemStatus, connectionStatus, onLogout }: LayoutProps) {
  return (
    <div className="min-h-screen flex">
      <Sidebar />
      <div className="flex-1 flex flex-col mr-64">
        <Header 
          systemStatus={systemStatus} 
          connectionStatus={connectionStatus}
          onLogout={onLogout} 
        />
        <main className="flex-1 p-6 overflow-auto scrollbar-thin">
          {children}
        </main>
      </div>
    </div>
  )
}
