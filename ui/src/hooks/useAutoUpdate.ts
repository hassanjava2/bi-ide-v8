//! Auto-update hook for Tauri app
import { useState, useEffect, useCallback } from 'react'

export interface UpdateInfo {
  version: string
  releaseNotes: string
  date: string
}

interface AutoUpdateState {
  isChecking: boolean
  isDownloading: boolean
  isReady: boolean
  updateInfo: UpdateInfo | null
  error: string | null
  progress: number
}

// Check if running in Tauri
const isTauri = typeof window !== 'undefined' && (window as any).__TAURI__ !== undefined

// Type definitions for Tauri APIs (avoids import errors)
interface TauriUpdateManifest {
  version: string
  body?: string
  date?: string
}

export function useAutoUpdate() {
  const [state, setState] = useState<AutoUpdateState>({
    isChecking: false,
    isDownloading: false,
    isReady: false,
    updateInfo: null,
    error: null,
    progress: 0
  })

  // Check for updates
  const checkForUpdates = useCallback(async () => {
    if (!isTauri) {
      console.log('[AutoUpdate] Not in Tauri environment')
      return
    }

    setState(prev => ({ ...prev, isChecking: true, error: null }))

    try {
      // Dynamically import Tauri API (only works in Tauri)
      const tauriUpdater = await eval("import('@tauri-apps/api/updater')").catch(() => null)
      if (!tauriUpdater) {
        throw new Error('Tauri updater not available')
      }
      const { checkUpdate } = tauriUpdater
      const result = await checkUpdate()
      const { shouldUpdate, manifest } = result as { shouldUpdate: boolean; manifest?: TauriUpdateManifest }

      if (shouldUpdate && manifest) {
        setState(prev => ({
          ...prev,
          isChecking: false,
          updateInfo: {
            version: manifest.version,
            releaseNotes: manifest.body || 'No release notes available',
            date: manifest.date || new Date().toISOString()
          }
        }))
      } else {
        setState(prev => ({ ...prev, isChecking: false }))
      }
    } catch (error) {
      console.error('[AutoUpdate] Check failed:', error)
      setState(prev => ({
        ...prev,
        isChecking: false,
        error: error instanceof Error ? error.message : 'Update check failed'
      }))
    }
  }, [])

  // Download update
  const downloadUpdate = useCallback(async () => {
    if (!isTauri) return

    setState(prev => ({ ...prev, isDownloading: true, error: null }))

    try {
      // Dynamically import Tauri API
      const tauriUpdater = await eval("import('@tauri-apps/api/updater')").catch(() => null)
      if (!tauriUpdater) {
        throw new Error('Tauri updater not available')
      }
      const { onUpdaterEvent, installUpdate } = tauriUpdater
      
      const unlisten = await onUpdaterEvent(({ event, data }: any) => {
        switch (event) {
          case 'DOWNLOAD_PROGRESS':
            setState(prev => ({ ...prev, progress: data }))
            break
          case 'DOWNLOAD_FINISHED':
            setState(prev => ({
              ...prev,
              isDownloading: false,
              isReady: true,
              progress: 100
            }))
            unlisten()
            break
          case 'ERROR':
            setState(prev => ({
              ...prev,
              isDownloading: false,
              error: data
            }))
            unlisten()
            break
        }
      })

      await installUpdate()
    } catch (error) {
      console.error('[AutoUpdate] Download failed:', error)
      setState(prev => ({
        ...prev,
        isDownloading: false,
        error: error instanceof Error ? error.message : 'Download failed'
      }))
    }
  }, [])

  // Install update (relaunch app)
  const installUpdate = useCallback(async () => {
    if (!isTauri) return

    try {
      const tauriProcess = await eval("import('@tauri-apps/api/process')").catch(() => null)
      if (!tauriProcess) {
        throw new Error('Tauri process API not available')
      }
      const { relaunch } = tauriProcess
      await relaunch()
    } catch (error) {
      console.error('[AutoUpdate] Relaunch failed:', error)
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Relaunch failed'
      }))
    }
  }, [])

  // Dismiss update
  const dismissUpdate = useCallback(() => {
    setState(prev => ({
      ...prev,
      updateInfo: null,
      isReady: false,
      error: null
    }))
  }, [])

  // Check on mount (if in Tauri)
  useEffect(() => {
    if (isTauri) {
      // Delay check to not interfere with startup
      const timer = setTimeout(() => {
        checkForUpdates()
      }, 10000)

      return () => clearTimeout(timer)
    }
  }, [checkForUpdates])

  return {
    ...state,
    checkForUpdates,
    downloadUpdate,
    installUpdate,
    dismissUpdate,
    isTauri
  }
}

// Simple update checker for web version
export function useWebUpdateCheck() {
  const [hasUpdate, setHasUpdate] = useState(false)

  useEffect(() => {
    // Check for new version every 5 minutes
    const checkVersion = async () => {
      try {
        const response = await fetch('/api/version')
        if (response.ok) {
          const { version } = await response.json()
          const currentVersion = localStorage.getItem('app_version')
          
          if (currentVersion && currentVersion !== version) {
            setHasUpdate(true)
          }
          localStorage.setItem('app_version', version)
        }
      } catch {
        // Silent fail
      }
    }

    checkVersion()
    const interval = setInterval(checkVersion, 5 * 60 * 1000)

    return () => clearInterval(interval)
  }, [])

  const dismiss = useCallback(() => {
    setHasUpdate(false)
  }, [])

  const reload = useCallback(() => {
    window.location.reload()
  }, [])

  return { hasUpdate, dismiss, reload }
}
