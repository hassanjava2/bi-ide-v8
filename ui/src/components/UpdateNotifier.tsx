//! Update notification component
import { useState } from 'react'
import { useAutoUpdate, useWebUpdateCheck } from '../hooks'
import { Download, RefreshCw, X, Check } from 'lucide-react'

export function UpdateNotifier() {
  const tauriUpdate = useAutoUpdate()
  const webUpdate = useWebUpdateCheck()
  const [dismissed, setDismissed] = useState(false)

  // Tauri update available
  if (tauriUpdate.isTauri && tauriUpdate.updateInfo && !dismissed) {
    return (
      <div className="fixed bottom-4 right-4 z-50">
        <div className="bg-gray-800 border border-blue-500/30 rounded-lg shadow-lg p-4 max-w-sm">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 bg-blue-500/20 rounded-full flex items-center justify-center flex-shrink-0">
              <Download className="w-5 h-5 text-blue-500" />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-white mb-1">
                Update Available
              </h3>
              <p className="text-sm text-gray-400 mb-2">
                Version {tauriUpdate.updateInfo.version} is ready to download
              </p>
              
              {tauriUpdate.isDownloading ? (
                <div className="space-y-2">
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-500 transition-all"
                      style={{ width: `${tauriUpdate.progress}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-500">
                    Downloading... {tauriUpdate.progress.toFixed(0)}%
                  </p>
                </div>
              ) : tauriUpdate.isReady ? (
                <div className="flex gap-2">
                  <button
                    onClick={tauriUpdate.installUpdate}
                    className="flex-1 bg-green-600 hover:bg-green-700 text-white text-sm py-2 px-3 rounded flex items-center justify-center gap-2 transition-colors"
                  >
                    <Check className="w-4 h-4" />
                    Install & Restart
                  </button>
                </div>
              ) : (
                <div className="flex gap-2">
                  <button
                    onClick={tauriUpdate.downloadUpdate}
                    className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm py-2 px-3 rounded transition-colors"
                  >
                    Download
                  </button>
                  <button
                    onClick={() => setDismissed(true)}
                    className="bg-gray-700 hover:bg-gray-600 text-white text-sm py-2 px-3 rounded transition-colors"
                  >
                    Later
                  </button>
                </div>
              )}
              
              {tauriUpdate.error && (
                <p className="text-xs text-red-400 mt-2">
                  Error: {tauriUpdate.error}
                </p>
              )}
            </div>
            <button
              onClick={() => setDismissed(true)}
              className="text-gray-500 hover:text-white transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Web update available
  if (!tauriUpdate.isTauri && webUpdate.hasUpdate && !dismissed) {
    return (
      <div className="fixed bottom-4 right-4 z-50">
        <div className="bg-gray-800 border border-green-500/30 rounded-lg shadow-lg p-4 max-w-sm">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 bg-green-500/20 rounded-full flex items-center justify-center flex-shrink-0">
              <RefreshCw className="w-5 h-5 text-green-500" />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-white mb-1">
                New Version Available
              </h3>
              <p className="text-sm text-gray-400 mb-3">
                A new version of the application is available. Reload to update.
              </p>
              <div className="flex gap-2">
                <button
                  onClick={webUpdate.reload}
                  className="flex-1 bg-green-600 hover:bg-green-700 text-white text-sm py-2 px-3 rounded transition-colors"
                >
                  Reload Now
                </button>
                <button
                  onClick={() => setDismissed(true)}
                  className="bg-gray-700 hover:bg-gray-600 text-white text-sm py-2 px-3 rounded transition-colors"
                >
                  Later
                </button>
              </div>
            </div>
            <button
              onClick={() => setDismissed(true)}
              className="text-gray-500 hover:text-white transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    )
  }

  return null
}

// Update check button for settings
export function UpdateCheckButton() {
  const { checkForUpdates, isChecking, isTauri } = useAutoUpdate()

  if (!isTauri) return null

  return (
    <button
      onClick={checkForUpdates}
      disabled={isChecking}
      className="flex items-center gap-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 text-white py-2 px-4 rounded transition-colors"
    >
      <RefreshCw className={`w-4 h-4 ${isChecking ? 'animate-spin' : ''}`} />
      {isChecking ? 'Checking...' : 'Check for Updates'}
    </button>
  )
}
