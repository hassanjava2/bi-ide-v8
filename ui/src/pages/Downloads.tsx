import { useEffect, useMemo, useState } from 'react'
import { Download, Laptop, MonitorSmartphone, Server } from 'lucide-react'
import { downloadInstaller, getInstallers, type InstallerItem } from '../services/api'

const platformNames: Record<string, string> = {
  windows: 'Windows',
  macos: 'macOS',
  linux: 'Linux',
  other: 'Other',
}

function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let size = bytes
  let idx = 0
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024
    idx += 1
  }
  return `${size.toFixed(size >= 100 ? 0 : 1)} ${units[idx]}`
}

function detectClientPlatform(): 'windows' | 'macos' | 'linux' | 'other' {
  const platform = navigator.platform.toLowerCase()
  if (platform.includes('win')) return 'windows'
  if (platform.includes('mac')) return 'macos'
  if (platform.includes('linux')) return 'linux'
  return 'other'
}

export default function Downloads() {
  const [items, setItems] = useState<InstallerItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [downloadingId, setDownloadingId] = useState<string | null>(null)

  const detectedPlatform = useMemo(() => detectClientPlatform(), [])

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      setError('')
      try {
        const data = await getInstallers()
        setItems(data.items || [])
      } catch (err: any) {
        setError(err?.message || 'فشل جلب ملفات التنصيب')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const grouped = useMemo(() => {
    return {
      windows: items.filter((x) => x.platform === 'windows'),
      macos: items.filter((x) => x.platform === 'macos'),
      linux: items.filter((x) => x.platform === 'linux'),
      other: items.filter((x) => x.platform === 'other'),
    }
  }, [items])

  const onDownload = async (item: InstallerItem) => {
    try {
      setDownloadingId(item.id)
      await downloadInstaller(item.id, item.name)
    } catch (err: any) {
      setError(err?.message || 'فشل تنزيل الملف')
    } finally {
      setDownloadingId(null)
    }
  }

  const sections = [
    { key: 'windows', icon: MonitorSmartphone, title: 'Windows', files: grouped.windows },
    { key: 'macos', icon: Laptop, title: 'macOS', files: grouped.macos },
    { key: 'linux', icon: Server, title: 'Linux', files: grouped.linux },
  ] as const

  return (
    <div className="space-y-6">
      <div className="glass-panel p-6">
        <h1 className="text-2xl font-bold text-white mb-2">تنزيل برنامج BI IDE Desktop</h1>
        <p className="text-gray-400">بعد تسجيل الدخول، اختر النظام ونزّل نسخة التثبيت الكاملة.</p>
        <p className="text-sm text-bi-accent mt-2">النظام المكتشف على جهازك: {platformNames[detectedPlatform]}</p>
      </div>

      {error && (
        <div className="glass-panel p-4 border border-red-500/30 text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <div className="glass-panel p-6 text-gray-300">جاري تحميل ملفات التنصيب...</div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {sections.map(({ key, icon: Icon, title, files }) => (
            <div key={key} className="glass-panel p-5">
              <div className="flex items-center gap-2 mb-4">
                <Icon className="w-5 h-5 text-bi-accent" />
                <h2 className="text-lg font-semibold text-white">{title}</h2>
              </div>

              {files.length === 0 ? (
                <p className="text-sm text-gray-400">لا توجد نسخة متاحة حالياً.</p>
              ) : (
                <div className="space-y-3">
                  {files.map((file) => (
                    <div key={file.id} className="bg-white/5 rounded-lg p-3">
                      <div className="text-sm text-white break-all">{file.name}</div>
                      <div className="text-xs text-gray-400 mt-1">
                        الإصدار {file.version} • {formatBytes(file.size_bytes)} {file.arch ? `• ${file.arch}` : ''}
                      </div>
                      <button
                        onClick={() => onDownload(file)}
                        disabled={downloadingId === file.id}
                        className="btn-primary mt-3 w-full flex items-center justify-center gap-2"
                      >
                        <Download className="w-4 h-4" />
                        {downloadingId === file.id ? 'جاري التنزيل...' : 'تنزيل'}
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
