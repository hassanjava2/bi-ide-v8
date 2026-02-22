import { FileCode, Folder, RefreshCw } from 'lucide-react'
import { memo, useCallback } from 'react'
import type { FileTreeNode } from './types'

interface FileExplorerProps {
  treeRoot: FileTreeNode | null
  selectedFileId: string
  onOpenFile: (node: FileTreeNode, targetLine?: number) => void
  onRefresh: () => void
}

const FileTreeItem = memo(function FileTreeItem({
  node,
  depth,
  selectedFileId,
  onOpenFile
}: {
  node: FileTreeNode
  depth: number
  selectedFileId: string
  onOpenFile: (node: FileTreeNode, targetLine?: number) => void
}) {
  const isFile = node.type === 'file'

  const handleClick = useCallback(() => {
    if (isFile) {
      onOpenFile(node)
    }
  }, [isFile, node, onOpenFile])

  return (
    <div key={node.id}>
      <div
        className={`flex items-center gap-2 p-2 hover:bg-white/5 rounded cursor-pointer ${selectedFileId === node.id ? 'bg-white/10' : ''}`}
        style={{ paddingRight: `${8 + depth * 14}px` }}
        onClick={handleClick}
      >
        {isFile ? (
          <FileCode className="w-4 h-4 text-blue-400" />
        ) : (
          <Folder className="w-4 h-4 text-yellow-400" />
        )}
        <span className="text-sm text-gray-300 truncate">{node.name}</span>
      </div>
      {!isFile && (node.children || []).map((child) => (
        <FileTreeItem
          key={child.id}
          node={child}
          depth={depth + 1}
          selectedFileId={selectedFileId}
          onOpenFile={onOpenFile}
        />
      ))}
    </div>
  )
})

export const FileExplorer = memo(function FileExplorer({
  treeRoot,
  selectedFileId,
  onOpenFile,
  onRefresh
}: FileExplorerProps) {
  return (
    <div className="w-64 glass-panel flex flex-col">
      <div className="p-3 border-b border-white/10 flex items-center justify-between">
        <span className="text-sm font-medium">المستكشف</span>
        <div className="flex items-center gap-2">
          <button
            onClick={onRefresh}
            className="p-1 hover:bg-white/10 rounded"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4 text-gray-400" />
          </button>
          <Folder className="w-4 h-4 text-gray-400" />
        </div>
      </div>
      <div className="flex-1 overflow-auto p-2 space-y-1">
        {treeRoot ? (
          <FileTreeItem
            node={treeRoot}
            depth={0}
            selectedFileId={selectedFileId}
            onOpenFile={onOpenFile}
          />
        ) : (
          <div className="text-xs text-gray-400 p-2">Loading files...</div>
        )}
      </div>
    </div>
  )
})
