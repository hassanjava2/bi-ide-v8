import { FileCode } from 'lucide-react'
import { memo, useCallback, type KeyboardEvent, type MouseEvent, type Ref } from 'react'

interface CodeEditorProps {
  editorRef: Ref<HTMLTextAreaElement>
  content: string
  selectedFileName: string
  saveMessage: string
  onContentChange: (content: string) => void
  onSelectionChange: (start: number, end: number) => void
  onMouseUp: (event: MouseEvent<HTMLTextAreaElement>) => void
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void
}

export const CodeEditor = memo(function CodeEditor({
  editorRef,
  content,
  selectedFileName,
  saveMessage,
  onContentChange,
  onSelectionChange,
  onMouseUp,
  onKeyDown
}: CodeEditorProps) {
  const handleChange = useCallback((event: React.ChangeEvent<HTMLTextAreaElement>) => {
    onContentChange(event.target.value)
  }, [onContentChange])

  const handleSelect = useCallback((event: React.SyntheticEvent<HTMLTextAreaElement>) => {
    const target = event.currentTarget
    onSelectionChange(target.selectionStart || 0, target.selectionEnd || 0)
  }, [onSelectionChange])

  const handleKeyUp = useCallback((event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    const target = event.currentTarget
    onSelectionChange(target.selectionStart || 0, target.selectionEnd || 0)
  }, [onSelectionChange])

  return (
    <div className="flex-1 glass-panel flex flex-col min-w-0">
      <div className="flex items-center gap-1 p-2 border-b border-white/10">
        <div className="px-4 py-2 bg-white/10 rounded-t-lg flex items-center gap-2">
          <FileCode className="w-4 h-4 text-blue-400" />
          <span className="text-sm">{selectedFileName || 'No file selected'}</span>
        </div>
        {saveMessage && <span className="text-xs text-gray-400 px-2">{saveMessage}</span>}
      </div>
      <div className="flex-1 p-4 font-mono text-sm overflow-auto">
        <textarea
          ref={editorRef}
          value={content}
          onChange={handleChange}
          onKeyDown={onKeyDown}
          onSelect={handleSelect}
          onMouseUp={onMouseUp}
          onKeyUp={handleKeyUp}
          className="w-full h-full bg-transparent text-gray-300 outline-none resize-none"
          placeholder="Select a file from explorer..."
          spellCheck={false}
        />
      </div>
    </div>
  )
})
