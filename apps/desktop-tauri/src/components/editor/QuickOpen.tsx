//! Quick Open Component (Cmd+P)
//! Fuzzy file search across the entire workspace

import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { Search, File, Folder, X, Loader2 } from "lucide-react";
import { useStore } from "../../lib/store";
import { fs } from "../../lib/tauri";
import { debounce } from "../../lib/utils";

interface QuickOpenProps {
  isOpen: boolean;
  onClose: () => void;
}

interface FileResult {
  path: string;
  name: string;
  relativePath: string;
  isDir: boolean;
  score: number;
}

// Fuzzy matching algorithm (simplified fuse.js-like)
function fuzzyMatch(pattern: string, str: string): number {
  pattern = pattern.toLowerCase();
  str = str.toLowerCase();
  
  const patternLength = pattern.length;
  const strLength = str.length;
  
  if (patternLength === 0) return 1;
  if (strLength === 0) return 0;
  
  let patternIdx = 0;
  let strIdx = 0;
  let matchedIndices: number[] = [];
  
  while (patternIdx < patternLength && strIdx < strLength) {
    if (pattern[patternIdx] === str[strIdx]) {
      matchedIndices.push(strIdx);
      patternIdx++;
    }
    strIdx++;
  }
  
  // Didn't match all characters
  if (patternIdx < patternLength) return 0;
  
  // Calculate score based on match quality
  let score = 0;
  let prevIdx = -1;
  
  for (const idx of matchedIndices) {
    // Bonus for consecutive matches
    if (prevIdx !== -1 && idx === prevIdx + 1) {
      score += 2;
    } else {
      score += 1;
    }
    
    // Bonus for matches at word boundaries
    if (idx === 0 || str[idx - 1] === "/" || str[idx - 1] === "_" || str[idx - 1] === "-") {
      score += 3;
    }
    
    prevIdx = idx;
  }
  
  // Penalty for length difference
  const lengthPenalty = (strLength - patternLength) * 0.1;
  score -= lengthPenalty;
  
  // Normalize to 0-1 range
  return Math.max(0, Math.min(1, score / (patternLength * 5)));
}

export function QuickOpen({ isOpen, onClose }: QuickOpenProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<FileResult[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [allFiles, setAllFiles] = useState<FileResult[]>([]);
  
  const inputRef = useRef<HTMLInputElement>(null);
  const { currentWorkspace, openFile } = useStore();

  // Load all files from workspace
  useEffect(() => {
    if (!isOpen || !currentWorkspace?.path) {
      setAllFiles([]);
      return;
    }

    const loadFiles = async () => {
      setIsLoading(true);
      try {
        const files = await fs.readDir(currentWorkspace.path, true);
        const fileResults: FileResult[] = [];
        
        const processEntry = (entry: any, basePath: string = "") => {
          const relativePath = basePath 
            ? `${basePath}/${entry.name}` 
            : entry.name;
          
          fileResults.push({
            path: entry.path || `${currentWorkspace.path}/${relativePath}`,
            name: entry.name,
            relativePath,
            isDir: entry.is_dir,
            score: 0,
          });
          
          if (entry.children && Array.isArray(entry.children)) {
            entry.children.forEach((child: any) => processEntry(child, relativePath));
          }
        };
        
        files.forEach((entry: any) => processEntry(entry));
        setAllFiles(fileResults);
      } catch (err) {
        console.error("Failed to load files:", err);
      } finally {
        setIsLoading(false);
      }
    };

    loadFiles();
  }, [isOpen, currentWorkspace?.path]);

  // Filter and sort results based on query
  const filteredResults = useMemo(() => {
    if (!query.trim()) {
      // Show recent files when no query
      return allFiles
        .filter(f => !f.isDir)
        .slice(0, 20)
        .map(f => ({ ...f, score: 1 }));
    }
    
    const scored = allFiles
      .map(file => ({
        ...file,
        score: fuzzyMatch(query, file.relativePath),
      }))
      .filter(file => file.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 50);
    
    return scored;
  }, [query, allFiles]);

  useEffect(() => {
    setResults(filteredResults);
    setSelectedIndex(0);
  }, [filteredResults]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  // Handle keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case "Escape":
          e.preventDefault();
          onClose();
          break;
          
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex(prev => 
            prev < results.length - 1 ? prev + 1 : prev
          );
          break;
          
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex(prev => prev > 0 ? prev - 1 : 0);
          break;
          
        case "Enter":
          e.preventDefault();
          if (results[selectedIndex] && !results[selectedIndex].isDir) {
            handleSelect(results[selectedIndex]);
          }
          break;
          
        case "Tab":
          e.preventDefault();
          if (e.shiftKey) {
            setSelectedIndex(prev => prev > 0 ? prev - 1 : results.length - 1);
          } else {
            setSelectedIndex(prev => 
              prev < results.length - 1 ? prev + 1 : 0
            );
          }
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, results, selectedIndex, onClose]);

  const handleSelect = useCallback(async (result: FileResult) => {
    if (result.isDir) return;
    
    try {
      const fileData = await fs.readFile(result.path);
      openFile(result.path, fileData.content);
      onClose();
      setQuery("");
    } catch (err) {
      console.error("Failed to open file:", err);
    }
  }, [openFile, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh] bg-black/50 backdrop-blur-sm">
      <div className="w-full max-w-2xl bg-dark-800 rounded-lg shadow-2xl border border-dark-600 overflow-hidden">
        {/* Search Input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-dark-600">
          <Search className="w-5 h-5 text-dark-400" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type to search files..."
            className="flex-1 bg-transparent text-dark-100 placeholder-dark-500 outline-none text-lg"
            spellCheck={false}
          />
          {isLoading && <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />}
          <button
            onClick={onClose}
            className="p-1 hover:bg-dark-700 rounded"
          >
            <X className="w-5 h-5 text-dark-400" />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-[50vh] overflow-y-auto">
          {results.length === 0 ? (
            <div className="px-4 py-8 text-center text-dark-500">
              {query ? "No matching files found" : "Start typing to search files"}
            </div>
          ) : (
            <div className="py-2">
              {results.map((result, index) => (
                <div
                  key={result.path}
                  onClick={() => handleSelect(result)}
                  className={`
                    flex items-center gap-3 px-4 py-2 cursor-pointer
                    ${index === selectedIndex ? "bg-primary-600/30" : "hover:bg-dark-700/50"}
                    ${result.isDir ? "text-dark-300" : "text-dark-100"}
                  `}
                >
                  {result.isDir ? (
                    <Folder className="w-4 h-4 text-yellow-500/70 flex-shrink-0" />
                  ) : (
                    <File className="w-4 h-4 text-blue-400/70 flex-shrink-0" />
                  )}
                  
                  <div className="flex-1 min-w-0">
                    <div className="truncate text-sm">{result.name}</div>
                    <div className="truncate text-xs text-dark-500">
                      {result.relativePath}
                    </div>
                  </div>
                  
                  {result.isDir && (
                    <span className="text-xs text-dark-500 px-2 py-0.5 bg-dark-700 rounded">
                      Folder
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 bg-dark-900/50 border-t border-dark-600 text-xs text-dark-500">
          <div className="flex items-center gap-4">
            <span><kbd className="px-1.5 py-0.5 bg-dark-700 rounded">↑↓</kbd> to navigate</span>
            <span><kbd className="px-1.5 py-0.5 bg-dark-700 rounded">↵</kbd> to open</span>
            <span><kbd className="px-1.5 py-0.5 bg-dark-700 rounded">esc</kbd> to close</span>
          </div>
          <span>{results.length} results</span>
        </div>
      </div>
    </div>
  );
}
