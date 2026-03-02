//! Search & Replace Panel (Cmd+Shift+F)
//! Global search across workspace using ripgrep

import { useState, useEffect, useCallback, useRef } from "react";
import { 
  Search, X, Replace, ChevronDown, ChevronRight, 
  FileText, Folder, Loader2, Settings2, Regex,
  CaseSensitive, WholeWord, Filter
} from "lucide-react";
import { useStore } from "../../lib/store";
import { invoke } from "@tauri-apps/api/core";

interface SearchPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

interface SearchResult {
  filePath: string;
  fileName: string;
  matches: MatchInfo[];
}

interface MatchInfo {
  lineNumber: number;
  column: number;
  lineText: string;
  matchText: string;
  replacement?: string;
}

interface SearchOptions {
  caseSensitive: boolean;
  wholeWord: boolean;
  useRegex: boolean;
  includePattern: string;
  excludePattern: string;
  maxResults: number;
}

export function SearchPanel({ isOpen, onClose }: SearchPanelProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [replaceQuery, setReplaceQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [expandedFiles, setExpandedFiles] = useState<Set<string>>(new Set());
  const [showOptions, setShowOptions] = useState(false);
  const [showReplace, setShowReplace] = useState(false);
  const [totalMatches, setTotalMatches] = useState(0);
  
  const [options, setOptions] = useState<SearchOptions>({
    caseSensitive: false,
    wholeWord: false,
    useRegex: false,
    includePattern: "",
    excludePattern: "node_modules/,dist/,build/,target/,.git/",
    maxResults: 1000,
  });

  const searchInputRef = useRef<HTMLInputElement>(null);
  const { currentWorkspace, openFile } = useStore();

  // Focus search input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => searchInputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  // Debounced search
  const performSearch = useCallback(async () => {
    if (!searchQuery.trim() || !currentWorkspace?.path) {
      setResults([]);
      setTotalMatches(0);
      return;
    }

    setIsSearching(true);
    try {
      // Call Rust command to perform ripgrep search
      const searchResults = await invoke<SearchResult[]>("search_workspace", {
        request: {
          query: searchQuery,
          workspace_path: currentWorkspace.path,
          options: {
            case_sensitive: options.caseSensitive,
            whole_word: options.wholeWord,
            use_regex: options.useRegex,
            include_pattern: options.includePattern || null,
            exclude_pattern: options.excludePattern || null,
            max_results: options.maxResults,
          },
        },
      });

      setResults(searchResults);
      setTotalMatches(searchResults.reduce((acc, r) => acc + r.matches.length, 0));
      
      // Expand all files with results
      setExpandedFiles(new Set(searchResults.map(r => r.filePath)));
    } catch (err) {
      console.error("Search failed:", err);
      setResults([]);
      setTotalMatches(0);
    } finally {
      setIsSearching(false);
    }
  }, [searchQuery, currentWorkspace?.path, options]);

  // Debounced search effect
  useEffect(() => {
    const timer = setTimeout(performSearch, 300);
    return () => clearTimeout(timer);
  }, [performSearch]);

  const toggleFileExpanded = (filePath: string) => {
    setExpandedFiles(prev => {
      const next = new Set(prev);
      if (next.has(filePath)) {
        next.delete(filePath);
      } else {
        next.add(filePath);
      }
      return next;
    });
  };

  const handleOpenFile = async (filePath: string, lineNumber?: number) => {
    try {
      const content = await invoke<string>("read_file", { 
        request: { path: filePath } 
      });
      openFile(filePath, content);
      // Could pass line number to scroll to it
    } catch (err) {
      console.error("Failed to open file:", err);
    }
  };

  const handleReplaceInFile = async (filePath: string) => {
    // Implementation for single file replace
    console.log("Replace in file:", filePath);
  };

  const handleReplaceAll = async () => {
    // Implementation for replace all
    console.log("Replace all");
  };

  if (!isOpen) return null;

  return (
    <div className="h-full flex flex-col bg-dark-800 border-r border-dark-700">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-dark-700">
        <div className="flex items-center gap-2">
          <Search className="w-4 h-4 text-primary-400" />
          <span className="text-sm font-medium">Search</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowOptions(!showOptions)}
            className={`p-1.5 rounded ${showOptions ? "bg-primary-600/30 text-primary-400" : "hover:bg-dark-700"}`}
            title="Search options"
          >
            <Settings2 className="w-4 h-4" />
          </button>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-dark-700 rounded"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Search Input */}
      <div className="p-4 space-y-3">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-400" />
          <input
            ref={searchInputRef}
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search across files..."
            className="w-full pl-10 pr-10 py-2 bg-dark-900 border border-dark-600 rounded text-sm text-dark-100 placeholder-dark-500 outline-none focus:border-primary-500"
          />
          {isSearching && (
            <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-primary-400 animate-spin" />
          )}
        </div>

        {/* Toggle Replace */}
        <button
          onClick={() => setShowReplace(!showReplace)}
          className="flex items-center gap-1 text-xs text-dark-400 hover:text-dark-200"
        >
          {showReplace ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          Toggle Replace
        </button>

        {/* Replace Input */}
        {showReplace && (
          <div className="relative">
            <Replace className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-400" />
            <input
              type="text"
              value={replaceQuery}
              onChange={(e) => setReplaceQuery(e.target.value)}
              placeholder="Replace with..."
              className="w-full pl-10 pr-4 py-2 bg-dark-900 border border-dark-600 rounded text-sm text-dark-100 placeholder-dark-500 outline-none focus:border-primary-500"
            />
          </div>
        )}

        {/* Options Panel */}
        {showOptions && (
          <div className="p-3 bg-dark-900 rounded border border-dark-700 space-y-3">
            <div className="flex flex-wrap gap-2">
              <ToggleButton
                active={options.caseSensitive}
                onClick={() => setOptions(o => ({ ...o, caseSensitive: !o.caseSensitive }))}
                icon={<CaseSensitive className="w-3.5 h-3.5" />}
                label="Match Case"
              />
              <ToggleButton
                active={options.wholeWord}
                onClick={() => setOptions(o => ({ ...o, wholeWord: !o.wholeWord }))}
                icon={<WholeWord className="w-3.5 h-3.5" />}
                label="Whole Word"
              />
              <ToggleButton
                active={options.useRegex}
                onClick={() => setOptions(o => ({ ...o, useRegex: !o.useRegex }))}
                icon={<Regex className="w-3.5 h-3.5" />}
                label="Regex"
              />
            </div>
            
            <div className="space-y-2">
              <div>
                <label className="text-xs text-dark-500">Include pattern</label>
                <input
                  type="text"
                  value={options.includePattern}
                  onChange={(e) => setOptions(o => ({ ...o, includePattern: e.target.value }))}
                  placeholder="e.g., *.ts, *.tsx"
                  className="w-full mt-1 px-2 py-1 bg-dark-800 border border-dark-600 rounded text-xs text-dark-100 outline-none"
                />
              </div>
              <div>
                <label className="text-xs text-dark-500">Exclude pattern</label>
                <input
                  type="text"
                  value={options.excludePattern}
                  onChange={(e) => setOptions(o => ({ ...o, excludePattern: e.target.value }))}
                  placeholder="e.g., node_modules/, dist/"
                  className="w-full mt-1 px-2 py-1 bg-dark-800 border border-dark-600 rounded text-xs text-dark-100 outline-none"
                />
              </div>
            </div>
          </div>
        )}

        {/* Replace Actions */}
        {showReplace && searchQuery && (
          <div className="flex gap-2">
            <button
              onClick={handleReplaceAll}
              className="flex-1 px-3 py-1.5 bg-primary-600 hover:bg-primary-500 text-white text-xs rounded transition-colors"
            >
              Replace All ({totalMatches})
            </button>
          </div>
        )}
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto">
        {results.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 text-dark-500">
            {searchQuery ? (
              <>
                <Search className="w-8 h-8 mb-2 opacity-50" />
                <span className="text-sm">No results found</span>
              </>
            ) : (
              <>
                <Filter className="w-8 h-8 mb-2 opacity-50" />
                <span className="text-sm">Type to search across files</span>
              </>
            )}
          </div>
        ) : (
          <div className="pb-4">
            <div className="px-4 py-2 text-xs text-dark-500 border-b border-dark-700">
              {totalMatches} results in {results.length} files
            </div>
            
            {results.map((result) => (
              <div key={result.filePath}>
                {/* File Header */}
                <button
                  onClick={() => toggleFileExpanded(result.filePath)}
                  className="w-full flex items-center gap-2 px-4 py-1.5 hover:bg-dark-700/50 text-left"
                >
                  {expandedFiles.has(result.filePath) ? (
                    <ChevronDown className="w-3.5 h-3.5 text-dark-500" />
                  ) : (
                    <ChevronRight className="w-3.5 h-3.5 text-dark-500" />
                  )}
                  <FileText className="w-4 h-4 text-blue-400" />
                  <span className="text-sm text-dark-200 truncate flex-1">{result.fileName}</span>
                  <span className="text-xs text-dark-500">{result.matches.length}</span>
                </button>

                {/* Matches */}
                {expandedFiles.has(result.filePath) && (
                  <div className="border-l-2 border-dark-700 ml-6">
                    {result.matches.map((match, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleOpenFile(result.filePath, match.lineNumber)}
                        className="w-full px-4 py-1.5 text-left hover:bg-dark-700/30 group"
                      >
                        <div className="flex items-start gap-3">
                          <span className="text-xs text-dark-500 w-10 text-right flex-shrink-0">
                            {match.lineNumber}
                          </span>
                          <code className="text-xs text-dark-300 font-mono truncate">
                            {highlightMatch(match.lineText, match.matchText, searchQuery)}
                          </code>
                        </div>
                      </button>
                    ))}
                    
                    {showReplace && (
                      <button
                        onClick={() => handleReplaceInFile(result.filePath)}
                        className="w-full px-4 py-1 text-left text-xs text-primary-400 hover:bg-dark-700/30"
                      >
                        Replace {result.matches.length} in this file
                      </button>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// Toggle Button Component
function ToggleButton({ 
  active, 
  onClick, 
  icon, 
  label 
}: { 
  active: boolean; 
  onClick: () => void; 
  icon: React.ReactNode; 
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        flex items-center gap-1.5 px-2 py-1 rounded text-xs transition-colors
        ${active 
          ? "bg-primary-600/30 text-primary-400 border border-primary-500/30" 
          : "bg-dark-800 text-dark-400 border border-dark-600 hover:border-dark-500"
        }
      `}
      title={label}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

// Highlight match in text
function highlightMatch(lineText: string, matchText: string, query: string): React.ReactNode {
  const index = lineText.toLowerCase().indexOf(matchText.toLowerCase());
  if (index === -1) return lineText;
  
  return (
    <>
      {lineText.slice(0, index)}
      <mark className="bg-primary-500/40 text-dark-100 px-0.5 rounded">
        {lineText.slice(index, index + matchText.length)}
      </mark>
      {lineText.slice(index + matchText.length)}
    </>
  );
}
