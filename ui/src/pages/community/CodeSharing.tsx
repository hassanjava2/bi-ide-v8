import React, { useState } from 'react';
import { Copy, Download, Eye, Star, Search, Plus, Tag, Terminal, FileCode, Check } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui';
import { useCommunity } from '../../hooks/useCommunity';

interface CodeSnippet {
  id: string;
  title: string;
  description: string;
  code: string;
  language: string;
  author: string;
  tags: string[];
  viewCount: number;
  downloadCount: number;
  stars: number;
  createdAt: string;
}

const languages = [
  { name: 'Python', color: 'bg-blue-500', icon: '🐍' },
  { name: 'JavaScript', color: 'bg-yellow-500', icon: '📜' },
  { name: 'TypeScript', color: 'bg-blue-600', icon: '🔷' },
  { name: 'Java', color: 'bg-orange-500', icon: '☕' },
  { name: 'C++', color: 'bg-purple-500', icon: '⚡' },
  { name: 'Go', color: 'bg-cyan-500', icon: '🐹' },
  { name: 'Rust', color: 'bg-orange-600', icon: '🦀' },
  { name: 'SQL', color: 'bg-green-500', icon: '🗄️' }
];

export const CodeSharing: React.FC = () => {
  const { communityData, createSnippet } = useCommunity();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState<string | null>(null);
  const [showNewSnippet, setShowNewSnippet] = useState(false);
  const [selectedSnippet, setSelectedSnippet] = useState<CodeSnippet | null>(null);
  const [copied, setCopied] = useState(false);
  const [newSnippet, setNewSnippet] = useState({
    title: '',
    description: '',
    code: '',
    language: 'Python',
    tags: ''
  });

  const codeData = communityData;
  const filteredSnippets = codeData?.snippets?.filter((snippet: CodeSnippet) => {
    const matchesSearch = snippet.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         snippet.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         snippet.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesLanguage = !selectedLanguage || snippet.language === selectedLanguage;
    return matchesSearch && matchesLanguage;
  });

  const handleCreateSnippet = async () => {
    await createSnippet({
      ...newSnippet,
      tags: newSnippet.tags.split(',').map(t => t.trim())
    });
    setShowNewSnippet(false);
    setNewSnippet({ title: '', description: '', code: '', language: 'Python', tags: '' });
  };

  const handleCopy = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getLanguageColor = (lang: string) => {
    const language = languages.find(l => l.name.toLowerCase() === lang.toLowerCase());
    return language?.color || 'bg-gray-500';
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Code Sharing</h1>
          <p className="text-gray-600 mt-1">Share and discover code snippets</p>
        </div>
        <button
          onClick={() => setShowNewSnippet(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <Plus size={20} />
          Share Code
        </button>
      </div>

      {/* Search */}
      <div className="flex gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
          <input
            type="text"
            placeholder="Search snippets by title, description, or tags..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-3 border rounded-lg"
          />
        </div>
      </div>

      {/* Language Filters */}
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setSelectedLanguage(null)}
          className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
            !selectedLanguage ? 'bg-gray-900 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          All Languages
        </button>
        {languages.map((lang) => (
          <button
            key={lang.name}
            onClick={() => setSelectedLanguage(lang.name)}
            className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
              selectedLanguage === lang.name 
                ? `${lang.color} text-white` 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <span className="mr-1">{lang.icon}</span>
            {lang.name}
          </button>
        ))}
      </div>

      {/* Snippets Grid */}
      <div className="grid grid-cols-1 gap-4">
        {filteredSnippets?.map((snippet: CodeSnippet) => (
          <Card
            key={snippet.id}
            className="cursor-pointer hover:shadow-lg transition-shadow"
            onClick={() => setSelectedSnippet(snippet)}
          >
            <CardContent className="p-6">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className={`px-3 py-1 ${getLanguageColor(snippet.language)} text-white rounded text-sm font-medium`}>
                      {snippet.language}
                    </span>
                    <h3 className="text-lg font-semibold">{snippet.title}</h3>
                  </div>
                  <p className="text-gray-600 text-sm mb-3">{snippet.description}</p>
                  <div className="flex items-center gap-4 text-sm text-gray-500">
                    <span className="flex items-center gap-1">
                      <Terminal size={14} />
                      {snippet.author}
                    </span>
                    <span className="flex items-center gap-1">
                      <Eye size={14} />
                      {snippet.viewCount} views
                    </span>
                    <span className="flex items-center gap-1">
                      <Download size={14} />
                      {snippet.downloadCount} downloads
                    </span>
                    <span className="flex items-center gap-1">
                      <Star size={14} className="text-yellow-400" />
                      {snippet.stars}
                    </span>
                  </div>
                  <div className="flex gap-2 mt-3">
                    {snippet.tags.map((tag, i) => (
                      <span key={i} className="px-2 py-1 bg-gray-100 rounded text-xs">#{tag}</span>
                    ))}
                  </div>
                </div>
                <div className="ml-4">
                  <FileCode size={48} className="text-gray-300" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Popular Tags */}
      <Card>
        <CardHeader>
          <CardTitle>Popular Tags</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {['algorithm', 'data-structure', 'web-scraping', 'api', 'database', 'machine-learning', 'automation', 'utility'].map((tag) => (
              <button
                key={tag}
                onClick={() => setSearchTerm(tag)}
                className="px-3 py-1 bg-gray-100 rounded-full text-sm hover:bg-gray-200"
              >
                #{tag}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Snippet Detail Modal */}
      {selectedSnippet && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b">
              <div className="flex justify-between items-start">
                <div>
                  <div className="flex items-center gap-3 mb-2">
                    <span className={`px-3 py-1 ${getLanguageColor(selectedSnippet.language)} text-white rounded text-sm font-medium`}>
                      {selectedSnippet.language}
                    </span>
                    <h2 className="text-2xl font-bold">{selectedSnippet.title}</h2>
                  </div>
                  <p className="text-gray-600">{selectedSnippet.description}</p>
                  <div className="flex items-center gap-4 mt-3 text-sm text-gray-500">
                    <span>By {selectedSnippet.author}</span>
                    <span>•</span>
                    <span>{new Date(selectedSnippet.createdAt).toLocaleDateString()}</span>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedSnippet(null)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>
            </div>
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-4">
                  <button
                    onClick={() => handleCopy(selectedSnippet.code)}
                    className="flex items-center gap-2 px-4 py-2 bg-gray-100 rounded-lg hover:bg-gray-200"
                  >
                    {copied ? <Check size={18} className="text-green-600" /> : <Copy size={18} />}
                    {copied ? 'Copied!' : 'Copy'}
                  </button>
                  <button className="flex items-center gap-2 px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200">
                    <Download size={18} />
                    Download
                  </button>
                </div>
                <div className="flex items-center gap-4 text-sm text-gray-500">
                  <span className="flex items-center gap-1">
                    <Eye size={16} />
                    {selectedSnippet.viewCount} views
                  </span>
                  <span className="flex items-center gap-1">
                    <Star size={16} className="text-yellow-400" />
                    {selectedSnippet.stars} stars
                  </span>
                </div>
              </div>
              <pre className="bg-gray-900 text-gray-100 p-6 rounded-lg overflow-x-auto">
                <code>{selectedSnippet.code}</code>
              </pre>
              <div className="mt-6 flex items-center gap-2">
                <Tag size={16} className="text-gray-400" />
                {selectedSnippet.tags.map((tag, i) => (
                  <span key={i} className="px-3 py-1 bg-gray-100 rounded-full text-sm">{tag}</span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* New Snippet Modal */}
      {showNewSnippet && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg w-full max-w-3xl max-h-[90vh] overflow-y-auto p-6">
            <h2 className="text-xl font-bold mb-4">Share Code Snippet</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Title</label>
                <input
                  type="text"
                  value={newSnippet.title}
                  onChange={(e) => setNewSnippet({ ...newSnippet, title: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                  placeholder="Give your snippet a title"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                <input
                  type="text"
                  value={newSnippet.description}
                  onChange={(e) => setNewSnippet({ ...newSnippet, description: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                  placeholder="Brief description of what this code does"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Language</label>
                <select
                  value={newSnippet.language}
                  onChange={(e) => setNewSnippet({ ...newSnippet, language: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                >
                  {languages.map(l => <option key={l.name} value={l.name}>{l.name}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Code</label>
                <textarea
                  value={newSnippet.code}
                  onChange={(e) => setNewSnippet({ ...newSnippet, code: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg h-64 font-mono text-sm"
                  placeholder="Paste your code here..."
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Tags (comma separated)</label>
                <input
                  type="text"
                  value={newSnippet.tags}
                  onChange={(e) => setNewSnippet({ ...newSnippet, tags: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                  placeholder="algorithm, python, sorting"
                />
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={handleCreateSnippet}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Share Snippet
              </button>
              <button
                onClick={() => setShowNewSnippet(false)}
                className="flex-1 px-4 py-2 border rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CodeSharing;
