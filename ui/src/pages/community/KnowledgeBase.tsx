import React, { useState } from 'react';
import { Search, Plus, Tag, Eye, ThumbsUp, ThumbsDown, FileText, FolderOpen, Clock, User } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui';
import { useCommunity } from '../../hooks/useCommunity';

interface Article {
  id: string;
  title: string;
  content: string;
  category: string;
  author: string;
  tags: string[];
  viewCount: number;
  helpfulCount: number;
  notHelpfulCount: number;
  createdAt: string;
  status: 'draft' | 'published' | 'archived';
}

interface Category {
  id: string;
  name: string;
  description: string;
  articleCount: number;
  icon: string;
  color: string;
}

export const KnowledgeBase: React.FC = () => {
  const { communityData, createArticle, searchArticles } = useCommunity();
  const [activeTab, setActiveTab] = useState<'browse' | 'categories' | 'my_articles'>('browse');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [showNewArticle, setShowNewArticle] = useState(false);
  const [selectedArticle, setSelectedArticle] = useState<Article | null>(null);
  const [newArticle, setNewArticle] = useState({
    title: '',
    content: '',
    categoryId: '',
    tags: ''
  });

  const categories: Category[] = [
    { id: '1', name: 'Getting Started', description: 'Basic guides and tutorials', articleCount: 15, icon: 'rocket', color: 'bg-blue-500' },
    { id: '2', name: 'API Documentation', description: 'API references and examples', articleCount: 32, icon: 'code', color: 'bg-green-500' },
    { id: '3', name: 'Troubleshooting', description: 'Common issues and solutions', articleCount: 24, icon: 'wrench', color: 'bg-orange-500' },
    { id: '4', name: 'Best Practices', description: 'Recommended approaches', articleCount: 18, icon: 'star', color: 'bg-purple-500' },
    { id: '5', name: 'FAQ', description: 'Frequently asked questions', articleCount: 45, icon: 'help', color: 'bg-pink-500' }
  ];

  const kbData = communityData as any;
  const filteredArticles = kbData?.articles?.filter((article: any) => {
    const matchesSearch = article.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         article.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         article.tags.some((tag: any) => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesCategory = !selectedCategory || article.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const handleCreateArticle = async () => {
    await createArticle({
      ...newArticle,
      tags: newArticle.tags.split(',').map(t => t.trim())
    });
    setShowNewArticle(false);
    setNewArticle({ title: '', content: '', categoryId: '', tags: '' });
  };

  const handleSearch = async () => {
    if (searchTerm) {
      await searchArticles(searchTerm);
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Knowledge Base</h1>
          <p className="text-gray-600 mt-1">Find answers, tutorials, and documentation</p>
        </div>
        <button
          onClick={() => setShowNewArticle(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <Plus size={20} />
          New Article
        </button>
      </div>

      {/* Search Bar */}
      <div className="relative">
        <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
        <input
          type="text"
          placeholder="Search articles, tutorials, and documentation..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          className="w-full pl-12 pr-4 py-4 text-lg border rounded-lg shadow-sm"
        />
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex gap-8">
          {(['browse', 'categories', 'my_articles'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-4 px-1 border-b-2 font-medium capitalize ${
                activeTab === tab
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab.replace('_', ' ')}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'browse' && (
        <div className="space-y-6">
          {/* Featured Articles */}
          <div>
            <h2 className="text-xl font-semibold mb-4">Featured Articles</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {kbData?.featuredArticles?.map((article: Article) => (
                <Card
                  key={article.id}
                  className="cursor-pointer hover:shadow-lg transition-shadow"
                  onClick={() => setSelectedArticle(article)}
                >
                  <CardContent className="p-6">
                    <div className="flex items-start gap-4">
                      <div className="p-3 bg-blue-100 rounded-lg">
                        <FileText className="text-blue-600" size={24} />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-semibold text-lg mb-2">{article.title}</h3>
                        <p className="text-gray-600 text-sm line-clamp-2">{article.content}</p>
                        <div className="flex items-center gap-4 mt-3 text-sm text-gray-500">
                          <span className="flex items-center gap-1">
                            <User size={14} />
                            {article.author}
                          </span>
                          <span className="flex items-center gap-1">
                            <Eye size={14} />
                            {article.viewCount} views
                          </span>
                          <span className="flex items-center gap-1">
                            <ThumbsUp size={14} />
                            {article.helpfulCount}
                          </span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          {/* All Articles */}
          <div>
            <h2 className="text-xl font-semibold mb-4">All Articles</h2>
            <div className="space-y-3">
              {filteredArticles?.map((article: Article) => (
                <Card
                  key={article.id}
                  className="cursor-pointer hover:shadow-md transition-shadow"
                  onClick={() => setSelectedArticle(article)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <FileText className="text-gray-400" size={20} />
                        <div>
                          <h3 className="font-medium">{article.title}</h3>
                          <div className="flex items-center gap-3 mt-1 text-sm text-gray-500">
                            <span>{article.category}</span>
                            <span>•</span>
                            <span className="flex items-center gap-1">
                              <Clock size={14} />
                              {new Date(article.createdAt).toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-4 text-sm text-gray-500">
                        <span className="flex items-center gap-1">
                          <Eye size={16} />
                          {article.viewCount}
                        </span>
                        <div className="flex items-center gap-1">
                          <ThumbsUp size={16} className="text-green-500" />
                          {article.helpfulCount}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'categories' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {categories.map((category) => (
            <Card
              key={category.id}
              className="cursor-pointer hover:shadow-lg transition-shadow"
              onClick={() => setSelectedCategory(category.name)}
            >
              <CardContent className="p-6">
                <div className={`w-12 h-12 ${category.color} rounded-lg flex items-center justify-center mb-4`}>
                  <FolderOpen className="text-white" size={24} />
                </div>
                <h3 className="text-lg font-semibold mb-2">{category.name}</h3>
                <p className="text-gray-600 text-sm mb-4">{category.description}</p>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">{category.articleCount} articles</span>
                  <span className="text-blue-600 text-sm font-medium">Browse →</span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {activeTab === 'my_articles' && (
        <Card>
          <CardHeader>
            <CardTitle>My Articles</CardTitle>
          </CardHeader>
          <CardContent>
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Title</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Category</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Status</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Views</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Rating</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {kbData?.myArticles?.map((article: Article) => (
                  <tr key={article.id}>
                    <td className="px-4 py-3 font-medium">{article.title}</td>
                    <td className="px-4 py-3 text-gray-600">{article.category}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-sm ${
                        article.status === 'published' ? 'bg-green-100 text-green-800' :
                        article.status === 'draft' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {article.status}
                      </span>
                    </td>
                    <td className="px-4 py-3">{article.viewCount}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <ThumbsUp size={16} className="text-green-500" />
                        {article.helpfulCount}
                        <ThumbsDown size={16} className="text-red-500 ml-2" />
                        {article.notHelpfulCount}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      {/* Article Detail Modal */}
      {selectedArticle && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b">
              <div className="flex justify-between items-start">
                <div>
                  <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                    {selectedArticle.category}
                  </span>
                  <h2 className="text-2xl font-bold mt-3">{selectedArticle.title}</h2>
                  <div className="flex items-center gap-4 mt-2 text-sm text-gray-500">
                    <span className="flex items-center gap-1">
                      <User size={16} />
                      {selectedArticle.author}
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock size={16} />
                      {new Date(selectedArticle.createdAt).toLocaleDateString()}
                    </span>
                    <span className="flex items-center gap-1">
                      <Eye size={16} />
                      {selectedArticle.viewCount} views
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedArticle(null)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>
            </div>
            <div className="p-6">
              <div className="prose max-w-none">
                {selectedArticle.content}
              </div>
              <div className="mt-6 pt-6 border-t">
                <div className="flex items-center gap-2 mb-4">
                  <Tag size={16} className="text-gray-400" />
                  {selectedArticle.tags.map((tag, i) => (
                    <span key={i} className="px-2 py-1 bg-gray-100 rounded text-sm">{tag}</span>
                  ))}
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-gray-600">Was this article helpful?</span>
                  <button className="flex items-center gap-2 px-4 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200">
                    <ThumbsUp size={18} />
                    Yes ({selectedArticle.helpfulCount})
                  </button>
                  <button className="flex items-center gap-2 px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200">
                    <ThumbsDown size={18} />
                    No ({selectedArticle.notHelpfulCount})
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* New Article Modal */}
      {showNewArticle && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg w-full max-w-2xl max-h-[90vh] overflow-y-auto p-6">
            <h2 className="text-xl font-bold mb-4">Create New Article</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Title</label>
                <input
                  type="text"
                  value={newArticle.title}
                  onChange={(e) => setNewArticle({ ...newArticle, title: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                  placeholder="Article title"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
                <select
                  value={newArticle.categoryId}
                  onChange={(e) => setNewArticle({ ...newArticle, categoryId: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                >
                  <option value="">Select Category</option>
                  {categories.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Content</label>
                <textarea
                  value={newArticle.content}
                  onChange={(e) => setNewArticle({ ...newArticle, content: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg h-48"
                  placeholder="Write your article content..."
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Tags (comma separated)</label>
                <input
                  type="text"
                  value={newArticle.tags}
                  onChange={(e) => setNewArticle({ ...newArticle, tags: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                  placeholder="api, tutorial, getting-started"
                />
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={handleCreateArticle}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Publish
              </button>
              <button
                onClick={() => setShowNewArticle(false)}
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

export default KnowledgeBase;
