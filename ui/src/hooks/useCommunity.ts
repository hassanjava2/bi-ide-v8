import { useState, useCallback } from 'react';

// Types
export interface Forum {
  id: string;
  name: string;
  description: string;
  topicCount: number;
  postCount: number;
  icon: string;
  color: string;
}

export interface Topic {
  id: string;
  title: string;
  authorName: string;
  status: string;
  viewCount: number;
  replyCount: number;
  hasSolution: boolean;
  createdAt: string;
  lastPostAt: string;
}

export interface Post {
  id: string;
  authorName: string;
  content: string;
  upvotes: number;
  downvotes: number;
  isSolution: boolean;
  createdAt: string;
}

export interface Article {
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
  status: string;
}

export interface CodeSnippet {
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

interface CommunityData {
  forums?: Forum[];
  topics?: Topic[];
  posts?: Post[];
  articles?: Article[];
  featuredArticles?: Article[];
  myArticles?: Article[];
  snippets?: CodeSnippet[];
}

export const useCommunity = () => {
  const [communityData, setCommunityData] = useState<CommunityData>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Forums
  const fetchForums = useCallback(async () => {
    setLoading(true);
    try {
      const mockForums: Forum[] = [
        { id: '1', name: 'General Discussion', description: 'General topics and discussions', topicCount: 156, postCount: 1240, icon: 'message', color: '#2196F3' },
        { id: '2', name: 'Technical Support', description: 'Get help with technical issues', topicCount: 89, postCount: 567, icon: 'help', color: '#4CAF50' },
        { id: '3', name: 'Feature Requests', description: 'Suggest new features', topicCount: 45, postCount: 234, icon: 'lightbulb', color: '#FF9800' }
      ];
      setCommunityData(prev => ({ ...prev, forums: mockForums }));
    } catch (err) {
      setError('Failed to fetch forums');
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchTopics = useCallback(async (_forumId: string) => {
    setLoading(true);
    try {
      // In real implementation, would filter by forumId
      const mockTopics: Topic[] = [
        { id: '1', title: 'Welcome to the community!', authorName: 'Admin', status: 'open', viewCount: 1200, replyCount: 45, hasSolution: false, createdAt: '2024-01-01', lastPostAt: '2024-01-20' },
        { id: '2', title: 'How to get started with BI-IDE?', authorName: 'NewUser123', status: 'open', viewCount: 567, replyCount: 23, hasSolution: true, createdAt: '2024-01-15', lastPostAt: '2024-01-18' },
        { id: '3', title: 'Best practices for AI hierarchy', authorName: 'DevPro', status: 'pinned', viewCount: 890, replyCount: 34, hasSolution: false, createdAt: '2024-01-10', lastPostAt: '2024-01-19' }
      ];
      setCommunityData(prev => ({ ...prev, topics: mockTopics }));
    } catch (err) {
      setError('Failed to fetch topics');
    } finally {
      setLoading(false);
    }
  }, []);

  const createTopic = useCallback(async (_forumId: string, data: { title: string; content: string; tags?: string[] }) => {
    try {
      // In real implementation, would use forumId
      const newTopic: Topic = {
        id: Date.now().toString(),
        title: data.title,
        authorName: 'Current User',
        status: 'open',
        viewCount: 0,
        replyCount: 0,
        hasSolution: false,
        createdAt: new Date().toISOString(),
        lastPostAt: new Date().toISOString()
      };
      setCommunityData(prev => ({
        ...prev,
        topics: [newTopic, ...(prev.topics || [])]
      }));
      return newTopic;
    } catch (err) {
      setError('Failed to create topic');
      throw err;
    }
  }, []);

  // Knowledge Base
  const fetchArticles = useCallback(async () => {
    setLoading(true);
    try {
      const mockArticles: Article[] = [
        { 
          id: '1', 
          title: 'Getting Started Guide', 
          content: 'This guide will help you get started with BI-IDE...',
          category: 'Getting Started', 
          author: 'Documentation Team',
          tags: ['beginner', 'tutorial', 'setup'],
          viewCount: 3456, 
          helpfulCount: 234, 
          notHelpfulCount: 12, 
          createdAt: '2024-01-01',
          status: 'published'
        },
        { 
          id: '2', 
          title: 'API Authentication', 
          content: 'Learn how to authenticate with our API...',
          category: 'API Documentation', 
          author: 'API Team',
          tags: ['api', 'auth', 'jwt'],
          viewCount: 1890, 
          helpfulCount: 156, 
          notHelpfulCount: 8, 
          createdAt: '2024-01-10',
          status: 'published'
        },
        { 
          id: '3', 
          title: 'Troubleshooting Common Errors', 
          content: 'Solutions to common issues...',
          category: 'Troubleshooting', 
          author: 'Support Team',
          tags: ['errors', 'debugging', 'help'],
          viewCount: 2345, 
          helpfulCount: 189, 
          notHelpfulCount: 15, 
          createdAt: '2024-01-15',
          status: 'published'
        }
      ];

      setCommunityData(prev => ({ 
        ...prev, 
        articles: mockArticles,
        featuredArticles: mockArticles.filter(a => a.viewCount > 2000)
      }));
    } catch (err) {
      setError('Failed to fetch articles');
    } finally {
      setLoading(false);
    }
  }, []);

  const searchArticles = useCallback(async (query: string) => {
    setLoading(true);
    try {
      // Mock search - filter existing articles
      const allArticles = communityData.articles || [];
      const filtered = allArticles.filter(a => 
        a.title.toLowerCase().includes(query.toLowerCase()) ||
        a.content.toLowerCase().includes(query.toLowerCase())
      );
      setCommunityData(prev => ({ ...prev, articles: filtered }));
    } catch (err) {
      setError('Failed to search articles');
    } finally {
      setLoading(false);
    }
  }, [communityData.articles]);

  const createArticle = useCallback(async (data: { title: string; content: string; categoryId: string; tags: string[] }) => {
    try {
      const newArticle: Article = {
        id: Date.now().toString(),
        title: data.title,
        content: data.content,
        category: data.categoryId,
        author: 'Current User',
        tags: data.tags,
        viewCount: 0,
        helpfulCount: 0,
        notHelpfulCount: 0,
        createdAt: new Date().toISOString(),
        status: 'published'
      };
      setCommunityData(prev => ({
        ...prev,
        articles: [newArticle, ...(prev.articles || [])]
      }));
      return newArticle;
    } catch (err) {
      setError('Failed to create article');
      throw err;
    }
  }, []);

  // Code Sharing
  const fetchSnippets = useCallback(async (language?: string) => {
    setLoading(true);
    try {
      const mockSnippets: CodeSnippet[] = [
        {
          id: '1',
          title: 'Python Quick Sort',
          description: 'Efficient quicksort implementation in Python',
          code: `def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)`,
          language: 'Python',
          author: 'AlgoMaster',
          tags: ['algorithm', 'sorting', 'python'],
          viewCount: 1234,
          downloadCount: 567,
          stars: 89,
          createdAt: '2024-01-10'
        },
        {
          id: '2',
          title: 'React useAuth Hook',
          description: 'Custom hook for authentication in React',
          code: `const useAuth = () => {\n  const [user, setUser] = useState(null);\n  const login = async (credentials) => {\n    const user = await api.login(credentials);\n    setUser(user);\n  };\n  return { user, login };\n};`,
          language: 'TypeScript',
          author: 'ReactPro',
          tags: ['react', 'hooks', 'auth'],
          viewCount: 2345,
          downloadCount: 890,
          stars: 156,
          createdAt: '2024-01-15'
        }
      ];

      const filtered = language 
        ? mockSnippets.filter(s => s.language.toLowerCase() === language.toLowerCase())
        : mockSnippets;

      setCommunityData(prev => ({ ...prev, snippets: filtered }));
    } catch (err) {
      setError('Failed to fetch code snippets');
    } finally {
      setLoading(false);
    }
  }, []);

  const createSnippet = useCallback(async (data: { title: string; description: string; code: string; language: string; tags: string[] }) => {
    try {
      const newSnippet: CodeSnippet = {
        id: Date.now().toString(),
        title: data.title,
        description: data.description,
        code: data.code,
        language: data.language,
        author: 'Current User',
        tags: data.tags,
        viewCount: 0,
        downloadCount: 0,
        stars: 0,
        createdAt: new Date().toISOString()
      };
      setCommunityData(prev => ({
        ...prev,
        snippets: [newSnippet, ...(prev.snippets || [])]
      }));
      return newSnippet;
    } catch (err) {
      setError('Failed to create snippet');
      throw err;
    }
  }, []);

  return {
    communityData,
    loading,
    error,
    fetchForums,
    fetchTopics,
    createTopic,
    fetchArticles,
    searchArticles,
    createArticle,
    fetchSnippets,
    createSnippet
  };
};

export default useCommunity;
