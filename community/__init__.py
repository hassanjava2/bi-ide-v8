# Community Module - وحدة المجتمع
"""
Community Module - وحدة مميزات المجتمع

المميزات:
- User profiles with reputation system
- Forums
- Knowledge base
- Code sharing
"""

from .profiles import UserProfile, ReputationSystem, ActivityFeed
from .forums import Forum, Topic, Post, ForumManager
from .knowledge_base import KnowledgeBase, Article, ArticleCategory
from .code_sharing import CodeSnippet, SnippetCollection, CodeSharingPlatform

__all__ = [
    'UserProfile', 'ReputationSystem', 'ActivityFeed',
    'Forum', 'Topic', 'Post', 'ForumManager',
    'KnowledgeBase', 'Article', 'ArticleCategory',
    'CodeSnippet', 'SnippetCollection', 'CodeSharingPlatform',
]
