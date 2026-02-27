"""
Forums - المنتديات

المميزات:
- الفئات والمواضيع والمشاركات
- أدوات الإشراف
- البحث
- الإشعارات
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class TopicStatus(Enum):
    """حالة الموضوع"""
    OPEN = "open"
    CLOSED = "closed"
    PINNED = "pinned"
    ANNOUNCEMENT = "announcement"


class PostStatus(Enum):
    """حالة المشاركة"""
    VISIBLE = "visible"
    HIDDEN = "hidden"
    DELETED = "deleted"
    FLAGGED = "flagged"


@dataclass
class Post:
    """مشاركة"""
    id: str
    topic_id: str
    author_id: str
    author_name: str
    content: str
    
    status: PostStatus = PostStatus.VISIBLE
    is_solution: bool = False         # هل هي حل للموضوع؟
    
    # Voting
    upvotes: int = 0
    downvotes: int = 0
    
    # Replies
    parent_id: Optional[str] = None   # للردود على ردود
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    edited_at: Optional[datetime] = None
    edit_count: int = 0
    
    @property
    def score(self) -> int:
        """النقاط (upvotes - downvotes)"""
        return self.upvotes - self.downvotes
    
    def upvote(self):
        """تصويت إيجابي"""
        self.upvotes += 1
    
    def downvote(self):
        """تصويت سلبي"""
        self.downvotes += 1
    
    def edit(self, new_content: str):
        """تعديل المشاركة"""
        self.content = new_content
        self.edited_at = datetime.now(timezone.utc)
        self.edit_count += 1
        self.updated_at = datetime.now(timezone.utc)
    
    def mark_as_solution(self):
        """تحديد كحل"""
        self.is_solution = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic_id": self.topic_id,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "content": self.content,
            "status": self.status.value,
            "is_solution": self.is_solution,
            "upvotes": self.upvotes,
            "downvotes": self.downvotes,
            "score": self.score,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "edited_at": self.edited_at.isoformat() if self.edited_at else None
        }


@dataclass
class Topic:
    """موضوع"""
    id: str
    forum_id: str
    title: str
    author_id: str
    author_name: str
    
    # Content
    posts: List[Post] = field(default_factory=list)
    
    # Status
    status: TopicStatus = TopicStatus.OPEN
    is_locked: bool = False
    
    # Statistics
    view_count: int = 0
    
    # Tags
    tags: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_post_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_post_by: Optional[str] = None
    
    @property
    def reply_count(self) -> int:
        """عدد الردود (بدون المشاركة الأولى)"""
        return max(0, len(self.posts) - 1)
    
    @property
    def has_solution(self) -> bool:
        """هل يوجد حل؟"""
        return any(p.is_solution for p in self.posts)
    
    def add_post(self, author_id: str, author_name: str, 
                content: str, parent_id: str = None) -> Post:
        """إضافة مشاركة"""
        post = Post(
            id=str(uuid.uuid4()),
            topic_id=self.id,
            author_id=author_id,
            author_name=author_name,
            content=content,
            parent_id=parent_id
        )
        
        self.posts.append(post)
        self.last_post_at = datetime.now(timezone.utc)
        self.last_post_by = author_name
        self.updated_at = datetime.now(timezone.utc)
        
        return post
    
    def increment_views(self):
        """زيادة عدد المشاهدات"""
        self.view_count += 1
    
    def close(self):
        """إغلاق الموضوع"""
        self.status = TopicStatus.CLOSED
        self.is_locked = True
        self.updated_at = datetime.now(timezone.utc)
    
    def pin(self):
        """تثبيت الموضوع"""
        self.status = TopicStatus.PINNED
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "forum_id": self.forum_id,
            "title": self.title,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "status": self.status.value,
            "is_locked": self.is_locked,
            "reply_count": self.reply_count,
            "view_count": self.view_count,
            "has_solution": self.has_solution,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "last_post_at": self.last_post_at.isoformat(),
            "last_post_by": self.last_post_by
        }


@dataclass
class Forum:
    """منتدى"""
    id: str
    name: str
    description: str = ""
    
    # Hierarchy
    parent_id: Optional[str] = None   # المنتدى الأب
    
    # Topics
    topics: List[Topic] = field(default_factory=list)
    
    # Settings
    is_public: bool = True
    require_approval: bool = False    # هل المشاركات تحتاج موافقة؟
    
    # Moderators
    moderator_ids: List[str] = field(default_factory=list)
    
    # Statistics
    topic_count: int = 0
    post_count: int = 0
    
    # Display
    icon: str = ""
    color: str = "#2196F3"
    display_order: int = 0
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def last_topic(self) -> Optional[Topic]:
        """آخر موضوع"""
        if not self.topics:
            return None
        return max(self.topics, key=lambda t: t.last_post_at)
    
    def add_topic(self, title: str, author_id: str, author_name: str,
                 content: str, tags: List[str] = None) -> Topic:
        """إضافة موضوع"""
        topic = Topic(
            id=str(uuid.uuid4()),
            forum_id=self.id,
            title=title,
            author_id=author_id,
            author_name=author_name,
            tags=tags or []
        )
        
        # Add first post
        topic.add_post(author_id, author_name, content)
        
        self.topics.append(topic)
        self.topic_count += 1
        self.post_count += 1
        
        return topic
    
    def is_moderator(self, user_id: str) -> bool:
        """هل المستخدم مشرف؟"""
        return user_id in self.moderator_ids
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parent_id": self.parent_id,
            "is_public": self.is_public,
            "topic_count": self.topic_count,
            "post_count": self.post_count,
            "icon": self.icon,
            "color": self.color,
            "display_order": self.display_order,
            "last_topic": self.last_topic.to_dict() if self.last_topic else None,
            "created_at": self.created_at.isoformat()
        }


class ForumManager:
    """
    مدير المنتديات
    """
    
    def __init__(self):
        self.forums: Dict[str, Forum] = {}
        self.notifications: Dict[str, List[Dict]] = {}  # user_id -> notifications
    
    def create_forum(self, name: str, description: str = "",
                    parent_id: str = None, is_public: bool = True) -> Forum:
        """إنشاء منتدى جديد"""
        forum = Forum(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            parent_id=parent_id,
            is_public=is_public
        )
        
        self.forums[forum.id] = forum
        return forum
    
    def get_forum(self, forum_id: str) -> Optional[Forum]:
        """الحصول على منتدى"""
        return self.forums.get(forum_id)
    
    def create_topic(self, forum_id: str, title: str, content: str,
                    author_id: str, author_name: str,
                    tags: List[str] = None) -> Topic:
        """إنشاء موضوع"""
        forum = self.forums.get(forum_id)
        if not forum:
            raise ValueError(f"Forum {forum_id} not found")
        
        topic = forum.add_topic(title, author_id, author_name, content, tags)
        
        # Notify moderators
        self._notify_moderators(forum, topic)
        
        return topic
    
    def reply_to_topic(self, topic_id: str, content: str,
                      author_id: str, author_name: str,
                      forum_id: str = None) -> Post:
        """الرد على موضوع"""
        # Find topic
        topic = None
        forum = None
        
        if forum_id:
            forum = self.forums.get(forum_id)
            if forum:
                topic = next((t for t in forum.topics if t.id == topic_id), None)
        else:
            for f in self.forums.values():
                topic = next((t for t in f.topics if t.id == topic_id), None)
                if topic:
                    forum = f
                    break
        
        if not topic:
            raise ValueError(f"Topic {topic_id} not found")
        
        post = topic.add_post(author_id, author_name, content)
        
        if forum:
            forum.post_count += 1
        
        # Notify topic author
        if topic.author_id != author_id:
            self._notify_user(topic.author_id, "new_reply", 
                            f"New reply on your topic: {topic.title}",
                            topic_id)
        
        return post
    
    def _notify_moderators(self, forum: Forum, topic: Topic):
        """إشعار المشرفين"""
        for mod_id in forum.moderator_ids:
            self._notify_user(mod_id, "new_topic",
                            f"New topic in {forum.name}: {topic.title}",
                            topic.id)
    
    def _notify_user(self, user_id: str, notification_type: str, 
                    message: str, reference_id: str):
        """إضافة إشعار للمستخدم"""
        if user_id not in self.notifications:
            self.notifications[user_id] = []
        
        self.notifications[user_id].append({
            "id": str(uuid.uuid4()),
            "type": notification_type,
            "message": message,
            "reference_id": reference_id,
            "read": False,
            "created_at": datetime.now(timezone.utc).isoformat()
        })
    
    def search(self, query: str, forum_id: str = None) -> List[Dict]:
        """البحث في المنتديات"""
        results = []
        query = query.lower()
        
        forums_to_search = [self.forums.get(forum_id)] if forum_id else self.forums.values()
        
        for forum in forums_to_search:
            if not forum:
                continue
            
            for topic in forum.topics:
                if (query in topic.title.lower() or
                    any(query in p.content.lower() for p in topic.posts)):
                    results.append({
                        "type": "topic",
                        "forum": forum.name,
                        "topic": topic.to_dict()
                    })
        
        return results
    
    def moderate_post(self, post_id: str, action: str, moderator_id: str):
        """إشراف على مشاركة"""
        # Find post
        for forum in self.forums.values():
            for topic in forum.topics:
                for post in topic.posts:
                    if post.id == post_id:
                        if action == "hide":
                            post.status = PostStatus.HIDDEN
                        elif action == "delete":
                            post.status = PostStatus.DELETED
                        elif action == "flag":
                            post.status = PostStatus.FLAGGED
                        return post
        
        return None
    
    def get_forum_tree(self) -> Dict[str, Any]:
        """الحصول على شجرة المنتديات"""
        tree = {}
        
        for forum in self.forums.values():
            if forum.parent_id is None:
                tree[forum.id] = self._build_forum_node(forum)
        
        return tree
    
    def _build_forum_node(self, forum: Forum) -> Dict[str, Any]:
        """بناء عقدة منتدى"""
        node = {
            "forum": forum.to_dict(),
            "children": {}
        }
        
        for child in self.forums.values():
            if child.parent_id == forum.id:
                node["children"][child.id] = self._build_forum_node(child)
        
        return node
    
    def get_recent_topics(self, limit: int = 20) -> List[Dict]:
        """الحصول على المواضيع الأخيرة"""
        all_topics = []
        
        for forum in self.forums.values():
            for topic in forum.topics:
                all_topics.append({
                    "topic": topic,
                    "forum_name": forum.name
                })
        
        # Sort by last post
        all_topics.sort(key=lambda x: x["topic"].last_post_at, reverse=True)
        
        return [
            {**t["topic"].to_dict(), "forum_name": t["forum_name"]}
            for t in all_topics[:limit]
        ]
    
    def get_unread_notifications(self, user_id: str) -> List[Dict]:
        """الحصول على الإشعارات غير المقروءة"""
        notifications = self.notifications.get(user_id, [])
        return [n for n in notifications if not n["read"]]
    
    def mark_notification_read(self, user_id: str, notification_id: str):
        """تحديد إشعار كمقروء"""
        notifications = self.notifications.get(user_id, [])
        for n in notifications:
            if n["id"] == notification_id:
                n["read"] = True
                break
