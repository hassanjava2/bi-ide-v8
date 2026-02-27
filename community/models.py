"""
Community Models - نماذج قاعدة البيانات للمجتمع
SQLAlchemy models for community features
"""
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Column, String, Text, DateTime, Integer, Boolean, 
    ForeignKey, Table, Enum as SQLEnum, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.types import JSON

from core.database import Base
import enum


# Forums
class TopicStatus(str, enum.Enum):
    """حالة الموضوع"""
    OPEN = "open"
    CLOSED = "closed"
    PINNED = "pinned"
    ANNOUNCEMENT = "announcement"


class PostStatus(str, enum.Enum):
    """حالة المشاركة"""
    VISIBLE = "visible"
    HIDDEN = "hidden"
    DELETED = "deleted"
    FLAGGED = "flagged"


class ForumCategoryDB(Base):
    """فئة المنتديات"""
    __tablename__ = "forum_categories"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    description = Column(Text)
    display_order = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    forums = relationship("ForumDB", back_populates="category")


class ForumDB(Base):
    """منتدى"""
    __tablename__ = "forums"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    category_id = Column(String, ForeignKey("forum_categories.id"), nullable=True)
    
    name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Settings
    is_public = Column(Boolean, default=True)
    require_approval = Column(Boolean, default=False)
    allow_guests = Column(Boolean, default=False)
    
    # Display
    icon = Column(String(100), default="")
    color = Column(String(20), default="#2196F3")
    display_order = Column(Integer, default=0)
    
    # Statistics
    topic_count = Column(Integer, default=0)
    post_count = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    category = relationship("ForumCategoryDB", back_populates="forums")
    topics = relationship("TopicDB", back_populates="forum", cascade="all, delete-orphan")
    moderators = relationship("ForumModeratorDB", back_populates="forum", cascade="all, delete-orphan")


class ForumModeratorDB(Base):
    """مشرف المنتدى"""
    __tablename__ = "forum_moderators"
    
    forum_id = Column(String, ForeignKey("forums.id"), primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    assigned_by = Column(String, ForeignKey("users.id"))
    assigned_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    forum = relationship("ForumDB", back_populates="moderators")


class TopicDB(Base):
    """موضوع"""
    __tablename__ = "topics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    forum_id = Column(String, ForeignKey("forums.id"), nullable=False)
    
    title = Column(String(300), nullable=False)
    slug = Column(String(350), unique=True, index=True)
    
    author_id = Column(String, ForeignKey("users.id"), nullable=False)
    author_name = Column(String(100))
    
    # Content (first post)
    content = Column(Text, nullable=False)
    content_html = Column(Text)  # Rendered HTML
    
    # Status
    status = Column(SQLEnum(TopicStatus), default=TopicStatus.OPEN)
    is_locked = Column(Boolean, default=False)
    is_pinned = Column(Boolean, default=False)
    is_announcement = Column(Boolean, default=False)
    
    # Statistics
    view_count = Column(Integer, default=0)
    reply_count = Column(Integer, default=0)
    
    # Solution
    has_solution = Column(Boolean, default=False)
    solution_post_id = Column(String, ForeignKey("posts.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
    last_post_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_post_by = Column(String(100))
    
    # Relationships
    forum = relationship("ForumDB", back_populates="topics")
    posts = relationship("PostDB", back_populates="topic", cascade="all, delete-orphan", foreign_keys="PostDB.topic_id")
    tags = relationship("TopicTagDB", back_populates="topic", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_topic_forum_created', 'forum_id', 'created_at'),
        Index('idx_topic_last_post', 'forum_id', 'last_post_at'),
    )


class PostDB(Base):
    """مشاركة"""
    __tablename__ = "posts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    topic_id = Column(String, ForeignKey("topics.id"), nullable=False)
    
    author_id = Column(String, ForeignKey("users.id"), nullable=False)
    author_name = Column(String(100))
    
    content = Column(Text, nullable=False)
    content_html = Column(Text)
    
    # Status
    status = Column(SQLEnum(PostStatus), default=PostStatus.VISIBLE)
    
    # Solution
    is_solution = Column(Boolean, default=False)
    
    # Voting
    upvotes = Column(Integer, default=0)
    downvotes = Column(Integer, default=0)
    
    # Replies (nested)
    parent_id = Column(String, ForeignKey("posts.id"), nullable=True)
    
    # Edit tracking
    edit_count = Column(Integer, default=0)
    edited_at = Column(DateTime(timezone=True))
    edit_reason = Column(String(200))
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    topic = relationship("TopicDB", back_populates="posts", foreign_keys=[topic_id])
    replies = relationship("PostDB", back_populates="parent", remote_side=[id])
    parent = relationship("PostDB", back_populates="replies", remote_side=[parent_id])


class TopicTagDB(Base):
    """وسم الموضوع"""
    __tablename__ = "topic_tags"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    topic_id = Column(String, ForeignKey("topics.id"), nullable=False)
    tag = Column(String(50), nullable=False, index=True)
    
    # Relationships
    topic = relationship("TopicDB", back_populates="tags")


# Knowledge Base
class KBCategoryDB(Base):
    """فئة قاعدة المعرفة"""
    __tablename__ = "kb_categories"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    description = Column(Text)
    icon = Column(String(100), default="")
    color = Column(String(20), default="#4CAF50")
    
    # Hierarchy
    parent_id = Column(String, ForeignKey("kb_categories.id"), nullable=True)
    
    display_order = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    parent = relationship("KBCategoryDB", remote_side=[id])
    articles = relationship("KBArticleDB", back_populates="category")


class KBArticleDB(Base):
    """مقالة قاعدة المعرفة"""
    __tablename__ = "kb_articles"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    category_id = Column(String, ForeignKey("kb_categories.id"), nullable=False)
    
    title = Column(String(300), nullable=False)
    slug = Column(String(350), unique=True, index=True)
    
    content = Column(Text, nullable=False)
    content_html = Column(Text)
    
    # Metadata
    author_id = Column(String, ForeignKey("users.id"), nullable=False)
    author_name = Column(String(100))
    
    # Status
    status = Column(String(20), default="published")  # draft, published, archived
    is_featured = Column(Boolean, default=False)
    
    # Statistics
    view_count = Column(Integer, default=0)
    helpful_count = Column(Integer, default=0)
    not_helpful_count = Column(Integer, default=0)
    
    # Tags
    tags = Column(JSON, default=list)
    
    # SEO
    meta_description = Column(Text)
    keywords = Column(JSON, default=list)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
    published_at = Column(DateTime(timezone=True))
    
    # Relationships
    category = relationship("KBCategoryDB", back_populates="articles")


# Code Sharing
class CodeSnippetDB(Base):
    """قطعة كود"""
    __tablename__ = "code_snippets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    title = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Code
    code = Column(Text, nullable=False)
    language = Column(String(50), nullable=False, index=True)
    
    # Metadata
    author_id = Column(String, ForeignKey("users.id"), nullable=False)
    author_name = Column(String(100))
    
    # Statistics
    view_count = Column(Integer, default=0)
    download_count = Column(Integer, default=0)
    
    # Status
    is_public = Column(Boolean, default=True)
    is_featured = Column(Boolean, default=False)
    
    # Tags
    tags = Column(JSON, default=list)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))


# User Profiles & Reputation
class UserProfileStatsDB(Base):
    """إحصائيات الملف الشخصي"""
    __tablename__ = "user_profile_stats"
    
    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    
    # Reputation
    reputation = Column(Integer, default=0)
    
    # Forum activity
    forum_posts = Column(Integer, default=0)
    forum_topics = Column(Integer, default=0)
    
    # Knowledge base
    kb_articles = Column(Integer, default=0)
    kb_helpful = Column(Integer, default=0)
    
    # Code sharing
    code_snippets = Column(Integer, default=0)
    code_downloads = Column(Integer, default=0)
    
    # Badges
    badges = Column(JSON, default=list)
    
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))


class ReputationHistoryDB(Base):
    """تاريخ السمعة"""
    __tablename__ = "reputation_history"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    amount = Column(Integer, nullable=False)
    reason = Column(String(200))
    source_type = Column(String(50))  # post, article, snippet, etc.
    source_id = Column(String)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# Notifications
class UserNotificationDB(Base):
    """إشعار المستخدم"""
    __tablename__ = "user_notifications"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    type = Column(String(50), nullable=False)  # new_reply, new_topic, mention, etc.
    title = Column(String(200), nullable=False)
    message = Column(Text)
    
    # Reference
    reference_type = Column(String(50))  # topic, post, article
    reference_id = Column(String)
    
    # Status
    is_read = Column(Boolean, default=False)
    read_at = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Index for fetching unread notifications
    __table_args__ = (
        Index('idx_notifications_user_read', 'user_id', 'is_read'),
    )
