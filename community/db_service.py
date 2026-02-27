"""
Community Database Service - خدمة قاعدة بيانات المجتمع
"""
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy import select, update, delete, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from community.models import (
    ForumCategoryDB, ForumDB, ForumModeratorDB,
    TopicDB, PostDB, TopicTagDB, TopicStatus, PostStatus,
    KBCategoryDB, KBArticleDB,
    CodeSnippetDB,
    UserProfileStatsDB, ReputationHistoryDB,
    UserNotificationDB
)


class CommunityDBService:
    """خدمة قاعدة بيانات المجتمع"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    # ==================== Forums ====================
    
    async def create_category(self, name: str, description: str = "", 
                             display_order: int = 0) -> ForumCategoryDB:
        """إنشاء فئة منتديات"""
        category = ForumCategoryDB(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            display_order=display_order
        )
        self.session.add(category)
        await self.session.commit()
        return category
    
    async def get_categories(self, include_empty: bool = False) -> List[ForumCategoryDB]:
        """جلب جميع الفئات"""
        result = await self.session.execute(
            select(ForumCategoryDB)
            .where(ForumCategoryDB.is_active == True)
            .order_by(ForumCategoryDB.display_order)
        )
        categories = result.scalars().all()
        
        if not include_empty:
            # Load forums for each category
            for cat in categories:
                await self.session.refresh(cat, ['forums'])
        
        return list(categories)
    
    async def create_forum(self, name: str, description: str = "",
                          category_id: str = None, is_public: bool = True,
                          icon: str = "", color: str = "#2196F3") -> ForumDB:
        """إنشاء منتدى"""
        forum = ForumDB(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            category_id=category_id,
            is_public=is_public,
            icon=icon,
            color=color
        )
        self.session.add(forum)
        await self.session.commit()
        return forum
    
    async def get_forums(self, category_id: str = None) -> List[ForumDB]:
        """جلب المنتديات"""
        query = select(ForumDB).where(ForumDB.is_public == True)
        
        if category_id:
            query = query.where(ForumDB.category_id == category_id)
        
        query = query.order_by(ForumDB.display_order)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_forum(self, forum_id: str) -> Optional[ForumDB]:
        """جلب منتدى بالـ ID"""
        result = await self.session.execute(
            select(ForumDB).where(ForumDB.id == forum_id)
        )
        return result.scalar_one_or_none()
    
    # ==================== Topics ====================
    
    async def create_topic(self, forum_id: str, title: str, content: str,
                          author_id: str, author_name: str,
                          tags: List[str] = None) -> TopicDB:
        """إنشاء موضوع"""
        topic = TopicDB(
            id=str(uuid.uuid4()),
            forum_id=forum_id,
            title=title,
            slug=self._create_slug(title),
            content=content,
            author_id=author_id,
            author_name=author_name,
            status=TopicStatus.OPEN
        )
        
        self.session.add(topic)
        
        # Update forum stats
        await self.session.execute(
            update(ForumDB)
            .where(ForumDB.id == forum_id)
            .values(
                topic_count=ForumDB.topic_count + 1,
                post_count=ForumDB.post_count + 1,
                updated_at=datetime.now(timezone.utc)
            )
        )
        
        # Add tags
        if tags:
            for tag in tags:
                topic_tag = TopicTagDB(
                    id=str(uuid.uuid4()),
                    topic_id=topic.id,
                    tag=tag.lower()
                )
                self.session.add(topic_tag)
        
        # Update user stats
        await self._increment_user_stat(author_id, 'forum_topics')
        
        await self.session.commit()
        return topic
    
    async def get_topics(self, forum_id: str, page: int = 1, 
                        per_page: int = 20) -> Dict[str, Any]:
        """جلب مواضيع منتدى"""
        offset = (page - 1) * per_page
        
        # Get topics
        result = await self.session.execute(
            select(TopicDB)
            .where(TopicDB.forum_id == forum_id)
            .where(TopicDB.status != TopicStatus.CLOSED)
            .order_by(desc(TopicDB.is_pinned), desc(TopicDB.last_post_at))
            .offset(offset)
            .limit(per_page)
        )
        topics = result.scalars().all()
        
        # Get total count
        count_result = await self.session.execute(
            select(func.count(TopicDB.id))
            .where(TopicDB.forum_id == forum_id)
            .where(TopicDB.status != TopicStatus.CLOSED)
        )
        total = count_result.scalar()
        
        return {
            "items": list(topics),
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page
        }
    
    async def get_topic(self, topic_id: str) -> Optional[TopicDB]:
        """جلب موضوع بالـ ID"""
        result = await self.session.execute(
            select(TopicDB).where(TopicDB.id == topic_id)
        )
        topic = result.scalar_one_or_none()
        
        if topic:
            # Increment view count
            topic.view_count += 1
            await self.session.commit()
        
        return topic
    
    async def create_post(self, topic_id: str, content: str,
                         author_id: str, author_name: str,
                         parent_id: str = None) -> PostDB:
        """إنشاء مشاركة"""
        post = PostDB(
            id=str(uuid.uuid4()),
            topic_id=topic_id,
            author_id=author_id,
            author_name=author_name,
            content=content,
            parent_id=parent_id
        )
        
        self.session.add(post)
        
        # Update topic stats
        await self.session.execute(
            update(TopicDB)
            .where(TopicDB.id == topic_id)
            .values(
                reply_count=TopicDB.reply_count + 1,
                last_post_at=datetime.now(timezone.utc),
                last_post_by=author_name,
                updated_at=datetime.now(timezone.utc)
            )
        )
        
        # Get forum_id for the topic
        topic_result = await self.session.execute(
            select(TopicDB.forum_id).where(TopicDB.id == topic_id)
        )
        forum_id = topic_result.scalar()
        
        # Update forum stats
        if forum_id:
            await self.session.execute(
                update(ForumDB)
                .where(ForumDB.id == forum_id)
                .values(
                    post_count=ForumDB.post_count + 1,
                    updated_at=datetime.now(timezone.utc)
                )
            )
        
        # Update user stats
        await self._increment_user_stat(author_id, 'forum_posts')
        
        # Add reputation
        await self._add_reputation(author_id, 1, "New forum post", "post", post.id)
        
        await self.session.commit()
        return post
    
    async def get_posts(self, topic_id: str, page: int = 1,
                       per_page: int = 20) -> Dict[str, Any]:
        """جلب مشاركات موضوع"""
        offset = (page - 1) * per_page
        
        result = await self.session.execute(
            select(PostDB)
            .where(PostDB.topic_id == topic_id)
            .where(PostDB.status == PostStatus.VISIBLE)
            .where(PostDB.parent_id == None)  # Top-level posts only
            .order_by(PostDB.created_at)
            .offset(offset)
            .limit(per_page)
        )
        posts = result.scalars().all()
        
        # Get total
        count_result = await self.session.execute(
            select(func.count(PostDB.id))
            .where(PostDB.topic_id == topic_id)
            .where(PostDB.status == PostStatus.VISIBLE)
            .where(PostDB.parent_id == None)
        )
        total = count_result.scalar()
        
        return {
            "items": list(posts),
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page
        }
    
    # ==================== Knowledge Base ====================
    
    async def create_kb_category(self, name: str, description: str = "",
                                 parent_id: str = None) -> KBCategoryDB:
        """إنشاء فئة KB"""
        category = KBCategoryDB(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            parent_id=parent_id
        )
        self.session.add(category)
        await self.session.commit()
        return category
    
    async def get_kb_categories(self) -> List[KBCategoryDB]:
        """جلب فئات KB"""
        result = await self.session.execute(
            select(KBCategoryDB)
            .where(KBCategoryDB.is_active == True)
            .order_by(KBCategoryDB.display_order)
        )
        return list(result.scalars().all())
    
    async def create_kb_article(self, category_id: str, title: str, content: str,
                               author_id: str, author_name: str,
                               tags: List[str] = None) -> KBArticleDB:
        """إنشاء مقالة KB"""
        article = KBArticleDB(
            id=str(uuid.uuid4()),
            category_id=category_id,
            title=title,
            slug=self._create_slug(title),
            content=content,
            author_id=author_id,
            author_name=author_name,
            tags=tags or [],
            status="published",
            published_at=datetime.now(timezone.utc)
        )
        
        self.session.add(article)
        
        # Update user stats
        await self._increment_user_stat(author_id, 'kb_articles')
        
        # Add reputation
        await self._add_reputation(author_id, 5, "New KB article", "article", article.id)
        
        await self.session.commit()
        return article
    
    async def get_kb_articles(self, category_id: str = None, 
                             page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """جلب مقالات KB"""
        offset = (page - 1) * per_page
        
        query = select(KBArticleDB).where(KBArticleDB.status == "published")
        
        if category_id:
            query = query.where(KBArticleDB.category_id == category_id)
        
        query = query.order_by(desc(KBArticleDB.is_featured), 
                               desc(KBArticleDB.created_at))
        query = query.offset(offset).limit(per_page)
        
        result = await self.session.execute(query)
        articles = result.scalars().all()
        
        # Count
        count_query = select(func.count(KBArticleDB.id)).where(KBArticleDB.status == "published")
        if category_id:
            count_query = count_query.where(KBArticleDB.category_id == category_id)
        count_result = await self.session.execute(count_query)
        total = count_result.scalar()
        
        return {
            "items": list(articles),
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page
        }
    
    async def get_kb_article(self, article_id: str) -> Optional[KBArticleDB]:
        """جلب مقالة بالـ ID"""
        result = await self.session.execute(
            select(KBArticleDB).where(KBArticleDB.id == article_id)
        )
        article = result.scalar_one_or_none()
        
        if article:
            article.view_count += 1
            await self.session.commit()
        
        return article
    
    # ==================== Code Sharing ====================
    
    async def create_code_snippet(self, title: str, code: str, language: str,
                                 author_id: str, author_name: str,
                                 description: str = "", tags: List[str] = None) -> CodeSnippetDB:
        """إنشاء قطعة كود"""
        snippet = CodeSnippetDB(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            code=code,
            language=language.lower(),
            author_id=author_id,
            author_name=author_name,
            tags=tags or []
        )
        
        self.session.add(snippet)
        
        # Update user stats
        await self._increment_user_stat(author_id, 'code_snippets')
        
        # Add reputation
        await self._add_reputation(author_id, 3, "New code snippet", "snippet", snippet.id)
        
        await self.session.commit()
        return snippet
    
    async def get_code_snippets(self, language: str = None, 
                               page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """جلب قطع الكود"""
        offset = (page - 1) * per_page
        
        query = select(CodeSnippetDB).where(CodeSnippetDB.is_public == True)
        
        if language:
            query = query.where(CodeSnippetDB.language == language.lower())
        
        query = query.order_by(desc(CodeSnippetDB.is_featured), 
                               desc(CodeSnippetDB.created_at))
        query = query.offset(offset).limit(per_page)
        
        result = await self.session.execute(query)
        snippets = result.scalars().all()
        
        # Count
        count_result = await self.session.execute(
            select(func.count(CodeSnippetDB.id))
            .where(CodeSnippetDB.is_public == True)
        )
        total = count_result.scalar()
        
        return {
            "items": list(snippets),
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page
        }
    
    async def get_code_snippet(self, snippet_id: str) -> Optional[CodeSnippetDB]:
        """جلب قطعة كود بالـ ID"""
        result = await self.session.execute(
            select(CodeSnippetDB).where(CodeSnippetDB.id == snippet_id)
        )
        snippet = result.scalar_one_or_none()
        
        if snippet:
            snippet.view_count += 1
            await self.session.commit()
        
        return snippet
    
    # ==================== User Profiles ====================
    
    async def get_user_stats(self, user_id: str) -> Optional[UserProfileStatsDB]:
        """جلب إحصائيات المستخدم"""
        result = await self.session.execute(
            select(UserProfileStatsDB).where(UserProfileStatsDB.user_id == user_id)
        )
        stats = result.scalar_one_or_none()
        
        if not stats:
            # Create default stats
            stats = UserProfileStatsDB(user_id=user_id)
            self.session.add(stats)
            await self.session.commit()
        
        return stats
    
    async def get_user_reputation_history(self, user_id: str, 
                                         page: int = 1, 
                                         per_page: int = 20) -> Dict[str, Any]:
        """جلب تاريخ السمعة"""
        offset = (page - 1) * per_page
        
        result = await self.session.execute(
            select(ReputationHistoryDB)
            .where(ReputationHistoryDB.user_id == user_id)
            .order_by(desc(ReputationHistoryDB.created_at))
            .offset(offset)
            .limit(per_page)
        )
        items = result.scalars().all()
        
        count_result = await self.session.execute(
            select(func.count(ReputationHistoryDB.id))
            .where(ReputationHistoryDB.user_id == user_id)
        )
        total = count_result.scalar()
        
        return {
            "items": list(items),
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page
        }
    
    # ==================== Notifications ====================
    
    async def create_notification(self, user_id: str, notification_type: str,
                                  title: str, message: str = "",
                                  reference_type: str = None, 
                                  reference_id: str = None) -> UserNotificationDB:
        """إنشاء إشعار"""
        notification = UserNotificationDB(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=notification_type,
            title=title,
            message=message,
            reference_type=reference_type,
            reference_id=reference_id
        )
        
        self.session.add(notification)
        await self.session.commit()
        return notification
    
    async def get_notifications(self, user_id: str, unread_only: bool = False,
                               page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """جلب إشعارات المستخدم"""
        offset = (page - 1) * per_page
        
        query = select(UserNotificationDB).where(UserNotificationDB.user_id == user_id)
        
        if unread_only:
            query = query.where(UserNotificationDB.is_read == False)
        
        query = query.order_by(desc(UserNotificationDB.created_at))
        query = query.offset(offset).limit(per_page)
        
        result = await self.session.execute(query)
        items = result.scalars().all()
        
        # Count
        count_query = select(func.count(UserNotificationDB.id)).where(UserNotificationDB.user_id == user_id)
        if unread_only:
            count_query = count_query.where(UserNotificationDB.is_read == False)
        count_result = await self.session.execute(count_query)
        total = count_result.scalar()
        
        return {
            "items": list(items),
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page
        }
    
    async def mark_notification_read(self, notification_id: str) -> bool:
        """تحديد إشعار كمقروء"""
        result = await self.session.execute(
            update(UserNotificationDB)
            .where(UserNotificationDB.id == notification_id)
            .values(is_read=True, read_at=datetime.now(timezone.utc))
        )
        await self.session.commit()
        return result.rowcount > 0
    
    # ==================== Helper Methods ====================
    
    def _create_slug(self, text: str) -> str:
        """إنشاء slug من النص"""
        import re
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug[:100]  # Limit length
    
    async def _increment_user_stat(self, user_id: str, field: str):
        """زيادة إحصائية المستخدم"""
        # Get or create stats
        result = await self.session.execute(
            select(UserProfileStatsDB).where(UserProfileStatsDB.user_id == user_id)
        )
        stats = result.scalar_one_or_none()
        
        if not stats:
            stats = UserProfileStatsDB(user_id=user_id)
            self.session.add(stats)
        
        # Increment field
        current = getattr(stats, field, 0)
        setattr(stats, field, current + 1)
        
        await self.session.commit()
    
    async def _add_reputation(self, user_id: str, amount: int, reason: str,
                              source_type: str = None, source_id: str = None):
        """إضافة سمعة للمستخدم"""
        # Add history entry
        history = ReputationHistoryDB(
            id=str(uuid.uuid4()),
            user_id=user_id,
            amount=amount,
            reason=reason,
            source_type=source_type,
            source_id=source_id
        )
        self.session.add(history)
        
        # Update total
        result = await self.session.execute(
            select(UserProfileStatsDB).where(UserProfileStatsDB.user_id == user_id)
        )
        stats = result.scalar_one_or_none()
        
        if stats:
            stats.reputation += amount
        else:
            stats = UserProfileStatsDB(user_id=user_id, reputation=amount)
            self.session.add(stats)
        
        await self.session.commit()
