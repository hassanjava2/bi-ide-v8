"""
Knowledge Base - قاعدة المعرفة

المميزات:
- المقالات مع الإصدارات
- الفئات والوسوم
- نظام التقييم
- المقالات ذات الصلة
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class ArticleStatus(Enum):
    """حالة المقالة"""
    DRAFT = "draft"                   # مسودة
    UNDER_REVIEW = "under_review"     # قيد المراجعة
    PUBLISHED = "published"           # منشورة
    ARCHIVED = "archived"             # مؤرشفة
    OUTDATED = "outdated"             # قديمة


@dataclass
class ArticleVersion:
    """إصدار من المقالة"""
    version_number: int
    content: str
    changes_summary: str = ""         # ملخص التغييرات
    created_by: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_number": self.version_number,
            "changes_summary": self.changes_summary,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Article:
    """مقالة"""
    id: str
    slug: str                         # الرابط
    title: str
    content: str = ""
    summary: str = ""                 # ملخص
    
    # Categorization
    category_id: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Authoring
    author_id: str = ""
    author_name: str = ""
    contributors: List[str] = field(default_factory=list)
    
    # Status
    status: ArticleStatus = ArticleStatus.DRAFT
    
    # Versions
    versions: List[ArticleVersion] = field(default_factory=list)
    current_version: int = 1
    
    # Rating
    rating_sum: int = 0
    rating_count: int = 0
    
    # Statistics
    view_count: int = 0
    helpful_count: int = 0
    not_helpful_count: int = 0
    
    # Related
    related_article_ids: List[str] = field(default_factory=list)
    
    # SEO
    meta_title: str = ""
    meta_description: str = ""
    keywords: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    published_at: Optional[datetime] = None
    
    @property
    def rating(self) -> float:
        """التقييم (من 5)"""
        if self.rating_count == 0:
            return 0
        return round(self.rating_sum / self.rating_count, 1)
    
    @property
    def helpfulness(self) -> float:
        """نسبة الإفادة"""
        total = self.helpful_count + self.not_helpful_count
        if total == 0:
            return 0
        return (self.helpful_count / total) * 100
    
    def publish(self):
        """نشر المقالة"""
        self.status = ArticleStatus.PUBLISHED
        self.published_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def archive(self):
        """أرشفة المقالة"""
        self.status = ArticleStatus.ARCHIVED
        self.updated_at = datetime.now(timezone.utc)
    
    def update_content(self, new_content: str, changes_summary: str,
                      updated_by: str):
        """تحديث محتوى المقالة (إنشاء إصدار جديد)"""
        # Save current version
        old_version = ArticleVersion(
            version_number=self.current_version,
            content=self.content,
            changes_summary="Previous version",
            created_by=self.author_id
        )
        self.versions.append(old_version)
        
        # Update
        self.content = new_content
        self.current_version += 1
        self.updated_at = datetime.now(timezone.utc)
        
        if updated_by not in self.contributors:
            self.contributors.append(updated_by)
        
        # Add new version record
        new_version = ArticleVersion(
            version_number=self.current_version,
            content=new_content,
            changes_summary=changes_summary,
            created_by=updated_by
        )
        self.versions.append(new_version)
    
    def rate(self, rating: int):
        """تقييم المقالة"""
        if 1 <= rating <= 5:
            self.rating_sum += rating
            self.rating_count += 1
    
    def mark_helpful(self, was_helpful: bool = True):
        """تحديد ما إذا كانت المقالة مفيدة"""
        if was_helpful:
            self.helpful_count += 1
        else:
            self.not_helpful_count += 1
    
    def increment_views(self):
        """زيادة عدد المشاهدات"""
        self.view_count += 1
    
    def add_related_article(self, article_id: str):
        """إضافة مقالة ذات صلة"""
        if article_id not in self.related_article_ids:
            self.related_article_ids.append(article_id)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "slug": self.slug,
            "title": self.title,
            "summary": self.summary,
            "category_id": self.category_id,
            "tags": self.tags,
            "author_name": self.author_name,
            "status": self.status.value,
            "current_version": self.current_version,
            "rating": self.rating,
            "rating_count": self.rating_count,
            "view_count": self.view_count,
            "helpfulness": self.helpfulness,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "published_at": self.published_at.isoformat() if self.published_at else None
        }
    
    def to_detail_dict(self) -> Dict[str, Any]:
        """تفاصيل كاملة"""
        result = self.to_dict()
        result["content"] = self.content
        result["versions"] = [v.to_dict() for v in self.versions]
        result["related_article_ids"] = self.related_article_ids
        return result


@dataclass
class ArticleCategory:
    """فئة مقالات"""
    id: str
    name: str
    slug: str
    description: str = ""
    parent_id: Optional[str] = None   # الفئة الأب
    icon: str = ""
    color: str = "#2196F3"
    display_order: int = 0
    article_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "parent_id": self.parent_id,
            "icon": self.icon,
            "color": self.color,
            "display_order": self.display_order,
            "article_count": self.article_count
        }


class KnowledgeBase:
    """
    قاعدة المعرفة
    """
    
    def __init__(self):
        self.articles: Dict[str, Article] = {}
        self.categories: Dict[str, ArticleCategory] = {}
        self.article_ratings: Dict[str, Dict[str, int]] = {}  # article_id -> {user_id: rating}
    
    def create_category(self, name: str, slug: str, description: str = "",
                       parent_id: str = None, icon: str = "") -> ArticleCategory:
        """إنشاء فئة"""
        category = ArticleCategory(
            id=str(uuid.uuid4()),
            name=name,
            slug=slug,
            description=description,
            parent_id=parent_id,
            icon=icon
        )
        
        self.categories[category.id] = category
        return category
    
    def get_category(self, category_id: str) -> Optional[ArticleCategory]:
        """الحصول على فئة"""
        return self.categories.get(category_id)
    
    def get_category_by_slug(self, slug: str) -> Optional[ArticleCategory]:
        """الحصول على فئة بالـ slug"""
        for category in self.categories.values():
            if category.slug == slug:
                return category
        return None
    
    def create_article(self, title: str, slug: str, content: str,
                      author_id: str, author_name: str,
                      category_id: str = "", tags: List[str] = None,
                      summary: str = "") -> Article:
        """إنشاء مقالة"""
        article = Article(
            id=str(uuid.uuid4()),
            slug=slug,
            title=title,
            content=content,
            summary=summary,
            category_id=category_id,
            tags=tags or [],
            author_id=author_id,
            author_name=author_name
        )
        
        self.articles[article.id] = article
        
        # Update category count
        if category_id in self.categories:
            self.categories[category_id].article_count += 1
        
        return article
    
    def get_article(self, article_id: str) -> Optional[Article]:
        """الحصول على مقالة"""
        return self.articles.get(article_id)
    
    def get_article_by_slug(self, slug: str) -> Optional[Article]:
        """الحصول على مقالة بالـ slug"""
        for article in self.articles.values():
            if article.slug == slug:
                return article
        return None
    
    def update_article(self, article_id: str, content: str,
                      changes_summary: str, updated_by: str) -> Article:
        """تحديث مقالة"""
        article = self.articles.get(article_id)
        if not article:
            raise ValueError(f"Article {article_id} not found")
        
        article.update_content(content, changes_summary, updated_by)
        return article
    
    def publish_article(self, article_id: str) -> Article:
        """نشر مقالة"""
        article = self.articles.get(article_id)
        if article:
            article.publish()
        return article
    
    def rate_article(self, article_id: str, user_id: str, rating: int) -> bool:
        """تقييم مقالة"""
        if rating < 1 or rating > 5:
            return False
        
        article = self.articles.get(article_id)
        if not article:
            return False
        
        # Check if user already rated
        if article_id not in self.article_ratings:
            self.article_ratings[article_id] = {}
        
        old_rating = self.article_ratings[article_id].get(user_id)
        self.article_ratings[article_id][user_id] = rating
        
        # Update article rating
        if old_rating:
            article.rating_sum = article.rating_sum - old_rating + rating
        else:
            article.rate(rating)
        
        return True
    
    def mark_article_helpful(self, article_id: str, was_helpful: bool = True):
        """تحديد ما إذا كانت المقالة مفيدة"""
        article = self.articles.get(article_id)
        if article:
            article.mark_helpful(was_helpful)
    
    def search_articles(self, query: str, category_id: str = None) -> List[Article]:
        """البحث في المقالات"""
        results = []
        query = query.lower()
        
        for article in self.articles.values():
            if article.status != ArticleStatus.PUBLISHED:
                continue
            
            if category_id and article.category_id != category_id:
                continue
            
            if (query in article.title.lower() or
                query in article.content.lower() or
                query in article.tags or
                query in article.summary.lower()):
                results.append(article)
        
        # Sort by relevance (simple: title matches first)
        results.sort(key=lambda a: (query not in a.title.lower(), -a.rating))
        
        return results
    
    def get_articles_by_category(self, category_id: str) -> List[Article]:
        """الحصول على مقالات فئة"""
        return [
            a for a in self.articles.values()
            if a.category_id == category_id and a.status == ArticleStatus.PUBLISHED
        ]
    
    def get_articles_by_tag(self, tag: str) -> List[Article]:
        """الحصول على مقالات بوسم"""
        return [
            a for a in self.articles.values()
            if tag in a.tags and a.status == ArticleStatus.PUBLISHED
        ]
    
    def get_related_articles(self, article_id: str, limit: int = 5) -> List[Article]:
        """الحصول على مقالات ذات صلة"""
        article = self.articles.get(article_id)
        if not article:
            return []
        
        # First, get explicitly related
        related = [self.articles.get(rid) for rid in article.related_article_ids]
        related = [r for r in related if r and r.status == ArticleStatus.PUBLISHED]
        
        # Then, get by same category or tags
        if len(related) < limit:
            for a in self.articles.values():
                if (a.id != article_id and 
                    a.status == ArticleStatus.PUBLISHED and
                    a not in related):
                    
                    if a.category_id == article.category_id:
                        related.append(a)
                    elif set(a.tags) & set(article.tags):
                        related.append(a)
                
                if len(related) >= limit:
                    break
        
        return related[:limit]
    
    def get_popular_articles(self, limit: int = 10) -> List[Article]:
        """الحصول على المقالات الأكثر شعبية"""
        published = [a for a in self.articles.values() if a.status == ArticleStatus.PUBLISHED]
        published.sort(key=lambda a: (a.view_count, a.rating), reverse=True)
        return published[:limit]
    
    def get_recent_articles(self, limit: int = 10) -> List[Article]:
        """الحصول على المقالات الحديثة"""
        published = [a for a in self.articles.values() if a.status == ArticleStatus.PUBLISHED]
        published.sort(key=lambda a: a.published_at or a.created_at, reverse=True)
        return published[:limit]
    
    def get_category_tree(self) -> Dict[str, Any]:
        """الحصول على شجرة الفئات"""
        tree = {}
        
        for category in self.categories.values():
            if category.parent_id is None:
                tree[category.id] = self._build_category_node(category)
        
        return tree
    
    def _build_category_node(self, category: ArticleCategory) -> Dict[str, Any]:
        """بناء عقدة فئة"""
        node = {
            "category": category.to_dict(),
            "children": {}
        }
        
        for child in self.categories.values():
            if child.parent_id == category.id:
                node["children"][child.id] = self._build_category_node(child)
        
        return node
    
    def get_statistics(self) -> Dict[str, Any]:
        """إحصائيات قاعدة المعرفة"""
        total_articles = len(self.articles)
        published = len([a for a in self.articles.values() if a.status == ArticleStatus.PUBLISHED])
        draft = len([a for a in self.articles.values() if a.status == ArticleStatus.DRAFT])
        archived = len([a for a in self.articles.values() if a.status == ArticleStatus.ARCHIVED])
        
        total_views = sum(a.view_count for a in self.articles.values())
        avg_rating = (
            sum(a.rating for a in self.articles.values()) / published
            if published > 0 else 0
        )
        
        return {
            "total_articles": total_articles,
            "published": published,
            "draft": draft,
            "archived": archived,
            "total_categories": len(self.categories),
            "total_views": total_views,
            "average_rating": round(avg_rating, 1)
        }
