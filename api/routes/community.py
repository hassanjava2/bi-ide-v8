"""
Community Routes - مسارات المجتمع
Forum, Knowledge Base, Code Sharing, Profiles
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_async_session
from community.db_service import CommunityDBService
from api.auth import get_current_user
from core.user_models import UserDB

router = APIRouter(prefix="/community", tags=["Community"])


# ==================== Schemas ====================

class CategoryCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    display_order: int = 0


class CategoryResponse(BaseModel):
    id: str
    name: str
    description: str
    display_order: int
    forums: List[dict] = []


class ForumCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    category_id: Optional[str] = None
    is_public: bool = True
    icon: str = ""
    color: str = "#2196F3"


class ForumResponse(BaseModel):
    id: str
    name: str
    description: str
    category_id: Optional[str]
    is_public: bool
    topic_count: int
    post_count: int
    icon: str
    color: str


class TopicCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1)
    tags: List[str] = []


class TopicResponse(BaseModel):
    id: str
    forum_id: str
    title: str
    author_name: str
    status: str
    view_count: int
    reply_count: int
    has_solution: bool
    created_at: str
    last_post_at: str


class PostCreate(BaseModel):
    content: str = Field(..., min_length=1)
    parent_id: Optional[str] = None


class PostResponse(BaseModel):
    id: str
    author_name: str
    content: str
    upvotes: int
    downvotes: int
    is_solution: bool
    created_at: str


class KBCategoryCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    parent_id: Optional[str] = None


class KBArticleCreate(BaseModel):
    category_id: str
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1)
    tags: List[str] = []


class CodeSnippetCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    code: str = Field(..., min_length=1)
    language: str = Field(..., min_length=1, max_length=50)
    tags: List[str] = []


# ==================== Forum Routes ====================

@router.get("/forums/categories", response_model=List[CategoryResponse])
async def get_forum_categories(
    session: AsyncSession = Depends(get_async_session)
):
    """جلب فئات المنتديات"""
    service = CommunityDBService(session)
    categories = await service.get_categories()
    
    return [{
        "id": cat.id,
        "name": cat.name,
        "description": cat.description or "",
        "display_order": cat.display_order,
        "forums": [{
            "id": f.id,
            "name": f.name,
            "description": f.description or "",
            "topic_count": f.topic_count,
            "post_count": f.post_count,
            "icon": f.icon,
            "color": f.color
        } for f in cat.forums] if hasattr(cat, 'forums') else []
    } for cat in categories]


@router.post("/forums/categories", response_model=CategoryResponse, status_code=status.HTTP_201_CREATED)
async def create_category(
    data: CategoryCreate,
    session: AsyncSession = Depends(get_async_session)
):
    """إنشاء فئة منتديات جديدة"""
    service = CommunityDBService(session)
    category = await service.create_category(
        name=data.name,
        description=data.description,
        display_order=data.display_order
    )
    
    return {
        "id": category.id,
        "name": category.name,
        "description": category.description or "",
        "display_order": category.display_order,
        "forums": []
    }


@router.get("/forums", response_model=List[ForumResponse])
async def get_forums(
    category_id: Optional[str] = None,
    session: AsyncSession = Depends(get_async_session)
):
    """جلب المنتديات"""
    service = CommunityDBService(session)
    forums = await service.get_forums(category_id=category_id)
    
    return [{
        "id": f.id,
        "name": f.name,
        "description": f.description or "",
        "category_id": f.category_id,
        "is_public": f.is_public,
        "topic_count": f.topic_count,
        "post_count": f.post_count,
        "icon": f.icon,
        "color": f.color
    } for f in forums]


@router.post("/forums", response_model=ForumResponse, status_code=status.HTTP_201_CREATED)
async def create_forum(
    data: ForumCreate,
    session: AsyncSession = Depends(get_async_session)
):
    """إنشاء منتدى جديد"""
    service = CommunityDBService(session)
    forum = await service.create_forum(
        name=data.name,
        description=data.description,
        category_id=data.category_id,
        is_public=data.is_public,
        icon=data.icon,
        color=data.color
    )
    
    return {
        "id": forum.id,
        "name": forum.name,
        "description": forum.description or "",
        "category_id": forum.category_id,
        "is_public": forum.is_public,
        "topic_count": 0,
        "post_count": 0,
        "icon": forum.icon,
        "color": forum.color
    }


@router.get("/forums/{forum_id}/topics")
async def get_forum_topics(
    forum_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session)
):
    """جلب مواضيع منتدى"""
    service = CommunityDBService(session)
    
    # Verify forum exists
    forum = await service.get_forum(forum_id)
    if not forum:
        raise HTTPException(status_code=404, detail="Forum not found")
    
    result = await service.get_topics(forum_id, page=page, per_page=per_page)
    
    return {
        "items": [{
            "id": t.id,
            "forum_id": t.forum_id,
            "title": t.title,
            "author_name": t.author_name,
            "status": t.status.value if hasattr(t.status, 'value') else str(t.status),
            "view_count": t.view_count,
            "reply_count": t.reply_count,
            "has_solution": t.has_solution,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "last_post_at": t.last_post_at.isoformat() if t.last_post_at else None
        } for t in result["items"]],
        "total": result["total"],
        "page": result["page"],
        "per_page": result["per_page"],
        "pages": result["pages"]
    }


@router.post("/forums/{forum_id}/topics", status_code=status.HTTP_201_CREATED)
async def create_topic(
    forum_id: str,
    data: TopicCreate,
    session: AsyncSession = Depends(get_async_session),
    current_user: UserDB = Depends(get_current_user)
):
    """إنشاء موضوع جديد"""
    service = CommunityDBService(session)
    
    # Verify forum exists
    forum = await service.get_forum(forum_id)
    if not forum:
        raise HTTPException(status_code=404, detail="Forum not found")
    
    author_id = current_user.id
    author_name = current_user.username or current_user.email
    
    topic = await service.create_topic(
        forum_id=forum_id,
        title=data.title,
        content=data.content,
        author_id=author_id,
        author_name=author_name,
        tags=data.tags
    )
    
    return {
        "id": topic.id,
        "forum_id": topic.forum_id,
        "title": topic.title,
        "author_name": topic.author_name,
        "status": topic.status.value if hasattr(topic.status, 'value') else str(topic.status),
        "view_count": topic.view_count,
        "reply_count": topic.reply_count,
        "has_solution": topic.has_solution,
        "created_at": topic.created_at.isoformat() if topic.created_at else None,
        "last_post_at": topic.last_post_at.isoformat() if topic.last_post_at else None
    }


@router.get("/topics/{topic_id}")
async def get_topic(
    topic_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """جلب موضوع"""
    service = CommunityDBService(session)
    topic = await service.get_topic(topic_id)
    
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    return {
        "id": topic.id,
        "forum_id": topic.forum_id,
        "title": topic.title,
        "content": topic.content,
        "author_name": topic.author_name,
        "status": topic.status.value if hasattr(topic.status, 'value') else str(topic.status),
        "view_count": topic.view_count,
        "reply_count": topic.reply_count,
        "has_solution": topic.has_solution,
        "created_at": topic.created_at.isoformat() if topic.created_at else None,
        "last_post_at": topic.last_post_at.isoformat() if topic.last_post_at else None
    }


@router.get("/topics/{topic_id}/posts")
async def get_topic_posts(
    topic_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session)
):
    """جلب مشاركات موضوع"""
    service = CommunityDBService(session)
    result = await service.get_posts(topic_id, page=page, per_page=per_page)
    
    return {
        "items": [{
            "id": p.id,
            "author_name": p.author_name,
            "content": p.content,
            "upvotes": p.upvotes,
            "downvotes": p.downvotes,
            "is_solution": p.is_solution,
            "created_at": p.created_at.isoformat() if p.created_at else None
        } for p in result["items"]],
        "total": result["total"],
        "page": result["page"],
        "per_page": result["per_page"],
        "pages": result["pages"]
    }


@router.post("/topics/{topic_id}/posts", status_code=status.HTTP_201_CREATED)
async def create_post(
    topic_id: str,
    data: PostCreate,
    session: AsyncSession = Depends(get_async_session),
    current_user: UserDB = Depends(get_current_user)
):
    """إضافة رد على موضوع"""
    service = CommunityDBService(session)
    
    author_id = current_user.id
    author_name = current_user.username or current_user.email
    
    post = await service.create_post(
        topic_id=topic_id,
        content=data.content,
        author_id=author_id,
        author_name=author_name,
        parent_id=data.parent_id
    )
    
    return {
        "id": post.id,
        "author_name": post.author_name,
        "content": post.content,
        "upvotes": post.upvotes,
        "downvotes": post.downvotes,
        "is_solution": post.is_solution,
        "created_at": post.created_at.isoformat() if post.created_at else None
    }


# ==================== Knowledge Base Routes ====================

@router.get("/kb/categories")
async def get_kb_categories(
    session: AsyncSession = Depends(get_async_session)
):
    """جلب فئات قاعدة المعرفة"""
    service = CommunityDBService(session)
    categories = await service.get_kb_categories()
    
    return [{
        "id": c.id,
        "name": c.name,
        "description": c.description or "",
        "icon": c.icon,
        "color": c.color,
        "article_count": len(c.articles) if hasattr(c, 'articles') else 0
    } for c in categories]


@router.post("/kb/categories", status_code=status.HTTP_201_CREATED)
async def create_kb_category(
    data: KBCategoryCreate,
    session: AsyncSession = Depends(get_async_session)
):
    """إنشاء فئة KB"""
    service = CommunityDBService(session)
    category = await service.create_kb_category(
        name=data.name,
        description=data.description,
        parent_id=data.parent_id
    )
    
    return {
        "id": category.id,
        "name": category.name,
        "description": category.description or "",
        "parent_id": category.parent_id
    }


@router.get("/kb/articles")
async def get_kb_articles(
    category_id: Optional[str] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session)
):
    """جلب مقالات قاعدة المعرفة"""
    service = CommunityDBService(session)
    result = await service.get_kb_articles(category_id, page=page, per_page=per_page)
    
    return {
        "items": [{
            "id": a.id,
            "title": a.title,
            "category_id": a.category_id,
            "author_name": a.author_name,
            "status": a.status,
            "view_count": a.view_count,
            "tags": a.tags or [],
            "created_at": a.created_at.isoformat() if a.created_at else None
        } for a in result["items"]],
        "total": result["total"],
        "page": result["page"],
        "per_page": result["per_page"],
        "pages": result["pages"]
    }


@router.post("/kb/articles", status_code=status.HTTP_201_CREATED)
async def create_kb_article(
    data: KBArticleCreate,
    session: AsyncSession = Depends(get_async_session),
    current_user: UserDB = Depends(get_current_user)
):
    """إنشاء مقالة KB"""
    service = CommunityDBService(session)
    
    author_id = current_user.id
    author_name = current_user.username or current_user.email
    
    article = await service.create_kb_article(
        category_id=data.category_id,
        title=data.title,
        content=data.content,
        author_id=author_id,
        author_name=author_name,
        tags=data.tags
    )
    
    return {
        "id": article.id,
        "title": article.title,
        "category_id": article.category_id,
        "author_name": article.author_name,
        "status": article.status,
        "view_count": article.view_count,
        "tags": article.tags or [],
        "created_at": article.created_at.isoformat() if article.created_at else None
    }


@router.get("/kb/articles/{article_id}")
async def get_kb_article(
    article_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """جلب مقالة KB"""
    service = CommunityDBService(session)
    article = await service.get_kb_article(article_id)
    
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    return {
        "id": article.id,
        "title": article.title,
        "content": article.content,
        "category_id": article.category_id,
        "author_name": article.author_name,
        "status": article.status,
        "view_count": article.view_count,
        "helpful_count": article.helpful_count,
        "tags": article.tags or [],
        "created_at": article.created_at.isoformat() if article.created_at else None,
        "updated_at": article.updated_at.isoformat() if article.updated_at else None
    }


# ==================== Code Sharing Routes ====================

@router.get("/code/snippets")
async def get_code_snippets(
    language: Optional[str] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session)
):
    """جلب قطع الكود"""
    service = CommunityDBService(session)
    result = await service.get_code_snippets(language, page=page, per_page=per_page)
    
    return {
        "items": [{
            "id": s.id,
            "title": s.title,
            "description": s.description or "",
            "language": s.language,
            "author_name": s.author_name,
            "view_count": s.view_count,
            "tags": s.tags or [],
            "created_at": s.created_at.isoformat() if s.created_at else None
        } for s in result["items"]],
        "total": result["total"],
        "page": result["page"],
        "per_page": result["per_page"],
        "pages": result["pages"]
    }


@router.post("/code/snippets", status_code=status.HTTP_201_CREATED)
async def create_code_snippet(
    data: CodeSnippetCreate,
    session: AsyncSession = Depends(get_async_session),
    current_user: UserDB = Depends(get_current_user)
):
    """إنشاء قطعة كود"""
    service = CommunityDBService(session)
    
    author_id = current_user.id
    author_name = current_user.username or current_user.email
    
    snippet = await service.create_code_snippet(
        title=data.title,
        code=data.code,
        language=data.language,
        author_id=author_id,
        author_name=author_name,
        description=data.description,
        tags=data.tags
    )
    
    return {
        "id": snippet.id,
        "title": snippet.title,
        "description": snippet.description or "",
        "language": snippet.language,
        "code": snippet.code,
        "author_name": snippet.author_name,
        "tags": snippet.tags or [],
        "created_at": snippet.created_at.isoformat() if snippet.created_at else None
    }


@router.get("/code/snippets/{snippet_id}")
async def get_code_snippet(
    snippet_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """جلب قطعة كود"""
    service = CommunityDBService(session)
    snippet = await service.get_code_snippet(snippet_id)
    
    if not snippet:
        raise HTTPException(status_code=404, detail="Snippet not found")
    
    return {
        "id": snippet.id,
        "title": snippet.title,
        "description": snippet.description or "",
        "code": snippet.code,
        "language": snippet.language,
        "author_name": snippet.author_name,
        "view_count": snippet.view_count,
        "download_count": snippet.download_count,
        "tags": snippet.tags or [],
        "created_at": snippet.created_at.isoformat() if snippet.created_at else None
    }


# ==================== Profile Routes ====================

@router.get("/profiles/{user_id}/stats")
async def get_user_stats(
    user_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """جلب إحصائيات المستخدم"""
    service = CommunityDBService(session)
    stats = await service.get_user_stats(user_id)
    
    if not stats:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "user_id": stats.user_id,
        "reputation": stats.reputation,
        "forum_posts": stats.forum_posts,
        "forum_topics": stats.forum_topics,
        "kb_articles": stats.kb_articles,
        "code_snippets": stats.code_snippets,
        "badges": stats.badges or []
    }


# ==================== Notification Routes ====================

@router.get("/notifications")
async def get_notifications(
    unread_only: bool = False,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
    current_user: UserDB = Depends(get_current_user)
):
    """جلب الإشعارات"""
    service = CommunityDBService(session)
    
    user_id = current_user.id
    
    result = await service.get_notifications(user_id, unread_only, page, per_page)
    
    return {
        "items": [{
            "id": n.id,
            "type": n.type,
            "title": n.title,
            "message": n.message,
            "is_read": n.is_read,
            "created_at": n.created_at.isoformat() if n.created_at else None
        } for n in result["items"]],
        "total": result["total"],
        "page": result["page"],
        "per_page": result["per_page"],
        "pages": result["pages"]
    }


@router.post("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """تحديد إشعار كمقروء"""
    service = CommunityDBService(session)
    success = await service.mark_notification_read(notification_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    return {"message": "Notification marked as read"}
