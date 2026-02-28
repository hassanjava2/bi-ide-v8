"""
روتر المجتمع - Community Router

يوفر نقاط النهاية للتفاعل المجتمعي ومشاركة المعرفة.
Provides endpoints for community interaction and knowledge sharing.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from .auth import get_current_active_user, User

router = APIRouter(prefix="/community", tags=["المجتمع | Community"])


# نماذج Pydantic - Pydantic Models
class Forum(BaseModel):
    """نموذج المنتدى | Forum model"""
    id: str
    name: str
    description: str
    icon: Optional[str] = None
    topics_count: int
    posts_count: int
    last_activity: datetime
    is_active: bool


class PostAuthor(BaseModel):
    """نموذج كاتب المنشور | Post author model"""
    id: int
    username: str
    avatar: Optional[str] = None
    reputation: int


class Post(BaseModel):
    """نموذج المنشور | Post model"""
    id: str
    forum_id: str
    title: str
    content: str
    author: PostAuthor
    tags: List[str]
    views_count: int
    replies_count: int
    likes_count: int
    is_pinned: bool
    is_solved: bool
    created_at: datetime
    updated_at: datetime


class PostCreate(BaseModel):
    """نموذج إنشاء المنشور | Post create model"""
    forum_id: str
    title: str = Field(..., min_length=5, max_length=200)
    content: str = Field(..., min_length=10, max_length=10000)
    tags: List[str] = Field(default=[])


class KnowledgeBaseArticle(BaseModel):
    """نموذج مقالة قاعدة المعرفة | Knowledge base article model"""
    id: str
    title: str
    content: str
    summary: str
    category: str
    author: PostAuthor
    tags: List[str]
    views_count: int
    helpful_count: int
    difficulty: str  # beginner, intermediate, advanced
    estimated_read_time: int  # minutes
    created_at: datetime
    updated_at: datetime


class SharedCode(BaseModel):
    """نموذج الكود المشترك | Shared code model"""
    id: str
    title: str
    description: str
    code: str
    language: str
    author: PostAuthor
    tags: List[str]
    likes_count: int
    forks_count: int
    comments_count: int
    is_public: bool
    created_at: datetime
    updated_at: datetime


class ShareCodeRequest(BaseModel):
    """نموذج طلب مشاركة الكود | Share code request model"""
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., max_length=2000)
    code: str = Field(..., min_length=1)
    language: str
    tags: List[str] = Field(default=[])
    is_public: bool = True


class Comment(BaseModel):
    """نموذج التعليق | Comment model"""
    id: str
    content: str
    author: PostAuthor
    likes_count: int
    created_at: datetime


# قواعد بيانات وهمية - Fake Databases
fake_forums = {
    "general": {
        "id": "general",
        "name": "النقاش العام | General Discussion",
        "description": "نقاشات عامة حول BI-IDE والبرمجة | General discussions about BI-IDE and programming",
        "icon": "💬",
        "topics_count": 150,
        "posts_count": 1200,
        "last_activity": datetime.utcnow(),
        "is_active": True
    },
    "help": {
        "id": "help",
        "name": "المساعدة والدعم | Help & Support",
        "description": "اطرح أسئلتك واحصل على المساعدة | Ask questions and get help",
        "icon": "❓",
        "topics_count": 89,
        "posts_count": 567,
        "last_activity": datetime.utcnow(),
        "is_active": True
    },
    "showcase": {
        "id": "showcase",
        "name": "المشاريع | Showcase",
        "description": "شارك مشاريعك وإنجازاتك | Share your projects and achievements",
        "icon": "🚀",
        "topics_count": 45,
        "posts_count": 230,
        "last_activity": datetime.utcnow(),
        "is_active": True
    },
    "code-sharing": {
        "id": "code-sharing",
        "name": "مشاركة الكود | Code Sharing",
        "description": "شارك أكواد مفيدة ومكتبات | Share useful code and libraries",
        "icon": "💻",
        "topics_count": 78,
        "posts_count": 450,
        "last_activity": datetime.utcnow(),
        "is_active": True
    }
}

fake_posts = {
    "post-001": {
        "id": "post-001",
        "forum_id": "general",
        "title": "مرحباً بكم في مجتمع BI-IDE!",
        "content": "نرحب بجميع المستخدمين في مجتمعنا...",
        "author": {
            "id": 1,
            "username": "admin",
            "avatar": None,
            "reputation": 1000
        },
        "tags": ["welcome", "announcement"],
        "views_count": 1250,
        "replies_count": 45,
        "likes_count": 89,
        "is_pinned": True,
        "is_solved": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
}

fake_articles = {
    "kb-001": {
        "id": "kb-001",
        "title": "كيفية البدء مع BI-IDE",
        "content": """
        # كيفية البدء مع BI-IDE
        
        ## التثبيت
        1. قم بتثبيت BI-IDE من الموقع الرسمي
        2. شغل التطبيق وقم بإنشاء حساب
        3. ابدأ مشروعك الأول
        
        ## المميزات الرئيسية
        - تحرير الكود الذكي
        - تدريب نماذج الذكاء الاصطناعي
        - تكامل ERP
        """,
        "summary": "دليل شامل للبدء مع BI-IDE",
        "category": "getting-started",
        "author": {
            "id": 1,
            "username": "admin",
            "avatar": None,
            "reputation": 1000
        },
        "tags": ["getting-started", "tutorial"],
        "views_count": 5000,
        "helpful_count": 450,
        "difficulty": "beginner",
        "estimated_read_time": 10,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    },
    "kb-002": {
        "id": "kb-002",
        "title": "أفضل ممارسات كتابة الكود",
        "content": "محتوى المقالة...",
        "summary": "تعرف على أفضل الممارسات لكتابة كود نظيف",
        "category": "best-practices",
        "author": {
            "id": 2,
            "username": "expert_dev",
            "avatar": None,
            "reputation": 850
        },
        "tags": ["best-practices", "coding"],
        "views_count": 3200,
        "helpful_count": 280,
        "difficulty": "intermediate",
        "estimated_read_time": 15,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
}

fake_shared_code = {
    "code-001": {
        "id": "code-001",
        "title": "دوال مساعدة للتعامل مع البيانات",
        "description": "مجموعة من الدوال المساعدة لتنظيف ومعالجة البيانات",
        "code": """
def clean_data(df):
    '''Clean and preprocess dataframe'''
    df = df.dropna()
    df = df.drop_duplicates()
    return df
        """,
        "language": "python",
        "author": {
            "id": 3,
            "username": "data_wizard",
            "avatar": None,
            "reputation": 500
        },
        "tags": ["python", "data-processing", "utilities"],
        "likes_count": 125,
        "forks_count": 45,
        "comments_count": 20,
        "is_public": True,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
}

fake_post_counter = 2
fake_article_counter = 3
fake_code_counter = 2


@router.get(
    "/forums",
    response_model=List[Forum],
    status_code=status.HTTP_200_OK,
    summary="قائمة المنتديات | List forums"
)
async def list_forums(current_user: User = Depends(get_current_active_user)):
    """
    الحصول على قائمة المنتديات المتاحة.
    Get list of available forums.
    """
    return [Forum(**f) for f in fake_forums.values()]


@router.get(
    "/forums/{forum_id}/posts",
    response_model=List[Post],
    status_code=status.HTTP_200_OK,
    summary="منشورات المنتدى | Forum posts"
)
async def get_forum_posts(
    forum_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على منشورات منتدى معين.
    Get posts from a specific forum.
    """
    if forum_id not in fake_forums:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="المنتدى غير موجود | Forum not found"
        )
    
    posts = [p for p in fake_posts.values() if p["forum_id"] == forum_id]
    
    return [Post(**p) for p in sorted(
        posts,
        key=lambda x: (x["is_pinned"], x["created_at"]),
        reverse=True
    )]


@router.post(
    "/posts",
    response_model=Post,
    status_code=status.HTTP_201_CREATED,
    summary="إنشاء منشور | Create post"
)
async def create_post(
    post: PostCreate,
    current_user: User = Depends(get_current_active_user)
):
    """
    إنشاء منشور جديد في منتدى.
    Create a new post in a forum.
    """
    global fake_post_counter
    
    if post.forum_id not in fake_forums:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="المنتدى غير موجود | Forum not found"
        )
    
    post_id = f"post-{fake_post_counter:03d}"
    fake_post_counter += 1
    
    new_post = {
        "id": post_id,
        "forum_id": post.forum_id,
        "title": post.title,
        "content": post.content,
        "author": {
            "id": current_user.id,
            "username": current_user.username,
            "avatar": None,
            "reputation": 100  # سيتم حسابه من قاعدة البيانات
        },
        "tags": post.tags,
        "views_count": 0,
        "replies_count": 0,
        "likes_count": 0,
        "is_pinned": False,
        "is_solved": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    fake_posts[post_id] = new_post
    
    # تحديث إحصائيات المنتدى | Update forum stats
    fake_forums[post.forum_id]["posts_count"] += 1
    fake_forums[post.forum_id]["last_activity"] = datetime.utcnow()
    
    return Post(**new_post)


@router.get(
    "/posts/{post_id}",
    response_model=Post,
    status_code=status.HTTP_200_OK,
    summary="تفاصيل المنشور | Post details"
)
async def get_post(
    post_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على تفاصيل منشور محدد.
    Get details of a specific post.
    """
    if post_id not in fake_posts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="المنشور غير موجود | Post not found"
        )
    
    # زيادة عدد المشاهدات | Increment view count
    fake_posts[post_id]["views_count"] += 1
    
    return Post(**fake_posts[post_id])


@router.get(
    "/knowledge-base",
    response_model=List[KnowledgeBaseArticle],
    status_code=status.HTTP_200_OK,
    summary="قاعدة المعرفة | Knowledge base"
)
async def list_knowledge_base(
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على مقالات قاعدة المعرفة.
    Get knowledge base articles.
    """
    articles = list(fake_articles.values())
    
    if category:
        articles = [a for a in articles if a["category"] == category]
    if difficulty:
        articles = [a for a in articles if a["difficulty"] == difficulty]
    
    return [KnowledgeBaseArticle(**a) for a in sorted(
        articles,
        key=lambda x: x["views_count"],
        reverse=True
    )]


@router.get(
    "/knowledge-base/{article_id}",
    response_model=KnowledgeBaseArticle,
    status_code=status.HTTP_200_OK,
    summary="مقالة قاعدة المعرفة | Knowledge base article"
)
async def get_article(
    article_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على مقالة محددة من قاعدة المعرفة.
    Get a specific knowledge base article.
    """
    if article_id not in fake_articles:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="المقالة غير موجودة | Article not found"
        )
    
    # زيادة عدد المشاهدات | Increment view count
    fake_articles[article_id]["views_count"] += 1
    
    return KnowledgeBaseArticle(**fake_articles[article_id])


@router.post(
    "/knowledge-base/{article_id}/helpful",
    status_code=status.HTTP_200_OK,
    summary="تقييم المقالة مفيدة | Mark article as helpful"
)
async def mark_article_helpful(
    article_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    وضع علامة على المقالة كمفيدة.
    Mark an article as helpful.
    """
    if article_id not in fake_articles:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="المقالة غير موجودة | Article not found"
        )
    
    fake_articles[article_id]["helpful_count"] += 1
    
    return {
        "article_id": article_id,
        "helpful_count": fake_articles[article_id]["helpful_count"],
        "message": "شكراً لتقييمك! | Thanks for your feedback!"
    }


@router.get(
    "/shared-code",
    response_model=List[SharedCode],
    status_code=status.HTTP_200_OK,
    summary="الكود المشترك | Shared code"
)
async def list_shared_code(
    language: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على قائمة الكود المشترك.
    Get list of shared code.
    """
    code_list = list(fake_shared_code.values())
    
    if language:
        code_list = [c for c in code_list if c["language"] == language]
    
    return [SharedCode(**c) for c in sorted(
        code_list,
        key=lambda x: x["likes_count"],
        reverse=True
    )]


@router.post(
    "/share-code",
    response_model=SharedCode,
    status_code=status.HTTP_201_CREATED,
    summary="مشاركة الكود | Share code"
)
async def share_code(
    request: ShareCodeRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    مشاركة كود جديد مع المجتمع.
    Share new code with the community.
    """
    global fake_code_counter
    
    code_id = f"code-{fake_code_counter:03d}"
    fake_code_counter += 1
    
    new_code = {
        "id": code_id,
        "title": request.title,
        "description": request.description,
        "code": request.code,
        "language": request.language,
        "author": {
            "id": current_user.id,
            "username": current_user.username,
            "avatar": None,
            "reputation": 100
        },
        "tags": request.tags,
        "likes_count": 0,
        "forks_count": 0,
        "comments_count": 0,
        "is_public": request.is_public,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    fake_shared_code[code_id] = new_code
    
    return SharedCode(**new_code)


@router.get(
    "/shared-code/{code_id}",
    response_model=SharedCode,
    status_code=status.HTTP_200_OK,
    summary="تفاصيل الكود المشترك | Shared code details"
)
async def get_shared_code(
    code_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    الحصول على تفاصيل كود مشترك محدد.
    Get details of specific shared code.
    """
    if code_id not in fake_shared_code:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="الكود غير موجود | Code not found"
        )
    
    return SharedCode(**fake_shared_code[code_id])


@router.post(
    "/shared-code/{code_id}/like",
    status_code=status.HTTP_200_OK,
    summary="إعجاب بالكود | Like code"
)
async def like_code(
    code_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    وضع إعجاب على كود مشترك.
    Like a shared code snippet.
    """
    if code_id not in fake_shared_code:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="الكود غير موجود | Code not found"
        )
    
    fake_shared_code[code_id]["likes_count"] += 1
    
    return {
        "code_id": code_id,
        "likes_count": fake_shared_code[code_id]["likes_count"],
        "message": "تم الإعجاب بالكود! | Code liked!"
    }
