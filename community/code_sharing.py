"""
Code Sharing - مشاركة الأكواد

المميزات:
- إبراز بناء الجملة (Syntax highlighting)
- أذونات المشاركة
- التعليقات والتقييمات
- المجموعات/المجلدات
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class Visibility(Enum):
    """رؤية المقطع"""
    PUBLIC = "public"                 # عام
    PRIVATE = "private"               # خاص
    UNLISTED = "unlisted"             # غير مدرج
    RESTRICTED = "restricted"         # مقيد (بأذونات محددة)


class Language(Enum):
    """لغات البرمجة المدعومة"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SQL = "sql"
    BASH = "bash"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"
    HTML = "html"
    CSS = "css"
    OTHER = "other"


@dataclass
class CodeComment:
    """تعليق على كود"""
    id: str
    author_id: str
    author_name: str
    content: str
    line_number: Optional[int] = None  # للتعليقات على سطر محدد
    parent_id: Optional[str] = None    # للردود
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "content": self.content,
            "line_number": self.line_number,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class CodeSnippet:
    """مقطع كود"""
    id: str
    title: str
    code: str
    language: Language
    author_id: str
    author_name: str
    
    # Description
    description: str = ""
    
    # Visibility
    visibility: Visibility = Visibility.PUBLIC
    allowed_user_ids: List[str] = field(default_factory=list)  # للـ RESTRICTED
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    
    # Statistics
    view_count: int = 0
    download_count: int = 0
    
    # Ratings
    rating_sum: int = 0
    rating_count: int = 0
    
    # Fork info
    is_fork: bool = False
    forked_from: Optional[str] = None
    
    # Comments
    comments: List[CodeComment] = field(default_factory=list)
    
    # Versioning
    version: int = 1
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def rating(self) -> float:
        """التقييم"""
        if self.rating_count == 0:
            return 0
        return round(self.rating_sum / self.rating_count, 1)
    
    @property
    def lines_count(self) -> int:
        """عدد الأسطر"""
        return len(self.code.splitlines())
    
    def can_view(self, user_id: str = None) -> bool:
        """التحقق من صلاحية المشاهدة"""
        if self.visibility == Visibility.PUBLIC:
            return True
        if self.visibility == Visibility.PRIVATE:
            return user_id == self.author_id
        if self.visibility == Visibility.UNLISTED:
            return True
        if self.visibility == Visibility.RESTRICTED:
            return user_id in self.allowed_user_ids or user_id == self.author_id
        return False
    
    def can_edit(self, user_id: str) -> bool:
        """التحقق من صلاحية التعديل"""
        return user_id == self.author_id
    
    def increment_views(self):
        """زيادة المشاهدات"""
        self.view_count += 1
    
    def increment_downloads(self):
        """زيادة التنزيلات"""
        self.download_count += 1
    
    def rate(self, rating: int):
        """تقييم"""
        if 1 <= rating <= 5:
            self.rating_sum += rating
            self.rating_count += 1
    
    def add_comment(self, author_id: str, author_name: str, 
                   content: str, line_number: int = None) -> CodeComment:
        """إضافة تعليق"""
        comment = CodeComment(
            id=str(uuid.uuid4()),
            author_id=author_id,
            author_name=author_name,
            content=content,
            line_number=line_number
        )
        
        self.comments.append(comment)
        return comment
    
    def fork(self, new_author_id: str, new_author_name: str) -> 'CodeSnippet':
        """إنشاء نسخة مشتقة"""
        forked = CodeSnippet(
            id=str(uuid.uuid4()),
            title=f"Fork: {self.title}",
            code=self.code,
            language=self.language,
            author_id=new_author_id,
            author_name=new_author_name,
            description=self.description,
            is_fork=True,
            forked_from=self.id
        )
        return forked
    
    def update_code(self, new_code: str, new_description: str = ""):
        """تحديث الكود"""
        self.code = new_code
        if new_description:
            self.description = new_description
        self.version += 1
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "language": self.language.value,
            "author_name": self.author_name,
            "visibility": self.visibility.value,
            "tags": self.tags,
            "view_count": self.view_count,
            "download_count": self.download_count,
            "rating": self.rating,
            "rating_count": self.rating_count,
            "lines_count": self.lines_count,
            "is_fork": self.is_fork,
            "forked_from": self.forked_from,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def to_detail_dict(self) -> Dict[str, Any]:
        """تفاصيل كاملة"""
        result = self.to_dict()
        result["code"] = self.code
        result["comments"] = [c.to_dict() for c in self.comments]
        return result


@dataclass
class SnippetCollection:
    """مجموعة مقاطع"""
    id: str
    name: str
    description: str = ""
    owner_id: str = ""
    
    snippet_ids: List[str] = field(default_factory=list)
    
    visibility: Visibility = Visibility.PUBLIC
    allowed_user_ids: List[str] = field(default_factory=list)
    
    # Organization
    folder_structure: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_snippet(self, snippet_id: str, folder: str = None):
        """إضافة مقطع"""
        if snippet_id not in self.snippet_ids:
            self.snippet_ids.append(snippet_id)
            self.updated_at = datetime.now(timezone.utc)
    
    def remove_snippet(self, snippet_id: str):
        """إزالة مقطع"""
        if snippet_id in self.snippet_ids:
            self.snippet_ids.remove(snippet_id)
            self.updated_at = datetime.now(timezone.utc)
    
    def can_view(self, user_id: str = None) -> bool:
        """التحقق من صلاحية المشاهدة"""
        if self.visibility == Visibility.PUBLIC:
            return True
        if self.visibility == Visibility.PRIVATE:
            return user_id == self.owner_id
        if self.visibility == Visibility.UNLISTED:
            return True
        if self.visibility == Visibility.RESTRICTED:
            return user_id in self.allowed_user_ids or user_id == self.owner_id
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "snippet_count": len(self.snippet_ids),
            "visibility": self.visibility.value,
            "created_at": self.created_at.isoformat()
        }


class CodeSharingPlatform:
    """
    منصة مشاركة الأكواد
    """
    
    def __init__(self):
        self.snippets: Dict[str, CodeSnippet] = {}
        self.collections: Dict[str, SnippetCollection] = {}
    
    def create_snippet(self, title: str, code: str, language: Language,
                      author_id: str, author_name: str,
                      description: str = "", visibility: Visibility = Visibility.PUBLIC,
                      tags: List[str] = None) -> CodeSnippet:
        """إنشاء مقطع كود"""
        snippet = CodeSnippet(
            id=str(uuid.uuid4()),
            title=title,
            code=code,
            language=language,
            author_id=author_id,
            author_name=author_name,
            description=description,
            visibility=visibility,
            tags=tags or []
        )
        
        self.snippets[snippet.id] = snippet
        return snippet
    
    def get_snippet(self, snippet_id: str, user_id: str = None) -> Optional[CodeSnippet]:
        """الحصول على مقطع"""
        snippet = self.snippets.get(snippet_id)
        if snippet and snippet.can_view(user_id):
            snippet.increment_views()
            return snippet
        return None
    
    def update_snippet(self, snippet_id: str, user_id: str,
                      new_code: str = None, new_description: str = None) -> CodeSnippet:
        """تحديث مقطع"""
        snippet = self.snippets.get(snippet_id)
        if not snippet:
            raise ValueError(f"Snippet {snippet_id} not found")
        
        if not snippet.can_edit(user_id):
            raise PermissionError("Not authorized to edit this snippet")
        
        snippet.update_code(
            new_code or snippet.code,
            new_description or snippet.description
        )
        return snippet
    
    def delete_snippet(self, snippet_id: str, user_id: str) -> bool:
        """حذف مقطع"""
        snippet = self.snippets.get(snippet_id)
        if not snippet:
            return False
        
        if not snippet.can_edit(user_id):
            return False
        
        del self.snippets[snippet_id]
        return True
    
    def fork_snippet(self, snippet_id: str, new_author_id: str, 
                    new_author_name: str) -> CodeSnippet:
        """إنشاء نسخة مشتقة"""
        original = self.snippets.get(snippet_id)
        if not original:
            raise ValueError(f"Snippet {snippet_id} not found")
        
        forked = original.fork(new_author_id, new_author_name)
        self.snippets[forked.id] = forked
        return forked
    
    def rate_snippet(self, snippet_id: str, user_id: str, rating: int) -> bool:
        """تقييم مقطع"""
        if rating < 1 or rating > 5:
            return False
        
        snippet = self.snippets.get(snippet_id)
        if not snippet:
            return False
        
        snippet.rate(rating)
        return True
    
    def add_comment(self, snippet_id: str, author_id: str, author_name: str,
                   content: str, line_number: int = None) -> CodeComment:
        """إضافة تعليق"""
        snippet = self.snippets.get(snippet_id)
        if not snippet:
            raise ValueError(f"Snippet {snippet_id} not found")
        
        return snippet.add_comment(author_id, author_name, content, line_number)
    
    def create_collection(self, name: str, owner_id: str,
                         description: str = "",
                         visibility: Visibility = Visibility.PUBLIC) -> SnippetCollection:
        """إنشاء مجموعة"""
        collection = SnippetCollection(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            owner_id=owner_id,
            visibility=visibility
        )
        
        self.collections[collection.id] = collection
        return collection
    
    def add_to_collection(self, collection_id: str, snippet_id: str, user_id: str):
        """إضافة مقطع لمجموعة"""
        collection = self.collections.get(collection_id)
        snippet = self.snippets.get(snippet_id)
        
        if not collection or not snippet:
            raise ValueError("Collection or snippet not found")
        
        if collection.owner_id != user_id:
            raise PermissionError("Not authorized")
        
        collection.add_snippet(snippet_id)
    
    def search_snippets(self, query: str, language: Language = None,
                       user_id: str = None) -> List[CodeSnippet]:
        """البحث في المقاطع"""
        results = []
        query = query.lower()
        
        for snippet in self.snippets.values():
            # Check visibility
            if not snippet.can_view(user_id):
                continue
            
            # Filter by language
            if language and snippet.language != language:
                continue
            
            # Search
            if (query in snippet.title.lower() or
                query in snippet.description.lower() or
                query in snippet.code.lower() or
                query in [t.lower() for t in snippet.tags]):
                results.append(snippet)
        
        # Sort by rating
        results.sort(key=lambda s: (s.rating, s.view_count), reverse=True)
        return results
    
    def get_snippets_by_language(self, language: Language) -> List[CodeSnippet]:
        """الحصول على مقاطع بلغة معينة"""
        return [
            s for s in self.snippets.values()
            if s.language == language and s.visibility == Visibility.PUBLIC
        ]
    
    def get_snippets_by_author(self, author_id: str, user_id: str = None) -> List[CodeSnippet]:
        """الحصول على مقاطع مؤلف"""
        return [
            s for s in self.snippets.values()
            if s.author_id == author_id and s.can_view(user_id)
        ]
    
    def get_popular_snippets(self, limit: int = 10) -> List[CodeSnippet]:
        """الحصول على المقاطع الأكثر شعبية"""
        public = [s for s in self.snippets.values() if s.visibility == Visibility.PUBLIC]
        public.sort(key=lambda s: (s.view_count, s.rating), reverse=True)
        return public[:limit]
    
    def get_recent_snippets(self, limit: int = 10) -> List[CodeSnippet]:
        """الحصول على المقاطع الحديثة"""
        public = [s for s in self.snippets.values() if s.visibility == Visibility.PUBLIC]
        public.sort(key=lambda s: s.created_at, reverse=True)
        return public[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """إحصائيات المنصة"""
        total_snippets = len(self.snippets)
        total_collections = len(self.collections)
        
        by_language = {}
        for lang in Language:
            count = len([s for s in self.snippets.values() if s.language == lang])
            if count > 0:
                by_language[lang.value] = count
        
        total_views = sum(s.view_count for s in self.snippets.values())
        total_downloads = sum(s.download_count for s in self.snippets.values())
        
        return {
            "total_snippets": total_snippets,
            "total_collections": total_collections,
            "by_language": by_language,
            "total_views": total_views,
            "total_downloads": total_downloads,
            "public_snippets": len([s for s in self.snippets.values() if s.visibility == Visibility.PUBLIC]),
            "private_snippets": len([s for s in self.snippets.values() if s.visibility == Visibility.PRIVATE])
        }
    
    def download_snippet(self, snippet_id: str, user_id: str = None) -> Optional[Dict]:
        """تحميل مقطع"""
        snippet = self.get_snippet(snippet_id, user_id)
        if snippet:
            snippet.increment_downloads()
            return {
                "title": snippet.title,
                "language": snippet.language.value,
                "code": snippet.code,
                "filename": f"{snippet.title.replace(' ', '_')}.{snippet.language.value}"
            }
        return None
