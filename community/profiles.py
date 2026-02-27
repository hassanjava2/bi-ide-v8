"""
User Profiles - ملفات المستخدمين الشخصية

المميزات:
- Bio, skills, portfolio
- نظام السمعة (Reputation system)
- Activity feed
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class ReputationAction(Enum):
    """أنواع إجراءات السمعة"""
    POST_CREATED = "post_created"
    POST_UPVOTED = "post_upvoted"
    ANSWER_ACCEPTED = "answer_accepted"
    CODE_SHARED = "code_shared"
    ARTICLE_PUBLISHED = "article_published"
    COMMENT_POSTED = "comment_posted"
    BUG_REPORTED = "bug_reported"
    CONTRIBUTION = "contribution"


@dataclass
class Badge:
    """شارة"""
    id: str
    name: str
    description: str
    icon: str = ""
    color: str = "#4CAF50"
    earned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "color": self.color,
            "earned_at": self.earned_at.isoformat()
        }


@dataclass
class ActivityItem:
    """عنصر نشاط"""
    id: str
    activity_type: str
    description: str
    reference_id: Optional[str] = None  # معرف المرجع (post, snippet, etc.)
    reference_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "activity_type": self.activity_type,
            "description": self.description,
            "reference_id": self.reference_id,
            "reference_type": self.reference_type,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class PortfolioItem:
    """عنصر في المحفظة"""
    id: str
    title: str
    description: str = ""
    project_url: str = ""
    image_urls: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "project_url": self.project_url,
            "image_urls": self.image_urls,
            "technologies": self.technologies,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class UserProfile:
    """
    ملف المستخدم الشخصي الموسع
    """
    user_id: str
    username: str
    
    # Basic Info
    display_name: str = ""
    bio: str = ""                     # نبذة
    location: str = ""
    website: str = ""
    company: str = ""
    job_title: str = ""
    
    # Profile Media
    avatar_url: str = ""
    cover_image_url: str = ""
    
    # Skills & Expertise
    skills: List[str] = field(default_factory=list)
    expertise_areas: List[str] = field(default_factory=list)
    
    # Social Links
    github_username: str = ""
    linkedin_url: str = ""
    twitter_username: str = ""
    
    # Portfolio
    portfolio: List[PortfolioItem] = field(default_factory=list)
    
    # Preferences
    is_public: bool = True
    show_email: bool = False
    show_activity: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_skill(self, skill: str):
        """إضافة مهارة"""
        skill = skill.strip().lower()
        if skill and skill not in self.skills:
            self.skills.append(skill)
            self.updated_at = datetime.now(timezone.utc)
    
    def remove_skill(self, skill: str):
        """إزالة مهارة"""
        skill = skill.strip().lower()
        if skill in self.skills:
            self.skills.remove(skill)
            self.updated_at = datetime.now(timezone.utc)
    
    def add_portfolio_item(self, title: str, description: str = "",
                          project_url: str = "", 
                          technologies: List[str] = None) -> PortfolioItem:
        """إضافة عنصر للمحفظة"""
        item = PortfolioItem(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            project_url=project_url,
            technologies=technologies or []
        )
        
        self.portfolio.append(item)
        self.updated_at = datetime.now(timezone.utc)
        return item
    
    def update_activity(self):
        """تحديث وقت النشاط الأخير"""
        self.last_active = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name or self.username,
            "bio": self.bio,
            "location": self.location,
            "website": self.website,
            "company": self.company,
            "job_title": self.job_title,
            "avatar_url": self.avatar_url,
            "skills": self.skills,
            "expertise_areas": self.expertise_areas,
            "github_username": self.github_username,
            "linkedin_url": self.linkedin_url,
            "twitter_username": self.twitter_username,
            "portfolio": [p.to_dict() for p in self.portfolio],
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat()
        }


class ReputationSystem:
    """
    نظام السمعة والنقاط
    """
    
    # Points for each action
    POINTS = {
        ReputationAction.POST_CREATED: 5,
        ReputationAction.POST_UPVOTED: 10,
        ReputationAction.ANSWER_ACCEPTED: 25,
        ReputationAction.CODE_SHARED: 15,
        ReputationAction.ARTICLE_PUBLISHED: 30,
        ReputationAction.COMMENT_POSTED: 2,
        ReputationAction.BUG_REPORTED: 10,
        ReputationAction.CONTRIBUTION: 50
    }
    
    # Badges thresholds
    BADGES = {
        "newcomer": {"name": "Newcomer", "description": "Joined the community", "threshold": 0},
        "contributor": {"name": "Contributor", "description": "Made first contribution", "threshold": 50},
        "active_member": {"name": "Active Member", "description": "100 reputation points", "threshold": 100},
        "expert": {"name": "Expert", "description": "500 reputation points", "threshold": 500},
        "guru": {"name": "Guru", "description": "1000 reputation points", "threshold": 1000},
        "legend": {"name": "Legend", "description": "5000 reputation points", "threshold": 5000}
    }
    
    def __init__(self):
        self.user_reputation: Dict[str, int] = {}
        self.user_badges: Dict[str, List[Badge]] = {}
        self.action_history: List[Dict] = []
    
    def get_reputation(self, user_id: str) -> int:
        """الحصول على سمعة المستخدم"""
        return self.user_reputation.get(user_id, 0)
    
    def add_points(self, user_id: str, action: ReputationAction,
                  reference_id: str = None, metadata: Dict = None):
        """إضافة نقاط"""
        points = self.POINTS.get(action, 0)
        
        # Update reputation
        current = self.user_reputation.get(user_id, 0)
        self.user_reputation[user_id] = current + points
        
        # Record action
        self.action_history.append({
            "user_id": user_id,
            "action": action.value,
            "points": points,
            "reference_id": reference_id,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Check for new badges
        self._check_badges(user_id)
        
        return points
    
    def _check_badges(self, user_id: str):
        """التحقق من استحقاق الشارات"""
        reputation = self.get_reputation(user_id)
        
        if user_id not in self.user_badges:
            self.user_badges[user_id] = []
        
        earned_badge_ids = {b.id for b in self.user_badges[user_id]}
        
        for badge_id, badge_info in self.BADGES.items():
            if badge_id not in earned_badge_ids and reputation >= badge_info["threshold"]:
                badge = Badge(
                    id=badge_id,
                    name=badge_info["name"],
                    description=badge_info["description"]
                )
                self.user_badges[user_id].append(badge)
    
    def get_badges(self, user_id: str) -> List[Badge]:
        """الحصول على شارات المستخدم"""
        return self.user_badges.get(user_id, [])
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """لوحة المتصدرين"""
        sorted_users = sorted(
            self.user_reputation.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {
                "user_id": user_id,
                "reputation": points,
                "rank": i + 1
            }
            for i, (user_id, points) in enumerate(sorted_users[:limit])
        ]


class ActivityFeed:
    """
    خلاصة النشاطات
    """
    
    def __init__(self):
        self.activities: Dict[str, List[ActivityItem]] = {}  # user_id -> activities
        self.global_feed: List[ActivityItem] = []
    
    def add_activity(self, user_id: str, activity_type: str, description: str,
                    reference_id: str = None, reference_type: str = None,
                    metadata: Dict = None) -> ActivityItem:
        """إضافة نشاط"""
        activity = ActivityItem(
            id=str(uuid.uuid4()),
            activity_type=activity_type,
            description=description,
            reference_id=reference_id,
            reference_type=reference_type,
            metadata=metadata or {}
        )
        
        # Add to user feed
        if user_id not in self.activities:
            self.activities[user_id] = []
        self.activities[user_id].insert(0, activity)
        
        # Add to global feed
        self.global_feed.insert(0, activity)
        
        return activity
    
    def get_user_feed(self, user_id: str, limit: int = 20) -> List[Dict]:
        """الحصول على خلاصة المستخدم"""
        user_activities = self.activities.get(user_id, [])
        return [a.to_dict() for a in user_activities[:limit]]
    
    def get_global_feed(self, limit: int = 50) -> List[Dict]:
        """الحصول على الخلاصة العامة"""
        return [a.to_dict() for a in self.global_feed[:limit]]
    
    def get_following_feed(self, user_ids: List[str], limit: int = 50) -> List[Dict]:
        """الحصول على خلاصة المستخدمين المتابعين"""
        all_activities = []
        for user_id in user_ids:
            all_activities.extend(self.activities.get(user_id, []))
        
        # Sort by time
        all_activities.sort(key=lambda x: x.created_at, reverse=True)
        
        return [a.to_dict() for a in all_activities[:limit]]
