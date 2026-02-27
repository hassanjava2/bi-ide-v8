"""
User Preferences and Profile Management
Store and learn user preferences from interactions
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import defaultdict


@dataclass
class UserProfile:
    """User profile with preferences and learned attributes."""
    user_id: str
    created_at: datetime = None
    updated_at: datetime = None
    
    # Language preferences
    preferred_language: str = 'en'  # 'en', 'ar', 'auto'
    language_proficiency: Dict[str, float] = None  # {'en': 0.9, 'ar': 0.7}
    
    # Communication style
    response_style: str = 'balanced'  # 'concise', 'balanced', 'detailed'
    code_style: str = 'clean'  # 'minimal', 'clean', 'documented'
    explanation_depth: str = 'moderate'  # 'brief', 'moderate', 'thorough'
    
    # Expertise areas
    expertise_areas: List[str] = None  # ['python', 'ml', 'web_dev']
    skill_levels: Dict[str, str] = None  # {'python': 'advanced', 'js': 'intermediate'}
    
    # Topics of interest
    interests: List[str] = None
    disliked_topics: List[str] = None
    
    # Interaction history
    total_conversations: int = 0
    total_messages: int = 0
    last_active: datetime = None
    
    # Custom preferences
    custom_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.language_proficiency is None:
            self.language_proficiency = {}
        if self.expertise_areas is None:
            self.expertise_areas = []
        if self.skill_levels is None:
            self.skill_levels = {}
        if self.interests is None:
            self.interests = []
        if self.disliked_topics is None:
            self.disliked_topics = []
        if self.custom_preferences is None:
            self.custom_preferences = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'preferred_language': self.preferred_language,
            'language_proficiency': self.language_proficiency,
            'response_style': self.response_style,
            'code_style': self.code_style,
            'explanation_depth': self.explanation_depth,
            'expertise_areas': self.expertise_areas,
            'skill_levels': self.skill_levels,
            'interests': self.interests,
            'disliked_topics': self.disliked_topics,
            'total_conversations': self.total_conversations,
            'total_messages': self.total_messages,
            'last_active': self.last_active.isoformat() if self.last_active else None,
            'custom_preferences': self.custom_preferences
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """Create from dictionary."""
        profile = cls(user_id=data['user_id'])
        
        if data.get('created_at'):
            profile.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('updated_at'):
            profile.updated_at = datetime.fromisoformat(data['updated_at'])
        
        profile.preferred_language = data.get('preferred_language', 'en')
        profile.language_proficiency = data.get('language_proficiency', {})
        profile.response_style = data.get('response_style', 'balanced')
        profile.code_style = data.get('code_style', 'clean')
        profile.explanation_depth = data.get('explanation_depth', 'moderate')
        profile.expertise_areas = data.get('expertise_areas', [])
        profile.skill_levels = data.get('skill_levels', {})
        profile.interests = data.get('interests', [])
        profile.disliked_topics = data.get('disliked_topics', [])
        profile.total_conversations = data.get('total_conversations', 0)
        profile.total_messages = data.get('total_messages', 0)
        
        if data.get('last_active'):
            profile.last_active = datetime.fromisoformat(data['last_active'])
        
        profile.custom_preferences = data.get('custom_preferences', {})
        
        return profile


class PreferenceStore:
    """Store for user preferences and profiles."""
    
    def __init__(self, db_path: str = 'data/user_preferences.db'):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
    
    def _get_connection(self):
        """Get database connection (thread-safe)."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _init_db(self):
        """Initialize database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interaction_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_patterns ON interaction_patterns(user_id, pattern_type)
        ''')
        
        conn.commit()
        cursor.close()
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT profile_data FROM user_profiles WHERE user_id = ?',
            (user_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        
        if row:
            data = json.loads(row['profile_data'])
            return UserProfile.from_dict(data)
        return None
    
    def create_profile(self, user_id: str, **kwargs) -> UserProfile:
        """Create new user profile."""
        profile = UserProfile(user_id=user_id, **kwargs)
        self.save_profile(profile)
        return profile
    
    def save_profile(self, profile: UserProfile) -> None:
        """Save or update user profile."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        profile.updated_at = datetime.now()
        data = json.dumps(profile.to_dict())
        
        cursor.execute('''
            INSERT INTO user_profiles (user_id, profile_data, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                profile_data = excluded.profile_data,
                updated_at = excluded.updated_at
        ''', (profile.user_id, data, profile.created_at, profile.updated_at))
        
        conn.commit()
        cursor.close()
    
    def update_preference(
        self,
        user_id: str,
        preference_name: str,
        value: Any
    ) -> bool:
        """Update a specific preference."""
        profile = self.get_profile(user_id)
        
        if profile is None:
            profile = self.create_profile(user_id)
        
        if hasattr(profile, preference_name):
            setattr(profile, preference_name, value)
            self.save_profile(profile)
            return True
        elif preference_name in profile.custom_preferences:
            profile.custom_preferences[preference_name] = value
            self.save_profile(profile)
            return True
        
        return False
    
    def delete_profile(self, user_id: str) -> bool:
        """Delete user profile."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM user_profiles WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM interaction_patterns WHERE user_id = ?', (user_id,))
        
        conn.commit()
        success = cursor.rowcount > 0
        cursor.close()
        
        return success
    
    def get_all_profiles(self) -> List[UserProfile]:
        """Get all user profiles."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT profile_data FROM user_profiles')
        rows = cursor.fetchall()
        cursor.close()
        
        return [UserProfile.from_dict(json.loads(row['profile_data'])) for row in rows]


class PreferenceLearner:
    """Learn user preferences from interactions."""
    
    def __init__(self, store: PreferenceStore):
        self.store = store
        self.language_detector = None  # Lazy import
    
    def learn_from_message(
        self,
        user_id: str,
        message: str,
        role: str = 'user'
    ) -> Dict[str, Any]:
        """
        Learn preferences from a single message.
        
        Returns:
            Dictionary of learned attributes
        """
        profile = self.store.get_profile(user_id)
        if profile is None:
            profile = self.store.create_profile(user_id)
        
        learned = {}
        
        # Detect language
        detected_lang = self._detect_language(message)
        if detected_lang:
            learned['language'] = detected_lang
            self._update_language_preference(profile, detected_lang)
        
        # Detect code
        if self._contains_code(message):
            code_language = self._detect_code_language(message)
            if code_language:
                learned['code_language'] = code_language
                self._update_expertise(profile, code_language)
        
        # Detect topics
        topics = self._extract_topics(message)
        if topics:
            learned['topics'] = topics
            for topic in topics:
                if topic not in profile.interests:
                    profile.interests.append(topic)
        
        # Update activity
        profile.total_messages += 1
        profile.last_active = datetime.now()
        
        self.store.save_profile(profile)
        
        return learned
    
    def learn_from_conversation(
        self,
        user_id: str,
        messages: List[Dict],
        conversation_metadata: Dict = None
    ) -> Dict[str, Any]:
        """
        Learn preferences from a conversation.
        
        Returns:
            Dictionary of learned attributes
        """
        profile = self.store.get_profile(user_id)
        if profile is None:
            profile = self.store.create_profile(user_id)
        
        learned = {
            'languages': set(),
            'topics': set(),
            'code_languages': set()
        }
        
        # Analyze all messages
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                
                # Language
                lang = self._detect_language(content)
                if lang:
                    learned['languages'].add(lang)
                
                # Topics
                topics = self._extract_topics(content)
                learned['topics'].update(topics)
                
                # Code
                if self._contains_code(content):
                    code_lang = self._detect_code_language(content)
                    if code_lang:
                        learned['code_languages'].add(code_lang)
        
        # Update profile
        for lang in learned['languages']:
            self._update_language_preference(profile, lang)
        
        for topic in learned['topics']:
            if topic not in profile.interests:
                profile.interests.append(topic)
        
        for code_lang in learned['code_languages']:
            self._update_expertise(profile, code_lang)
        
        # Update conversation count
        profile.total_conversations += 1
        profile.last_active = datetime.now()
        
        self.store.save_profile(profile)
        
        # Convert sets to lists for return
        return {k: list(v) if isinstance(v, set) else v for k, v in learned.items()}
    
    def learn_feedback(
        self,
        user_id: str,
        interaction_id: str,
        feedback: Dict[str, Any]
    ) -> None:
        """Learn from explicit user feedback."""
        profile = self.store.get_profile(user_id)
        if profile is None:
            return
        
        # Update style preferences based on feedback
        if feedback.get('response_too_long'):
            if profile.response_style == 'detailed':
                profile.response_style = 'balanced'
            elif profile.response_style == 'balanced':
                profile.response_style = 'concise'
        
        if feedback.get('response_too_short'):
            if profile.response_style == 'concise':
                profile.response_style = 'balanced'
            elif profile.response_style == 'balanced':
                profile.response_style = 'detailed'
        
        if feedback.get('code_style_preference'):
            profile.code_style = feedback['code_style_preference']
        
        if feedback.get('explanation_depth'):
            profile.explanation_depth = feedback['explanation_depth']
        
        # Update custom preferences
        for key, value in feedback.items():
            if key not in ['response_too_long', 'response_too_short', 'code_style_preference', 'explanation_depth']:
                profile.custom_preferences[f"feedback_{key}"] = value
        
        self.store.save_profile(profile)
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language of text."""
        # Simple detection based on character ranges
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return None
        
        if arabic_chars / total_chars > 0.5:
            return 'ar'
        else:
            return 'en'
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code."""
        code_indicators = [
            'def ', 'class ', 'import ', 'return ',
            'function', 'const ', 'let ', 'var ',
            '```', '`', '#', '//', '/*'
        ]
        return any(indicator in text for indicator in code_indicators)
    
    def _detect_code_language(self, text: str) -> Optional[str]:
        """Detect programming language in code."""
        patterns = {
            'python': [r'\bdef\s+\w+\s*\(', r'\bimport\s+\w+', r'\bprint\s*\(', r':\s*\n\s+'],
            'javascript': [r'\bconst\s+', r'\bfunction\s+', r'\bconsole\.log', r'=>'],
            'typescript': [r':\s*(string|number|boolean)', r'interface\s+'],
            'java': [r'\bpublic\s+class', r'\bSystem\.out\.print'],
            'cpp': [r'#include', r'std::', r'cout\s*<<'],
            'sql': [r'\bSELECT\b', r'\bFROM\b', r'\bWHERE\b', r'\bINSERT\b'],
        }
        
        import re
        scores = {}
        for lang, patterns_list in patterns.items():
            scores[lang] = sum(1 for p in patterns_list if re.search(p, text, re.IGNORECASE))
        
        if scores:
            best_lang = max(scores, key=scores.get)
            if scores[best_lang] > 0:
                return best_lang
        
        return None
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text."""
        # Simple keyword-based extraction
        topic_keywords = {
            'machine_learning': ['ml', 'machine learning', 'model', 'training', 'neural', 'ai'],
            'web_development': ['web', 'html', 'css', 'react', 'vue', 'angular', 'frontend'],
            'backend': ['server', 'api', 'database', 'backend', 'rest', 'graphql'],
            'mobile': ['mobile', 'android', 'ios', 'flutter', 'react native'],
            'devops': ['docker', 'kubernetes', 'ci/cd', 'deployment', 'aws', 'cloud'],
            'data_science': ['pandas', 'numpy', 'data', 'visualization', 'analysis'],
        }
        
        text_lower = text.lower()
        found_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found_topics.append(topic)
        
        return found_topics
    
    def _update_language_preference(self, profile: UserProfile, lang: str) -> None:
        """Update language preference."""
        if lang not in profile.language_proficiency:
            profile.language_proficiency[lang] = 0.0
        
        # Increase proficiency score
        profile.language_proficiency[lang] = min(1.0, profile.language_proficiency[lang] + 0.05)
        
        # Update preferred language if Arabic is detected frequently
        if lang == 'ar' and profile.language_proficiency.get('ar', 0) > 0.3:
            profile.preferred_language = 'ar'
    
    def _update_expertise(self, profile: UserProfile, skill: str) -> None:
        """Update expertise area."""
        if skill not in profile.expertise_areas:
            profile.expertise_areas.append(skill)
        
        # Update skill level
        current_level = profile.skill_levels.get(skill, 'beginner')
        levels = ['beginner', 'intermediate', 'advanced', 'expert']
        
        if current_level in levels:
            idx = levels.index(current_level)
            if idx < len(levels) - 1:
                profile.skill_levels[skill] = levels[idx + 1]


def get_user_context(user_id: str, store: PreferenceStore) -> Dict[str, Any]:
    """
    Get user context for personalization.
    
    Returns:
        Dictionary with user preferences and context
    """
    profile = store.get_profile(user_id)
    
    if profile is None:
        return {
            'user_id': user_id,
            'preferred_language': 'auto',
            'response_style': 'balanced',
            'expertise_areas': [],
            'is_new_user': True
        }
    
    return {
        'user_id': user_id,
        'preferred_language': profile.preferred_language,
        'response_style': profile.response_style,
        'code_style': profile.code_style,
        'explanation_depth': profile.explanation_depth,
        'expertise_areas': profile.expertise_areas,
        'skill_levels': profile.skill_levels,
        'interests': profile.interests,
        'total_conversations': profile.total_conversations,
        'is_new_user': profile.total_conversations < 3
    }


if __name__ == '__main__':
    print("User Preferences Module Demo")
    print("="*50)
    
    # Create store and learner
    store = PreferenceStore()
    learner = PreferenceLearner(store)
    
    # Learn from messages
    learner.learn_from_message('user_1', 'كيف يمكنني تعلم بايثون؟', 'user')
    learner.learn_from_message('user_1', 'def hello():\n    print("Hello")', 'user')
    
    # Get profile
    profile = store.get_profile('user_1')
    print(f"User Profile:")
    print(f"  Language: {profile.preferred_language}")
    print(f"  Expertise: {profile.expertise_areas}")
    print(f"  Interests: {profile.interests}")
    
    # Get context
    context = get_user_context('user_1', store)
    print(f"\nUser Context: {context}")
