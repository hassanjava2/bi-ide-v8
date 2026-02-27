"""
BI-IDE v8 - Event Tracking and Analytics Module
User actions, funnel analysis, and retention metrics
Integrates with Mixpanel and Amplitude
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
from uuid import uuid4

import aiohttp

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of trackable events"""
    # User lifecycle
    USER_SIGNUP = "user_signup"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    
    # Feature usage
    PAGE_VIEW = "page_view"
    FEATURE_USED = "feature_used"
    AI_REQUEST = "ai_request"
    QUERY_EXECUTED = "query_executed"
    REPORT_GENERATED = "report_generated"
    EXPORT_CREATED = "export_created"
    
    # Business events
    SUBSCRIPTION_STARTED = "subscription_started"
    SUBSCRIPTION_CANCELLED = "subscription_cancelled"
    PAYMENT_SUCCEEDED = "payment_succeeded"
    PAYMENT_FAILED = "payment_failed"
    
    # Engagement
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TUTORIAL_COMPLETED = "tutorial_completed"
    ONBOARDING_STEP = "onboarding_step"
    
    # Errors
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_ISSUE = "performance_issue"


@dataclass
class Event:
    """Analytics event structure"""
    event_type: str
    user_id: Optional[str]
    anonymous_id: Optional[str]
    timestamp: datetime
    properties: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: Optional[str] = None


@dataclass
class UserProfile:
    """User profile for analytics"""
    user_id: str
    created_at: datetime
    traits: Dict[str, Any] = field(default_factory=dict)
    groups: Dict[str, Any] = field(default_factory=dict)


class AnalyticsProvider(ABC):
    """Abstract analytics provider"""
    
    @abstractmethod
    async def track(self, event: Event) -> bool:
        pass
    
    @abstractmethod
    async def identify(self, user_profile: UserProfile) -> bool:
        pass
    
    @abstractmethod
    async def group(self, user_id: str, group_id: str, 
                    traits: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    async def flush(self) -> bool:
        pass


class MixpanelProvider(AnalyticsProvider):
    """Mixpanel analytics integration"""
    
    API_URL = "https://api.mixpanel.com"
    
    def __init__(self, api_token: str, api_secret: Optional[str] = None):
        self.api_token = api_token
        self.api_secret = api_secret
        self._session: Optional[aiohttp.ClientSession] = None
        self._batch_queue: List[Dict] = []
        self._batch_size = 50
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def track(self, event: Event) -> bool:
        """Track event to Mixpanel"""
        payload = {
            "event": event.event_type,
            "properties": {
                "token": self.api_token,
                "time": int(event.timestamp.timestamp()),
                "distinct_id": event.user_id or event.anonymous_id,
                "$insert_id": event.event_id,
                **event.properties
            }
        }
        
        self._batch_queue.append(payload)
        
        if len(self._batch_queue) >= self._batch_size:
            return await self.flush()
        
        return True
    
    async def identify(self, user_profile: UserProfile) -> bool:
        """Identify user in Mixpanel"""
        payload = {
            "$token": self.api_token,
            "$distinct_id": user_profile.user_id,
            "$time": int(user_profile.created_at.timestamp()),
            "$set": user_profile.traits
        }
        
        session = await self._get_session()
        
        try:
            async with session.post(
                f"{self.API_URL}/engage",
                json=[payload],
                headers={"Accept": "text/plain"}
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Mixpanel identify error: {e}")
            return False
    
    async def group(self, user_id: str, group_id: str,
                    traits: Dict[str, Any]) -> bool:
        """Set group properties in Mixpanel"""
        payload = {
            "$token": self.api_token,
            "$group_key": "company",
            "$group_id": group_id,
            "$set": traits
        }
        
        session = await self._get_session()
        
        try:
            async with session.post(
                f"{self.API_URL}/groups",
                json=[payload]
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Mixpanel group error: {e}")
            return False
    
    async def flush(self) -> bool:
        """Flush batch queue"""
        if not self._batch_queue:
            return True
        
        session = await self._get_session()
        batch = self._batch_queue[:]
        self._batch_queue = []
        
        try:
            async with session.post(
                f"{self.API_URL}/import",
                json=batch,
                auth=aiohttp.BasicAuth(self.api_secret, '') if self.api_secret else None
            ) as response:
                if response.status == 200:
                    logger.debug(f"Flushed {len(batch)} events to Mixpanel")
                    return True
                else:
                    logger.error(f"Mixpanel flush failed: {response.status}")
                    # Re-queue failed events
                    self._batch_queue.extend(batch)
                    return False
        except Exception as e:
            logger.error(f"Mixpanel flush error: {e}")
            self._batch_queue.extend(batch)
            return False


class AmplitudeProvider(AnalyticsProvider):
    """Amplitude analytics integration"""
    
    API_URL = "https://api2.amplitude.com"
    
    def __init__(self, api_key: str, secret_key: Optional[str] = None):
        self.api_key = api_key
        self.secret_key = secret_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._event_queue: List[Dict] = []
        self._batch_size = 100
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def track(self, event: Event) -> bool:
        """Track event to Amplitude"""
        payload = {
            "user_id": event.user_id,
            "device_id": event.anonymous_id,
            "event_type": event.event_type,
            "time": int(event.timestamp.timestamp() * 1000),
            "event_id": abs(hash(event.event_id)) % (10 ** 9),
            "session_id": event.session_id,
            "event_properties": event.properties,
            "user_properties": event.context.get('user_traits', {}),
            "app_version": event.context.get('app_version'),
            "platform": event.context.get('platform'),
            "os_name": event.context.get('os_name'),
            "os_version": event.context.get('os_version'),
            "device_model": event.context.get('device_model'),
            "carrier": event.context.get('carrier'),
            "country": event.context.get('country'),
            "region": event.context.get('region'),
            "city": event.context.get('city'),
            "dma": event.context.get('dma'),
        }
        
        self._event_queue.append(payload)
        
        if len(self._event_queue) >= self._batch_size:
            return await self.flush()
        
        return True
    
    async def identify(self, user_profile: UserProfile) -> bool:
        """Identify user in Amplitude"""
        identification = {
            "user_id": user_profile.user_id,
            "user_properties": user_profile.traits
        }
        
        return await self._send_identify([identification])
    
    async def group(self, user_id: str, group_type: str,
                    group_properties: Dict[str, Any]) -> bool:
        """Set group properties in Amplitude"""
        # Amplitude handles groups via user properties
        return await self.identify(UserProfile(
            user_id=user_id,
            created_at=datetime.now(),
            traits={f"[Amplitude] Group: {group_type}": group_properties}
        ))
    
    async def flush(self) -> bool:
        """Flush event queue"""
        if not self._event_queue:
            return True
        
        session = await self._get_session()
        events = self._event_queue[:]
        self._event_queue = []
        
        data = {
            "api_key": self.api_key,
            "events": events
        }
        
        try:
            async with session.post(
                f"{self.API_URL}/2/httpapi",
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('code') == 200:
                        logger.debug(f"Flushed {len(events)} events to Amplitude")
                        return True
                
                logger.error(f"Amplitude flush failed: {response.status}")
                self._event_queue.extend(events)
                return False
        except Exception as e:
            logger.error(f"Amplitude flush error: {e}")
            self._event_queue.extend(events)
            return False
    
    async def _send_identify(self, identifications: List[Dict]) -> bool:
        """Send identification to Amplitude"""
        session = await self._get_session()
        
        data = {
            "api_key": self.api_key,
            "identification": json.dumps(identifications)
        }
        
        try:
            async with session.post(
                f"{self.API_URL}/identify",
                data=data
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Amplitude identify error: {e}")
            return False


class EventTracker:
    """Main event tracking coordinator"""
    
    def __init__(self):
        self.providers: List[AnalyticsProvider] = []
        self._user_profiles: Dict[str, UserProfile] = {}
        self._sessions: Dict[str, Dict] = {}
        self._funnel_definitions: Dict[str, List[str]] = {}
        self._flush_interval = 30  # seconds
        self._flush_task: Optional[asyncio.Task] = None
    
    def add_provider(self, provider: AnalyticsProvider):
        """Add analytics provider"""
        self.providers.append(provider)
    
    async def initialize(self):
        """Initialize event tracker"""
        self._flush_task = asyncio.create_task(self._periodic_flush())
    
    async def close(self):
        """Cleanup resources"""
        if self._flush_task:
            self._flush_task.cancel()
        
        # Final flush
        await self.flush_all()
    
    async def _periodic_flush(self):
        """Periodic flush task"""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                await self.flush_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic flush error: {e}")
    
    async def flush_all(self):
        """Flush all providers"""
        for provider in self.providers:
            try:
                await provider.flush()
            except Exception as e:
                logger.error(f"Flush error for {provider.__class__.__name__}: {e}")
    
    # Event tracking methods
    
    async def track(self, event_type: EventType, user_id: Optional[str] = None,
                    properties: Optional[Dict] = None,
                    context: Optional[Dict] = None) -> bool:
        """Track an event"""
        event = Event(
            event_type=event_type.value,
            user_id=user_id,
            anonymous_id=self._get_anonymous_id(context),
            timestamp=datetime.now(),
            properties=properties or {},
            context=context or {},
            session_id=self._get_session_id(user_id, context)
        )
        
        # Track to all providers
        results = await asyncio.gather(*[
            provider.track(event)
            for provider in self.providers
        ], return_exceptions=True)
        
        return any(r is True for r in results if not isinstance(r, Exception))
    
    async def track_page_view(self, user_id: Optional[str], path: str,
                               referrer: Optional[str] = None,
                               title: Optional[str] = None,
                               properties: Optional[Dict] = None) -> bool:
        """Track page view"""
        props = {
            "path": path,
            "referrer": referrer,
            "title": title,
            **(properties or {})
        }
        
        return await self.track(EventType.PAGE_VIEW, user_id, props)
    
    async def track_ai_request(self, user_id: str, model: str,
                               tokens_used: int, latency_ms: float,
                               cost_usd: float,
                               properties: Optional[Dict] = None) -> bool:
        """Track AI API request"""
        props = {
            "model": model,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            **(properties or {})
        }
        
        return await self.track(EventType.AI_REQUEST, user_id, props)
    
    async def identify(self, user_id: str, traits: Dict[str, Any],
                       created_at: Optional[datetime] = None) -> bool:
        """Identify user with traits"""
        profile = UserProfile(
            user_id=user_id,
            created_at=created_at or datetime.now(),
            traits=traits
        )
        
        self._user_profiles[user_id] = profile
        
        results = await asyncio.gather(*[
            provider.identify(profile)
            for provider in self.providers
        ], return_exceptions=True)
        
        return any(r is True for r in results if not isinstance(r, Exception))
    
    # Session management
    
    def _get_anonymous_id(self, context: Optional[Dict]) -> Optional[str]:
        """Get or create anonymous ID"""
        if context and 'anonymous_id' in context:
            return context['anonymous_id']
        return None
    
    def _get_session_id(self, user_id: Optional[str],
                        context: Optional[Dict]) -> Optional[str]:
        """Get session ID"""
        if context and 'session_id' in context:
            return context['session_id']
        
        if user_id and user_id in self._sessions:
            return self._sessions[user_id].get('id')
        
        return None
    
    async def start_session(self, user_id: str,
                            properties: Optional[Dict] = None) -> str:
        """Start a new session"""
        session_id = str(uuid4())
        
        self._sessions[user_id] = {
            'id': session_id,
            'start_time': datetime.now(),
            'events': 0
        }
        
        await self.track(EventType.SESSION_START, user_id, {
            'session_id': session_id,
            **(properties or {})
        })
        
        return session_id
    
    async def end_session(self, user_id: str,
                          properties: Optional[Dict] = None) -> bool:
        """End a session"""
        session = self._sessions.get(user_id)
        
        if session:
            duration = (datetime.now() - session['start_time']).total_seconds()
            
            props = {
                'session_id': session['id'],
                'duration_seconds': duration,
                'event_count': session['events'],
                **(properties or {})
            }
            
            del self._sessions[user_id]
            
            return await self.track(EventType.SESSION_END, user_id, props)
        
        return False
    
    # Funnel analysis
    
    def define_funnel(self, funnel_name: str, steps: List[str]):
        """Define a funnel"""
        self._funnel_definitions[funnel_name] = steps
    
    def analyze_funnel(self, funnel_name: str,
                       user_events: List[str]) -> Dict[str, Any]:
        """Analyze user progression through funnel"""
        if funnel_name not in self._funnel_definitions:
            return {"error": "Funnel not defined"}
        
        steps = self._funnel_definitions[funnel_name]
        user_events_set = set(user_events)
        
        results = []
        total_completed = 0
        
        for i, step in enumerate(steps):
            completed = step in user_events_set
            
            if i == 0:
                conversion_rate = 100.0 if completed else 0.0
            else:
                prev_completed = results[i-1]['completed'] if results else False
                conversion_rate = 100.0 if (completed and prev_completed) else 0.0
            
            if completed:
                total_completed += 1
            
            results.append({
                'step': step,
                'completed': completed,
                'conversion_rate': conversion_rate
            })
        
        return {
            'funnel_name': funnel_name,
            'total_steps': len(steps),
            'steps_completed': total_completed,
            'overall_conversion': (total_completed / len(steps) * 100) if steps else 0,
            'step_results': results
        }
    
    # Retention metrics
    
    def calculate_retention(self, signup_date: datetime,
                           activity_dates: List[datetime],
                           periods: List[int] = None) -> Dict[int, bool]:
        """Calculate retention for specific periods"""
        periods = periods or [1, 3, 7, 14, 30]
        retention = {}
        
        for period in periods:
            target_date = signup_date + timedelta(days=period)
            active = any(
                (target_date - activity).days <= 1 and
                (target_date - activity).days >= -1
                for activity in activity_dates
            )
            retention[period] = active
        
        return retention
    
    async def track_cohort_retention(self, cohort_date: datetime,
                                      user_ids: List[str],
                                      active_user_ids: List[str]):
        """Track cohort retention"""
        cohort_size = len(user_ids)
        active_count = len(set(user_ids) & set(active_user_ids))
        
        await self.track(
            EventType("cohort_retention"),
            None,
            {
                'cohort_date': cohort_date.isoformat(),
                'cohort_size': cohort_size,
                'active_users': active_count,
                'retention_rate': active_count / cohort_size if cohort_size > 0 else 0
            }
        )


# Global tracker instance
_tracker: Optional[EventTracker] = None


def init_tracker() -> EventTracker:
    """Initialize global event tracker"""
    global _tracker
    _tracker = EventTracker()
    return _tracker


def get_tracker() -> EventTracker:
    """Get global event tracker"""
    if _tracker is None:
        raise RuntimeError("Event tracker not initialized")
    return _tracker


# Convenience functions
async def track_event(event_type: EventType, user_id: Optional[str] = None,
                      properties: Optional[Dict] = None) -> bool:
    """Track event convenience function"""
    return await get_tracker().track(event_type, user_id, properties)


async def identify_user(user_id: str, traits: Dict[str, Any]) -> bool:
    """Identify user convenience function"""
    return await get_tracker().identify(user_id, traits)
