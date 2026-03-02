"""
نظام الكشافة - Scout System
الـ4 كشافة الذين يجلبون المعلومات من الخارج

🕵️ أنواع الكشافة:
- Tech Scout: رصد التقنيات
- Market Scout: مراقبة السوق
- Competitor Scout: تجسس المنافسين
- Opportunity Scout: صيد الفرص
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from enum import Enum
import asyncio
import json


class IntelType(Enum):
    """أنواع المعلومات الاستخباراتية"""
    TECH = "تقني"
    MARKET = "سوقي"
    COMPETITOR = "منافسة"
    OPPORTUNITY = "فرصة"
    THREAT = "تهديد"


@dataclass
class IntelReport:
    """تقرير استخباراتي"""
    intel_id: str
    scout_name: str
    intel_type: IntelType
    source: str
    content: str
    confidence: float  # 0-1
    urgency: int       # 1-10
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class TechScout:
    """
    🧪 كشاف التقنية
    
    يرصد:
    - إصدارات جديدة من المكتبات
    - تقنيات ثورية
    - ثغرات أمنية
    """
    
    def __init__(self):
        self.name = "Tech Scout"
        self.sources = [
            'github_trending',
            'pypi_new_releases',
            'security_advisories',
            'research_papers',
            'tech_news'
        ]
        self.known_packages: Dict[str, str] = {}
        print(f"🧪 {self.name} initialized")
    
    async def gather_intel(self) -> List[IntelReport]:
        """جمع معلومات تقنية"""
        reports = []
        
        # فحص GitHub Trending
        trending = await self._check_github_trending()
        for repo in trending:
            reports.append(IntelReport(
                intel_id=f"tech_{datetime.now(timezone.utc).timestamp()}",
                scout_name=self.name,
                intel_type=IntelType.TECH,
                source='github',
                content=f"مشروع متصاعد: {repo['name']} - {repo['description']}",
                confidence=0.85,
                urgency=5,
                timestamp=datetime.now(timezone.utc),
                metadata={'stars': repo.get('stars'), 'language': repo.get('lang')}
            ))
        
        # فحص الثغرات الأمنية
        vulnerabilities = await self._check_security_advisories()
        for vuln in vulnerabilities:
            reports.append(IntelReport(
                intel_id=f"sec_{vuln['cve']}",
                scout_name=self.name,
                intel_type=IntelType.THREAT,
                source='security',
                content=f"ثغرة خطيرة: {vuln['description']}",
                confidence=0.95,
                urgency=9,
                timestamp=datetime.now(timezone.utc),
                metadata={'severity': vuln['severity'], 'package': vuln['package']}
            ))
        
        return reports
    
    async def _check_github_trending(self) -> List[Dict]:
        """فحص GitHub Trending — Real HTTP fetch"""
        try:
            import urllib.request
            import urllib.error
            
            # Use GitHub API to get trending-like repos (most starred recently)
            url = "https://api.github.com/search/repositories?q=created:>2026-02-01&sort=stars&order=desc&per_page=5"
            req = urllib.request.Request(url, headers={
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'BI-IDE-Scout/1.0'
            })
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            repos = []
            for item in data.get('items', [])[:5]:
                repos.append({
                    'name': item.get('full_name', ''),
                    'description': item.get('description', 'No description'),
                    'stars': item.get('stargazers_count', 0),
                    'lang': item.get('language', 'Unknown'),
                    '_source': 'github_api'
                })
            
            print(f"✅ SCOUT: Fetched {len(repos)} trending repos from GitHub API")
            return repos
            
        except Exception as e:
            print(f"⚠️ SCOUT: GitHub API fetch failed ({e}) — returning empty list")
            return []
    
    async def _check_security_advisories(self) -> List[Dict]:
        """فحص التنبيهات الأمنية — Real HTTP fetch"""
        try:
            import urllib.request
            
            # Use GitHub Advisory Database API
            url = "https://api.github.com/advisories?per_page=3&severity=critical"
            req = urllib.request.Request(url, headers={
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'BI-IDE-Scout/1.0'
            })
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            advisories = []
            for item in data[:3] if isinstance(data, list) else []:
                advisories.append({
                    'cve': item.get('cve_id', 'UNKNOWN'),
                    'description': item.get('summary', 'No description')[:200],
                    'severity': item.get('severity', 'unknown'),
                    'package': item.get('vulnerabilities', [{}])[0].get('package', {}).get('name', 'unknown') if item.get('vulnerabilities') else 'unknown'
                })
            
            print(f"✅ SCOUT: Fetched {len(advisories)} security advisories")
            return advisories
            
        except Exception as e:
            print(f"⚠️ SCOUT: Security advisory fetch failed ({e}) — returning empty list")
            return []


class MarketScout:
    """
    📊 كشاف السوق
    
    يرصد بيانات حقيقية من:
    - GitHub Topics (ERP, IDE, AI, Cloud repos)
    - npm Registry (top package download stats)
    - PyPI (Python package trends)
    """
    
    def __init__(self):
        self.name = "Market Scout"
        self.monitored_segments = ['ERP', 'IDE', 'AI', 'Cloud']
        self.market_data: Dict = {}
        print(f"📊 {self.name} initialized")
    
    async def gather_intel(self) -> List[IntelReport]:
        """جمع معلومات السوق — Real HTTP"""
        reports = []
        
        # 1. GitHub topic trends (real repos per segment)
        for segment in self.monitored_segments:
            trend = await self._fetch_github_topic(segment.lower())
            if trend:
                reports.append(IntelReport(
                    intel_id=f"mkt_{segment}_{datetime.now(timezone.utc).timestamp()}",
                    scout_name=self.name,
                    intel_type=IntelType.MARKET,
                    source='github_topics',
                    content=f"سوق {segment}: {trend['total_repos']} repos, أعلى ⭐: {trend['top_repo']} ({trend['top_stars']} stars)",
                    confidence=0.85,
                    urgency=5,
                    timestamp=datetime.now(timezone.utc),
                    metadata={**trend, '_source': 'github_api'}
                ))
        
        # 2. npm trends — top packages
        npm_trends = await self._fetch_npm_trends()
        for pkg in npm_trends:
            reports.append(IntelReport(
                intel_id=f"npm_{pkg['name']}_{datetime.now(timezone.utc).timestamp()}",
                scout_name=self.name,
                intel_type=IntelType.MARKET,
                source='npm_registry',
                content=f"احتياج سوقي: {pkg['name']} — {pkg['description'][:80]}",
                confidence=0.75,
                urgency=4,
                timestamp=datetime.now(timezone.utc),
                metadata={**pkg, '_source': 'npm_registry'}
            ))
        
        return reports
    
    async def _fetch_github_topic(self, topic: str) -> Optional[Dict]:
        """GitHub topic search — real HTTP"""
        try:
            import urllib.request
            
            url = f"https://api.github.com/search/repositories?q=topic:{topic}&sort=stars&order=desc&per_page=3"
            req = urllib.request.Request(url, headers={
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'BI-IDE-Scout/1.0'
            })
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            items = data.get('items', [])
            top = items[0] if items else {}
            
            result = {
                'topic': topic,
                'total_repos': data.get('total_count', 0),
                'top_repo': top.get('full_name', 'N/A'),
                'top_stars': top.get('stargazers_count', 0),
                'top_description': top.get('description', '')[:150],
                'repos': [{'name': i.get('full_name'), 'stars': i.get('stargazers_count')} for i in items[:3]]
            }
            print(f"✅ SCOUT: Market data for '{topic}': {result['total_repos']} repos")
            return result
        except Exception as e:
            print(f"⚠️ SCOUT: GitHub topic fetch failed ({e})")
            return None
    
    async def _fetch_npm_trends(self) -> List[Dict]:
        """npm registry — popular packages as demand signals"""
        try:
            import urllib.request
            
            # Search npm for ERP/IDE related packages
            url = "https://registry.npmjs.org/-/v1/search?text=keywords:erp,ide,ai&size=3&popularity=1.0"
            req = urllib.request.Request(url, headers={
                'User-Agent': 'BI-IDE-Scout/1.0'
            })
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            packages = []
            for obj in data.get('objects', [])[:3]:
                pkg = obj.get('package', {})
                packages.append({
                    'name': pkg.get('name', ''),
                    'description': pkg.get('description', ''),
                    'version': pkg.get('version', ''),
                    'publisher': pkg.get('publisher', {}).get('username', ''),
                })
            
            print(f"✅ SCOUT: npm trends: {len(packages)} packages")
            return packages
        except Exception as e:
            print(f"⚠️ SCOUT: npm fetch failed ({e})")
            return []


class CompetitorScout:
    """
    🎯 كشاف المنافسين
    
    يرصد:
    - تحركات المنافسين
    - مميزاتهم الجديدة
    - نقاط ضعفهم
    """
    
    def __init__(self):
        self.name = "Competitor Scout"
        self.competitors = {
            'odoo': {'name': 'Odoo', 'website': 'odoo.com'},
            'sap': {'name': 'SAP', 'website': 'sap.com'},
            'zoho': {'name': 'Zoho', 'website': 'zoho.com'}
        }
        self.tracking_data: Dict = {}
        print(f"🎯 {self.name} initialized")
    
    async def gather_intel(self) -> List[IntelReport]:
        """جمع معلومات المنافسين"""
        reports = []
        
        for comp_id, comp_info in self.competitors.items():
            # مراقبة الموقع
            updates = await self._monitor_website(comp_info['website'])
            if updates:
                reports.append(IntelReport(
                    intel_id=f"comp_{comp_id}_{datetime.now(timezone.utc).timestamp()}",
                    scout_name=self.name,
                    intel_type=IntelType.COMPETITOR,
                    source=comp_info['website'],
                    content=f"{comp_info['name']}: {updates['headline']}",
                    confidence=0.90,
                    urgency=updates.get('urgency', 5),
                    timestamp=datetime.now(timezone.utc),
                    metadata=updates
                ))
            
            # GitHub activity for competitor
            gh_activity = await self._check_github_activity(comp_id)
            if gh_activity:
                reports.append(IntelReport(
                    intel_id=f"comp_gh_{comp_id}_{datetime.now(timezone.utc).timestamp()}",
                    scout_name=self.name,
                    intel_type=IntelType.COMPETITOR,
                    source='github',
                    content=f"{comp_info['name']} GitHub: {gh_activity['repos']} repos, {gh_activity['recent_commits']} recent",
                    confidence=0.85,
                    urgency=4,
                    timestamp=datetime.now(timezone.utc),
                    metadata={**gh_activity, '_source': 'github_api'}
                ))
        
        return reports
    
    async def _monitor_website(self, url: str) -> Optional[Dict]:
        """مراقبة موقع المنافس — HTTP check"""
        try:
            import urllib.request
            
            req = urllib.request.Request(f"https://{url}", headers={
                'User-Agent': 'BI-IDE-Scout/1.0'
            })
            with urllib.request.urlopen(req, timeout=5) as response:
                status = response.status
                content_length = response.headers.get('Content-Length', 'unknown')
            
            return {
                'url': url,
                'headline': f'{url} is online (HTTP {status})',
                'status': 'monitored',
                'http_status': status,
                '_source': 'live_http_check'
            }
        except Exception as e:
            # Site unreachable or error — still valid intel
            return {
                'url': url,
                'headline': f'{url} unreachable: {str(e)[:100]}',
                'status': 'error',
                '_source': 'live_http_check'
            }
    
    async def _check_github_activity(self, competitor: str) -> Optional[Dict]:
        """GitHub activity for competitor org — real HTTP"""
        try:
            import urllib.request
            
            url = f"https://api.github.com/orgs/{competitor}/repos?sort=updated&per_page=5"
            req = urllib.request.Request(url, headers={
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'BI-IDE-Scout/1.0'
            })
            with urllib.request.urlopen(req, timeout=8) as response:
                repos = json.loads(response.read().decode())
            
            if not isinstance(repos, list):
                return None
            
            return {
                'org': competitor,
                'repos': len(repos),
                'recent_commits': sum(1 for r in repos if r.get('pushed_at', '') > '2026-01-01'),
                'top_repos': [{'name': r.get('name'), 'stars': r.get('stargazers_count', 0)} for r in repos[:3]],
            }
        except Exception as e:
            print(f"⚠️ SCOUT: GitHub org check failed for {competitor} ({e})")
            return None


class OpportunityScout:
    """
    💎 كشاف الفرص
    
    يرصد بيانات حقيقية من:
    - HackerNews (top stories — funding, launches, hiring)
    - Product Hunt (trending products — partnership opportunities)
    - GitHub Trending (new tools & frameworks)
    """
    
    def __init__(self):
        self.name = "Opportunity Scout"
        self.opportunity_keywords = [
            'funding', 'launch', 'startup', 'erp', 'ide', 'ai',
            'open source', 'developer tools', 'saas', 'iraq', 'mena'
        ]
        print(f"💎 {self.name} initialized")
    
    async def gather_intel(self) -> List[IntelReport]:
        """جمع الفرص — Real HTTP"""
        reports = []
        
        # 1. HackerNews top stories
        hn_stories = await self._fetch_hackernews()
        for story in hn_stories:
            reports.append(IntelReport(
                intel_id=f"hn_{story['id']}",
                scout_name=self.name,
                intel_type=IntelType.OPPORTUNITY,
                source='hackernews',
                content=f"فرصة: {story['title']}",
                confidence=0.70,
                urgency=story.get('urgency', 6),
                timestamp=datetime.now(timezone.utc),
                metadata={**story, '_source': 'hackernews_api'}
            ))
        
        # 2. GitHub — new developer tools
        dev_tools = await self._fetch_new_dev_tools()
        for tool in dev_tools:
            reports.append(IntelReport(
                intel_id=f"devtool_{tool['name']}_{datetime.now(timezone.utc).timestamp()}",
                scout_name=self.name,
                intel_type=IntelType.OPPORTUNITY,
                source='github',
                content=f"أداة جديدة: {tool['name']} — {tool['description'][:100]}",
                confidence=0.75,
                urgency=5,
                timestamp=datetime.now(timezone.utc),
                metadata={**tool, '_source': 'github_api'}
            ))
        
        return reports
    
    async def _fetch_hackernews(self) -> List[Dict]:
        """HackerNews top stories — real HTTP"""
        try:
            import urllib.request
            
            # Get top story IDs
            url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            req = urllib.request.Request(url, headers={'User-Agent': 'BI-IDE-Scout/1.0'})
            with urllib.request.urlopen(req, timeout=8) as response:
                story_ids = json.loads(response.read().decode())[:10]  # top 10
            
            stories = []
            for sid in story_ids[:5]:  # fetch first 5
                try:
                    item_url = f"https://hacker-news.firebaseio.com/v0/item/{sid}.json"
                    req = urllib.request.Request(item_url, headers={'User-Agent': 'BI-IDE-Scout/1.0'})
                    with urllib.request.urlopen(req, timeout=5) as response:
                        item = json.loads(response.read().decode())
                    
                    title = (item.get('title', '') or '').lower()
                    # Filter for relevant stories
                    relevant = any(kw in title for kw in self.opportunity_keywords)
                    
                    stories.append({
                        'id': sid,
                        'title': item.get('title', ''),
                        'url': item.get('url', ''),
                        'score': item.get('score', 0),
                        'comments': item.get('descendants', 0),
                        'relevant': relevant,
                        'urgency': 8 if relevant else 4,
                    })
                except Exception:
                    continue
            
            print(f"✅ SCOUT: HackerNews: {len(stories)} stories fetched")
            return stories
        except Exception as e:
            print(f"⚠️ SCOUT: HackerNews fetch failed ({e})")
            return []
    
    async def _fetch_new_dev_tools(self) -> List[Dict]:
        """GitHub — recently created developer tools"""
        try:
            import urllib.request
            
            url = "https://api.github.com/search/repositories?q=topic:developer-tools+created:>2026-01-01&sort=stars&order=desc&per_page=3"
            req = urllib.request.Request(url, headers={
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'BI-IDE-Scout/1.0'
            })
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            tools = []
            for item in data.get('items', [])[:3]:
                tools.append({
                    'name': item.get('full_name', ''),
                    'description': item.get('description', 'No description'),
                    'stars': item.get('stargazers_count', 0),
                    'language': item.get('language', 'Unknown'),
                    'url': item.get('html_url', ''),
                })
            
            print(f"✅ SCOUT: New dev tools: {len(tools)} found")
            return tools
        except Exception as e:
            print(f"⚠️ SCOUT: Dev tools fetch failed ({e})")
            return []


class ScoutManager:
    """
    مدير الكشافة
    
    يدير الـ4 كشافة ويوزع المعلومات
    """
    
    def __init__(self):
        self.scouts = [
            TechScout(),
            MarketScout(),
            CompetitorScout(),
            OpportunityScout()
        ]
        self.intel_buffer: List[IntelReport] = []
        self.high_priority_queue: List[IntelReport] = []
        print("🕵️ Scout Manager initialized (4 scouts)")
    
    async def gather_all_intel(self) -> Dict:
        """جمع كل المعلومات"""
        all_reports = []
        
        # تشغيل الكشافة بالتوازي
        tasks = [scout.gather_intel() for scout in self.scouts]
        results = await asyncio.gather(*tasks)
        
        for reports in results:
            all_reports.extend(reports)
        
        # التصنيف حسب الأولوية
        for report in all_reports:
            if report.urgency >= 8:
                self.high_priority_queue.append(report)
        
        self.intel_buffer.extend(all_reports)
        
        return {
            'total_reports': len(all_reports),
            'high_priority': len(self.high_priority_queue),
            'by_type': self._categorize_by_type(all_reports),
            'reports': all_reports
        }
    
    def _categorize_by_type(self, reports: List[IntelReport]) -> Dict:
        """تصنيف حسب النوع"""
        result = {}
        for report in reports:
            t = report.intel_type.value
            result[t] = result.get(t, 0) + 1
        return result
    
    async def continuous_intelligence(self, high_council):
        """جمع استخباراتي مستمر"""
        while True:
            intel = await self.gather_all_intel()
            
            # إرسال العاجل للحكماء
            if intel['high_priority'] > 0:
                urgent = self.high_priority_queue[-intel['high_priority']:]
                await high_council.receive_urgent_intel(urgent)
            
            # إرسال ملخص دوري
            print(f"🔍 Intel gathered: {intel['total_reports']} reports, {intel['high_priority']} urgent")
            
            await asyncio.sleep(1800)  # كل 30 دقيقة
    
    def get_intel_summary(self, hours: int = 24) -> str:
        """ملخص استخباراتي"""
        recent = [r for r in self.intel_buffer 
                  if (datetime.now() - r.timestamp).seconds < hours * 3600]
        
        return f"""
📊 Intel Summary (last {hours}h)
━━━━━━━━━━━━━━━━━━━━━━━
🧪 Tech: {len([r for r in recent if r.intel_type == IntelType.TECH])}
📊 Market: {len([r for r in recent if r.intel_type == IntelType.MARKET])}
🎯 Competitors: {len([r for r in recent if r.intel_type == IntelType.COMPETITOR])}
💎 Opportunities: {len([r for r in recent if r.intel_type == IntelType.OPPORTUNITY])}
⚠️ Threats: {len([r for r in recent if r.intel_type == IntelType.THREAT])}
━━━━━━━━━━━━━━━━━━━━━━━
🚨 High Priority: {len([r for r in recent if r.urgency >= 8])}
"""


# Singleton
scout_manager = ScoutManager()
