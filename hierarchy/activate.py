"""
تنشيط النظام الهرمي - System Activation
يحمل الـ 6500 مثال كـ "ذاكرة أولية" مو كـ تدريب
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
from typing import Dict, List, Any
from datetime import datetime

# Import Hierarchy (relative)
from hierarchy import ai_hierarchy, AIHierarchy


class KnowledgeLoader:
    """
    محمل المعرفة الأولية
    يحول ملفات JSON لـ "خبرات" الحكماء
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.knowledge_base = {
            "erp_basic": [],
            "erp_advanced": [],
            "company_history": [],
            "conversations": [],
            "reinforcement": []
        }
        self.total_examples = 0
    
    def load_all(self) -> Dict[str, List]:
        """تحميل كل المعرفة"""
        files = {
            "erp_basic": "erp_basic.json",
            "erp_advanced": "erp_advanced.json", 
            "company_history": "company_history.json",
            "conversations": "conversations.json",
            "reinforcement": "reinforcement.json"
        }
        
        print("📚 تحميل الذاكرة الأولية...")
        
        for key, filename in files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.knowledge_base[key] = data
                        self.total_examples += len(data)
                        print(f"  ✅ {filename}: {len(data)} مثال")
                    elif isinstance(data, dict) and 'examples' in data:
                        self.knowledge_base[key] = data['examples']
                        self.total_examples += len(data['examples'])
                        print(f"  ✅ {filename}: {len(data['examples'])} مثال")
            else:
                print(f"  ⚠️  {filename}: غير موجود")
        
        print(f"\n📊 إجمالي الأمثلة: {self.total_examples}")
        return self.knowledge_base
    
    def seed_high_council(self, hierarchy: AIHierarchy):
        """
 زرع المعرفة في مجلس الحكماء
        هذه "الخبرة" مو "تدريب"
        """
        print("\n🧠 زرع المعرفة في High Council...")
        
        # تجميع كل المعرفة
        all_knowledge = []
        for category, examples in self.knowledge_base.items():
            for ex in examples:
                all_knowledge.append({
                    "category": category,
                    "content": ex,
                    "timestamp": datetime.now().isoformat(),
                    "source": "initial_memory"
                })
        
        # توزيع على الحكماء الـ 16
        wise_men_count = 16
        knowledge_per_wise = len(all_knowledge) // wise_men_count
        
        for i in range(wise_men_count):
            start_idx = i * knowledge_per_wise
            end_idx = start_idx + knowledge_per_wise if i < wise_men_count - 1 else len(all_knowledge)
            
            wise_man_knowledge = all_knowledge[start_idx:end_idx]
            
            # إضافة للحكيم (في الذاكرة، مو في الـ weights)
            if hasattr(hierarchy.council, 'wise_men') and i < len(hierarchy.council.wise_men):
                wise_man = hierarchy.council.wise_men[i]
                if not hasattr(wise_man, 'memory'):
                    wise_man.memory = []
                wise_man.memory.extend(wise_man_knowledge)
                
        print(f"  ✅ زرعت {len(all_knowledge)} خبرة على {wise_men_count} حكيم")
    
    def seed_domain_experts(self, hierarchy: AIHierarchy):
        """
        زرع المعرفة لخبراء المجالات
        """
        print("\n👥 زرع المعرفة لـ Domain Experts...")
        
        # توزيع حسب التخصص
        domain_mapping = {
            "erp_basic": ["business", "accounting", "erp"],
            "erp_advanced": ["business", "accounting", "analytics"],
            "company_history": ["business", "strategy"],
            "conversations": ["communication", "psychology"],
            "reinforcement": ["learning", "optimization"]
        }
        
        for category, examples in self.knowledge_base.items():
            domains = domain_mapping.get(category, ["general"])
            
            for domain in domains:
                if domain in hierarchy.experts.experts:
                    expert = hierarchy.experts.experts[domain]
                    if not hasattr(expert, 'knowledge_base'):
                        expert.knowledge_base = []
                    
                    # إضافة 20% من الأمثلة لكل خبير متخصص
                    sample_size = max(1, len(examples) // 5)
                    expert.knowledge_base.extend(examples[:sample_size])
                    
        print("  ✅ توزيعت المعرفة على الخبراء")


async def activate_system():
    """
    تنشيط النظام كامل
    """
    print("=" * 60)
    print("🔥 تنشيط النظام الهرمي - SYSTEM ACTIVATION")
    print("=" * 60)
    
    # 1. تحميل المعرفة
    loader = KnowledgeLoader("data")
    knowledge = loader.load_all()
    
    if loader.total_examples == 0:
        print("❌ لا توجد معرفة للتحميل!")
        return
    
    # 2. تهيئة الهرم
    print("\n🏛️ تهيئة AI Hierarchy...")
    hierarchy = ai_hierarchy
    await hierarchy.initialize()
    
    # 3. زرع المعرفة
    loader.seed_high_council(hierarchy)
    loader.seed_domain_experts(hierarchy)
    
    # 4. تفعيل التعلم الذاتي
    print("\n🔄 تفعيل التعلم الذاتي...")
    hierarchy.active_mode = "learning"
    
    # 5. حالة النظام
    print("\n" + "=" * 60)
    print("✅ النظام الهرمي نشط!")
    print("=" * 60)
    print(f"🧠 High Council: 16 حكيم (كل واحد عنده ~{loader.total_examples // 16} خبرة)")
    print(f"👥 Domain Experts: {len(hierarchy.experts.experts)} خبير")
    print(f"📊 Meta Team: 16 مدير")
    print(f"🎯 الوضع: {hierarchy.active_mode}")
    print("\n💡 النظام يتعلم أوتماتيكياً من التفاعلات")
    print("=" * 60)
    
    return hierarchy


if __name__ == "__main__":
    # تشغيل
    hierarchy = asyncio.run(activate_system())
    
    # الاستمرار في العمل
    print("\n⏳ النظام شغال... اضغط Ctrl+C للإيقاف")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 إيقاف النظام")
