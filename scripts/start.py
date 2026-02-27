"""
تشغيل النظام الهرمي مع التعلم الذاتي
AI Hierarchy with Observational Learning
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
from hierarchy import ai_hierarchy
from hierarchy.autonomous_learning import get_learning_system
from hierarchy.connect_services import connect_all_services

# Services
from ide.ide_service import IDEService
from erp.erp_service import ERPService


async def main():
    print("=" * 60)
    print("🔥 BI IDE - Autonomous Learning System")
    print("=" * 60)
    print("\n💡 النظام رح يتعلم من شغلك أوتماتيكياً")
    print("   • الكود اللي تكتبه")
    print("   • المعاملات في ERP")  
    print("   • القرارات اللي تتخذها")
    print("   • الأخطاء والنجاحات")
    print()
    
    # 1. تهيئة الـ Hierarchy
    print("🧠 Initializing AI Hierarchy (15 layers)...")
    hierarchy = ai_hierarchy
    await hierarchy.initialize()
    
    # 2. إنشاء نظام التعلم
    print("📚 Initializing Observational Learning...")
    learning = get_learning_system(hierarchy)
    
    # 3. إنشاء الخدمات
    print("💻 Starting IDE Service...")
    ide_service = IDEService(hierarchy)
    
    print("🏢 Starting ERP Service...")
    erp_service = ERPService(hierarchy)
    
    # 4. توصيل التعلم
    connect_all_services(hierarchy, ide_service, erp_service)
    
    # 5. حالة النظام
    print("=" * 60)
    print("✅ System Ready!")
    print("=" * 60)
    print("\n📊 Status:")
    print(f"   • 15 AI Layers: Active")
    print(f"   • Observational Learning: Enabled")
    print(f"   • IDE: Ready (watching code)")
    print(f"   • ERP: Ready (watching transactions)")
    print()
    print("🚀 The system is learning from you...")
    print("   Write code → AI learns patterns")
    print("   Create invoice → AI learns business")
    print("   Make decisions → AI learns strategy")
    print()
    print("📡 Access API at: http://localhost:8000")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 60)
    
    # 6. الاستمرار
    try:
        while True:
            await asyncio.sleep(10)
            
            # عرض إحصائيات كل فترة
            stats = learning.get_learning_stats()
            if stats['buffer_size'] > 0:
                print(f"💡 Learned {stats['buffer_size']} new experiences")
                
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping system...")
        
        # حفظ آخر التعلم
        learning._save_learning()
        
        print("✅ System stopped. Learning saved.")


if __name__ == "__main__":
    asyncio.run(main())
