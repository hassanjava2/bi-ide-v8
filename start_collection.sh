#!/bin/bash
# سكربت جمع بيانات أوتوماتيكي من Claude

cd /home/bi/bi-ide-v8

# التحقق من المفتاح
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ANTHROPIC_API_KEY not set!"
    echo "Get key from: https://console.anthropic.com/"
    exit 1
fi

echo "🚀 Starting automatic data collection..."
echo "🎯 Target: 10,000 samples/day"
echo "⏱️  Duration: Running 24/7 until interrupted"
echo ""

python3 << 'PYTHON_EOF'
import asyncio
import os
from ai.training.knowledge_distillation_pipeline import distillation_pipeline
from ai.training.data_flywheel import data_flywheel

async def collect_continuously():
    """جمع مستمر 24/7"""
    print("📚 Starting Knowledge Distillation from Claude...")
    
    # جمع 1000 عينة كل يوم
    while True:
        try:
            result = await distillation_pipeline.run_collection_session(
                daily_target=10000  # 10,000 سؤال/يوم
            )
            
            print(f"✅ Collected today: {result['collected_today']}")
            print(f"📊 Total: {result['total_collected']}")
            
            # تصدير للتدريب
            await distillation_pipeline.export_for_training(
                "learning_data/training_data.jsonl"
            )
            
            # انتظار قبل الدفعة التالية
            print("⏳ Sleeping for 1 hour...")
            await asyncio.sleep(3600)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await asyncio.sleep(60)

# تشغيل
asyncio.run(collect_continuously())
PYTHON_EOF
