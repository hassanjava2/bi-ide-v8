"""
Example: استخدام RTX 4090 مع مجلس الحكماء
RTX 4090 + Smart Council Integration Example
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.rtx4090_client import RTX4090Client, RTX4090Pool
from ai.training.rtx4090_trainer import RTX4090Trainer
from council_ai import SmartCouncil, WiseManAI


async def test_rtx4090_health():
    """Test RTX 4090 server health"""
    print("=" * 60)
    print("🧪 Testing RTX 4090 Connection")
    print("=" * 60)
    
    async with RTX4090Client() as client:
        try:
            health = await client.health_check()
            print(f"✅ Server: {health.get('name', 'Unknown')}")
            print(f"   Version: {health.get('version', 'N/A')}")
            print(f"   GPU: {health.get('gpu', 'N/A')}")
            print(f"   Mode: {health.get('mode', 'N/A')}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False


async def test_rtx4090_inference():
    """Test RTX 4090 text generation"""
    print("\n" + "=" * 60)
    print("📝 Testing Text Generation")
    print("=" * 60)
    
    async with RTX4090Client() as client:
        try:
            response = await client.generate_text(
                prompt="ما هي نصيحتك للقائد الناجح؟",
                max_tokens=256,
                temperature=0.8,
                top_p=0.9
            )
            print(f"Response:\n{response[:300]}...")
            return True
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return False


async def test_rtx4090_with_wise_man():
    """Test RTX 4090 with a Wise Man"""
    print("\n" + "=" * 60)
    print("🎭 Testing RTX 4090 + Wise Man")
    print("=" * 60)
    
    # Create a wise man
    wise_man = WiseManAI(
        name="حكيم القرار",
        role="رئيس المجلس",
        specialty="الاستراتيجية"
    )
    
    # Test with RTX 4090
    result = await wise_man.think_with_rtx4090(
        message="كيف أتخذ قراراً صائباً في موقف صعب؟",
        max_tokens=512,
        temperature=0.8
    )
    
    print(f"Wise Man: {wise_man.name}")
    print(f"Source: {result.get('source')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"RTX Available: {result.get('rtx_available')}")
    print(f"\nResponse:\n{result.get('response', 'No response')[:400]}...")
    
    return result.get('rtx_available', False)


async def test_rtx4090_pool():
    """Test RTX 4090 pool with multiple servers"""
    print("\n" + "=" * 60)
    print("🌐 Testing RTX 4090 Pool")
    print("=" * 60)
    
    # Create pool (defaults to single server from env)
    pool = RTX4090Pool()
    
    # Health check all
    health_results = await pool.health_check_all()
    
    print(f"Total servers: {len(pool.hosts)}")
    print(f"Healthy servers: {pool.get_healthy_count()}")
    
    for host, status in health_results.items():
        health_icon = "✅" if status.get("healthy") else "❌"
        print(f"   {health_icon} {host}: {'Healthy' if status.get('healthy') else 'Unhealthy'}")
    
    return pool.get_healthy_count() > 0


async def test_training_pipeline():
    """Test training pipeline"""
    print("\n" + "=" * 60)
    print("📚 Testing Training Pipeline")
    print("=" * 60)
    
    trainer = RTX4090Trainer()
    
    # List checkpoints
    checkpoints = await trainer.client.list_checkpoints()
    print(f"Available checkpoints: {len(checkpoints)}")
    
    for ckpt in checkpoints[:5]:  # Show first 5
        print(f"   📦 {ckpt.get('layer')}/{ckpt.get('file')} ({ckpt.get('size_mb')} MB)")
    
    return len(checkpoints) > 0


async def test_smart_council_with_rtx4090():
    """Test Smart Council using RTX 4090"""
    print("\n" + "=" * 60)
    print("🏛️ Testing Smart Council with RTX 4090")
    print("=" * 60)
    
    council = SmartCouncil()
    
    # Ask a question and use RTX 4090 for response
    wise_man_name = "حكيم القرار"
    message = "ما هو سر القرارات الناجحة؟"
    
    if wise_man_name in council.wise_men:
        wise_man = council.wise_men[wise_man_name]
        result = await wise_man.think_with_rtx4090(message)
        
        print(f"Question: {message}")
        print(f"Wise Man: {wise_man_name}")
        print(f"Source: {result.get('source')}")
        print(f"\nResponse:\n{result.get('response', 'No response')[:500]}...")
        
        return result.get('rtx_available', False)
    else:
        print(f"❌ Wise man not found: {wise_man_name}")
        return False


async def main():
    """Run all tests"""
    print("\n" + "🌟" * 30)
    print("RTX 4090 + Smart Council Integration Tests")
    print("🌟" * 30 + "\n")
    
    results = {
        "Health Check": await test_rtx4090_health(),
        "Text Generation": await test_rtx4090_inference(),
        "RTX 4090 Pool": await test_rtx4090_pool(),
        "Training Pipeline": await test_training_pipeline(),
        "Wise Man + RTX 4090": await test_rtx4090_with_wise_man(),
        "Smart Council + RTX 4090": await test_smart_council_with_rtx4090(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        icon = "✅" if passed else "❌"
        status = "PASSED" if passed else "FAILED"
        print(f"{icon} {test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())
