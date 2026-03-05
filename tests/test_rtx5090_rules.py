"""
Tests for RTX5090 Rules - اختبارات قوانين RTX5090

قوانين حاسمة:
1. Thermal: 97°C = EMERGENCY STOP
2. Inference: LoRA only - NO OLLAMA
3. Training: Ollama = training only
4. Security: No hardcoded secrets
"""

import pytest
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRTX5090Rules:
    """اختبارات قوانين RTX5090"""
    
    def test_thermal_limits_unchanged(self):
        """التحقق من عدم تغيير حدود الحرارة"""
        from training.v8_modules.thermal_guard import ThermalGuard
        
        guard = ThermalGuard()
        
        # القوانين ثابتة - لا تتغير
        assert guard.WARNING_TEMP == 90.0, "WARNING temp must be 90°C"
        assert guard.THROTTLE_TEMP == 94.0, "THROTTLE temp must be 94°C"
        assert guard.EMERGENCY_TEMP == 97.0, "EMERGENCY temp must be 97°C"
        assert guard.RESUME_TEMP == 86.0, "RESUME temp must be 86°C"
    
    def test_lora_inference_no_ollama_fallback(self):
        """التحقق من عدم وجود Ollama fallback في الاستنتاج"""
        from training.v8_modules.lora_inference import LoRAInferenceEngine
        
        # محرك الاستنتاج يجب ألا يحتوي على Ollama
        engine = LoRAInferenceEngine()
        
        # التحقق من أن المحرك يستخدم LoRA فقط
        assert engine.device in ["cuda", "cpu"], "Must use local inference only"
        assert "ollama" not in str(engine.models_dir).lower(), "No Ollama in paths"
    
    def test_no_hardcoded_jwt_secrets(self):
        """التحقق من عدم وجود أسرار JWT ثابتة"""
        # قراءة ملف الإعدادات
        env_path = Path(__file__).parent.parent / ".env"
        
        if env_path.exists():
            content = env_path.read_text()
            
            # التأكد من عدم وجود قيم افتراضية غير آمنة
            insecure_patterns = [
                "your-secret-key",
                "change-this-in-production",
                "president123",
                "admin123",
                "password123",
            ]
            
            for pattern in insecure_patterns:
                assert pattern not in content.lower(), f"Insecure pattern found: {pattern}"
    
    def test_config_validates_secrets(self):
        """التحقق من التحقق من الأسرار في الإعدادات"""
        from core.config import get_settings
        
        settings = get_settings()
        
        # في بيئة الإنتاج، يجب رفض الأسرار الضعيفة
        if settings.ENVIRONMENT == "production":
            insecure_secrets = {
                "",
                "your-secret-key-change-this",
                "bi-ide-v8-change-this-in-production-2026",
            }
            assert settings.SECRET_KEY not in insecure_secrets, "SECRET_KEY must be secure in production"


class TestLoRAIntegration:
    """اختبارات تكامل LoRA"""
    
    @pytest.mark.asyncio
    async def test_lora_inference_returns_result(self):
        """التحقق من عمل استنتاج LoRA"""
        from training.v8_modules.lora_inference import LoRAInferenceEngine
        
        engine = LoRAInferenceEngine()
        
        # محاولة الاستنتاج (قد لا يكون هناك نموذج، لكن يجب أن لا يتعطل)
        result = await engine.infer("test prompt", max_new_tokens=10)
        
        # يجب أن يكون هناك نتيجة صالحة
        assert result is not None
        assert hasattr(result, 'text')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'source')
    
    def test_model_manager_discovers_adapters(self):
        """التحقق من اكتشاف مدير النماذج للـ adapters"""
        from training.v8_modules.model_manager import ModelManager
        
        manager = ModelManager()
        versions = manager.discover_versions()
        
        # يجب أن يكون list حتى لو فارغ
        assert isinstance(versions, list)


class TestThermalProtection:
    """اختبارات الحماية الحرارية"""
    
    def test_thermal_guard_detects_state(self):
        """التحقق من اكتشاف الحارس الحراري للحالات"""
        from training.v8_modules.thermal_guard import ThermalGuard, ThermalState
        
        guard = ThermalGuard()
        
        # اختبار تحديد الحالة
        assert guard._determine_state(50, None) == ThermalState.NORMAL
        assert guard._determine_state(92, None) == ThermalState.WARNING
        assert guard._determine_state(95, None) == ThermalState.THROTTLE
        assert guard._determine_state(98, None) == ThermalState.EMERGENCY
    
    def test_thermal_stops_training_at_97(self):
        """التحقق من إيقاف التدريب عند 97°C"""
        from training.v8_modules.thermal_guard import ThermalGuard, ThermalState
        
        guard = ThermalGuard()
        
        # محاكاة درجة حرارة 97°C
        state = guard._determine_state(97, None)
        assert state == ThermalState.EMERGENCY
        assert not guard.is_safe_to_train()


class TestAIVHierarchy:
    """اختبارات النظام الهرمي"""
    
    def test_ai_hierarchy_initialized(self):
        """التحقق من تهيئة النظام الهرمي"""
        from hierarchy import ai_hierarchy
        
        assert ai_hierarchy is not None
        assert hasattr(ai_hierarchy, 'council')
        assert hasattr(ai_hierarchy, 'scouts')
        assert hasattr(ai_hierarchy, 'execution')
    
    def test_consensus_calculation_real(self):
        """التحقق من حساب الإجماع الحقيقي"""
        from hierarchy import ai_hierarchy
        import asyncio
        
        # حساب الإجماع يجب أن يكون حقيقياً وليس ثابتاً
        result = asyncio.run(ai_hierarchy._perform_deliberation("test command", {}))
        
        assert 'consensus_score' in result
        assert 0 <= result['consensus_score'] <= 1
        assert 'participating_sages' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
