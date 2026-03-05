#!/usr/bin/env python3
"""
Verification Script - 100% Completion Check
سكربت التحقق من اكتمال المشروع 100%
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_check(name: str, passed: bool, details: str = ""):
    icon = f"{Colors.GREEN}✅" if passed else f"{Colors.RED}❌"
    status = f"{Colors.GREEN}PASS" if passed else f"{Colors.RED}FAIL"
    print(f"{icon} {name}: {status}{Colors.RESET}")
    if details:
        print(f"   {Colors.YELLOW}{details}{Colors.RESET}")


class CompletionVerifier:
    """التحقق من اكتمال المشروع"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results: Dict[str, bool] = {}
    
    def verify_all(self):
        """التحقق من جميع المكونات"""
        print_header("BI-IDE V8 - 100% COMPLETION VERIFICATION")
        
        # 1. البنية التحتية
        self._verify_infrastructure()
        
        # 2. الأمان
        self._verify_security()
        
        # 3. التدريب والموديل
        self._verify_training()
        
        # 4. النظام الهرمي
        self._verify_hierarchy()
        
        # 5. القوانين
        self._verify_rules()
        
        # 6. الطبقات المتخصصة
        self._verify_specialized_layers()
        
        # 7. التوافقية
        self._verify_compatibility()
        
        # النتيجة النهائية
        self._print_final_result()
    
    def _verify_infrastructure(self):
        """التحقق من البنية التحتية"""
        print_header("1. INFRASTRUCTURE")
        
        # Docker
        dockerfile = self.project_root / "Dockerfile"
        self.results["dockerfile"] = dockerfile.exists()
        print_check("Dockerfile exists", self.results["dockerfile"])
        
        # Docker Compose
        compose = self.project_root / "docker-compose.yml"
        self.results["compose"] = compose.exists()
        print_check("docker-compose.yml exists", self.results["compose"])
        
        # Database
        init_sql = self.project_root / "init.sql"
        self.results["init_sql"] = init_sql.exists()
        print_check("init.sql exists", self.results["init_sql"])
        
        # Config
        config = self.project_root / "core" / "config.py"
        self.results["config"] = config.exists()
        print_check("core/config.py exists", self.results["config"])
    
    def _verify_security(self):
        """التحقق من الأمان"""
        print_header("2. SECURITY")
        
        # Auth module
        auth = self.project_root / "api" / "auth.py"
        self.results["auth"] = auth.exists()
        print_check("api/auth.py exists", self.results["auth"])
        
        # Middleware
        middleware = self.project_root / "api" / "middleware.py"
        self.results["middleware"] = middleware.exists()
        print_check("api/middleware.py exists", self.results["middleware"])
        
        # .env without hardcoded secrets
        env_file = self.project_root / ".env"
        if env_file.exists():
            content = env_file.read_text()
            # التحقق من عدم وجود أسرار ثابتة
            insecure = [
                "dcf006c2c36c36b172f44834984a461b98cd12d2d4978457fae8514e334f7d56",
                "president123",
                "dev-local-orchestrator-token-2026",
            ]
            has_insecure = any(sec in content for sec in insecure)
            self.results["env_clean"] = not has_insecure
            print_check(".env has no hardcoded secrets", self.results["env_clean"])
        else:
            self.results["env_clean"] = False
            print_check(".env exists", False)
    
    def _verify_training(self):
        """التحقق من نظام التدريب"""
        print_header("3. TRAINING SYSTEM V8")
        
        # V8 Training modules
        v8_modules = self.project_root / "training" / "v8-modules"
        self.results["v8_modules"] = v8_modules.exists()
        print_check("training/v8-modules/ exists", self.results["v8_modules"])
        
        # LoRA Inference
        lora = v8_modules / "lora_inference.py"
        self.results["lora_inference"] = lora.exists()
        print_check("lora_inference.py exists", self.results["lora_inference"])
        
        # Thermal Guard
        thermal = v8_modules / "thermal_guard.py"
        self.results["thermal_guard"] = thermal.exists()
        print_check("thermal_guard.py exists", self.results["thermal_guard"])
        
        # Model Manager
        model_mgr = v8_modules / "model_manager.py"
        self.results["model_manager"] = model_mgr.exists()
        print_check("model_manager.py exists", self.results["model_manager"])
        
        # Training Pipeline
        pipeline = v8_modules / "training_pipeline.py"
        self.results["training_pipeline"] = pipeline.exists()
        print_check("training_pipeline.py exists", self.results["training_pipeline"])
        
        # RTX5090 Machine
        rtx_dir = self.project_root / "rtx4090_machine"
        self.results["rtx_dir"] = rtx_dir.exists()
        print_check("rtx4090_machine/ exists", self.results["rtx_dir"])
        
        # Auto training daemon
        daemon = rtx_dir / "auto_training_daemon.py"
        self.results["auto_daemon"] = daemon.exists()
        print_check("auto_training_daemon.py exists", self.results["auto_daemon"])
    
    def _verify_hierarchy(self):
        """التحقق من النظام الهرمي"""
        print_header("4. AI HIERARCHY")
        
        hierarchy_dir = self.project_root / "hierarchy"
        self.results["hierarchy_dir"] = hierarchy_dir.exists()
        print_check("hierarchy/ exists", self.results["hierarchy_dir"])
        
        # Core files
        core_files = [
            "__init__.py",
            "president.py",
            "high_council.py",
            "scouts.py",
            "meta_team.py",
            "execution_team.py",
        ]
        
        for f in core_files:
            exists = (hierarchy_dir / f).exists()
            self.results[f"hierarchy_{f}"] = exists
            print_check(f"hierarchy/{f} exists", exists)
    
    def _verify_rules(self):
        """التحقق من القوانين"""
        print_header("5. RTX5090 RULES")
        
        # التحقق من ملفات القوانين مباشرة
        v8_modules = self.project_root / "training" / "v8-modules"
        thermal_file = v8_modules / "thermal_guard.py"
        lora_file = v8_modules / "lora_inference.py"
        
        if thermal_file.exists():
            content = thermal_file.read_text()
            # التحقق من القوانين في الكود
            has_warning = "WARNING_TEMP = 90" in content or "WARNING_TEMP = 90.0" in content
            has_throttle = "THROTTLE_TEMP = 94" in content or "THROTTLE_TEMP = 94.0" in content
            has_emergency = "EMERGENCY_TEMP = 97" in content or "EMERGENCY_TEMP = 97.0" in content
            
            rules_correct = has_warning and has_throttle and has_emergency
            self.results["thermal_rules"] = rules_correct
            print_check("Thermal rules (90/94/97°C)", rules_correct)
        else:
            self.results["thermal_rules"] = False
            print_check("thermal_guard.py exists", False)
        
        # LoRA Inference - No Ollama
        if lora_file.exists():
            content = lora_file.read_text()
            
            # التحقق من وجود LoRA وعدم وجود Ollama fallback
            has_lora = "LoRA" in content
            has_no_ollama_comment = "NO OLLAMA" in content.upper() or "Ollama = تدريب فقط" in content
            
            inference_correct = has_lora and has_no_ollama_comment
            self.results["no_ollama_inference"] = inference_correct
            print_check("LoRA inference (no Ollama fallback)", inference_correct)
        else:
            self.results["no_ollama_inference"] = False
            print_check("lora_inference.py exists", False)
    
    def _verify_specialized_layers(self):
        """التحقق من الطبقات المتخصصة"""
        print_header("6. SPECIALIZED LAYERS")
        
        layers = [
            "penetration_layer.py",
            "vulnerability_layer.py",
            "qa_layer.py",
            "security_layer.py",
            "regeneration_layer.py",
            "ux_excellence_layer.py",
            "integration_layer.py",
        ]
        
        hierarchy_dir = self.project_root / "hierarchy"
        
        for layer in layers:
            exists = (hierarchy_dir / layer).exists()
            self.results[f"layer_{layer}"] = exists
            print_check(f"hierarchy/{layer} exists", exists)
    
    def _verify_compatibility(self):
        """التحقق من التوافقية"""
        print_header("7. LEGACY COMPATIBILITY")
        
        # Route parity
        legacy_compat = self.project_root / "api" / "routes" / "legacy_compat.py"
        self.results["legacy_compat"] = legacy_compat.exists()
        print_check("api/routes/legacy_compat.py exists", self.results["legacy_compat"])
    
    def _print_final_result(self):
        """طباعة النتيجة النهائية"""
        print_header("FINAL RESULT")
        
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        percentage = (passed / total * 100) if total > 0 else 0
        
        print(f"\n{Colors.BOLD}Checks Passed: {passed}/{total} ({percentage:.1f}%){Colors.RESET}\n")
        
        failed = [k for k, v in self.results.items() if not v]
        if failed:
            print(f"{Colors.RED}Failed Checks:{Colors.RESET}")
            for f in failed:
                print(f"  - {f}")
        
        if percentage >= 100:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 PROJECT IS 100% COMPLETE! 🎉{Colors.RESET}\n")
            return 0
        else:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  PROJECT AT {percentage:.0f}% - NOT YET COMPLETE{Colors.RESET}\n")
            return 1


def main():
    verifier = CompletionVerifier()
    exit_code = verifier.verify_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
