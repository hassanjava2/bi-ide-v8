#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                Bi IDE - التدريب الشامل الكامل                               ║
║                    Full Training Pipeline                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  هذا السكريبت يشغّل جميع مراحل التدريب:                                     ║
║    1. تجهيز وتحميل البيانات                                                  ║
║    2. التدريب على إصلاح الأخطاء                                              ║
║    3. التدريب على توليد الكود                                                ║
║    4. تجميع ودمج البيانات                                                    ║
║    5. تصدير للاستخدام في التطبيق                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

التشغيل:
    python run_full_training.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# إعداد الترميز
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# إضافة مسار المشروع
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'training'))


def print_header(text: str, char: str = "="):
    """طباعة عنوان منسق"""
    width = 70
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)
    print()


def print_step(number: int, text: str):
    """طباعة خطوة"""
    print(f"\n{'-' * 50}")
    print(f"  Step {number}: {text}")
    print(f"{'-' * 50}\n")


def run_training():
    """تشغيل خط التدريب الكامل"""
    
    # طباعة الترحيب
    print("\n")
    print("=" * 70)
    print("          Bi IDE - Full AI Training System")
    print("=" * 70)
    print("  Code Understanding | Error Fixing | Code Generation")
    print("=" * 70)
    print()
    
    start_time = time.time()
    results = {
        'started_at': datetime.now().isoformat(),
        'steps': []
    }
    
    OUTPUT_DIR = BASE_DIR / "training" / "output"
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    
    # إنشاء المجلدات
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # ===========================================================
        # Step 1: Basic Training
        # ===========================================================
        print_step(1, "التدريب الأساسي - تحميل وتجهيز البيانات")
        
        from train_ai import AITrainer
        
        trainer = AITrainer()
        trainer.prepare_data()
        trainer.save_training_data()
        trainer._save_for_nodejs()
        
        results['steps'].append({
            'name': 'basic_training',
            'status': 'success',
            'samples': trainer.stats.total_samples
        })
        
        print("[OK] Basic training completed")
        
        # ===========================================================
        # Step 2: Advanced Error Fixing Training
        # ===========================================================
        print_step(2, "التدريب المتقدم على إصلاح الأخطاء")
        
        from advanced_training import AdvancedTrainingGenerator
        
        error_trainer = AdvancedTrainingGenerator()
        error_data = error_trainer.save_training_data(OUTPUT_DIR)
        
        error_count = sum(len(d) for d in error_data.values())
        results['steps'].append({
            'name': 'error_fixing_training',
            'status': 'success',
            'samples': error_count
        })
        
        print("[OK] Error fixing training completed")
        
        # ===========================================================
        # Step 3: Code Generation Training
        # ===========================================================
        print_step(3, "التدريب على توليد وإكمال الكود")
        
        from code_generation_training import CodeGenerationTrainer
        
        code_trainer = CodeGenerationTrainer()
        code_data = code_trainer.save_training_data(OUTPUT_DIR)
        
        code_count = sum(len(d) for d in code_data.values())
        results['steps'].append({
            'name': 'code_generation_training',
            'status': 'success',
            'samples': code_count
        })
        
        print("[OK] Code generation training completed")
        
        # ===========================================================
        # Step 4: Consolidate All Data
        # ===========================================================
        print_step(4, "تجميع ودمج جميع بيانات التدريب")
        
        all_training_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0',
                'description': 'Bi IDE AI Training Data'
            },
            'categories': {
                'error_detection': [],
                'error_fixing': [],
                'code_generation': [],
                'code_completion': [],
                'qa_pairs': [],
                'best_practices': []
            }
        }
        
        # جمع من التدريب الأساسي
        for category, items in trainer.training_data.items():
            if category in all_training_data['categories']:
                all_training_data['categories'][category].extend(items)
        
        # جمع من تدريب الأخطاء
        for data_type, items in error_data.items():
            if data_type in all_training_data['categories']:
                all_training_data['categories'][data_type].extend(items)
        
        # جمع من توليد الكود
        for data_type, items in code_data.items():
            if data_type == 'code_generation':
                all_training_data['categories']['code_generation'].extend(items)
            elif data_type == 'code_completion':
                all_training_data['categories']['code_completion'].extend(items)
        
        # حساب الإحصائيات
        total_samples = sum(
            len(items) for items in all_training_data['categories'].values()
        )
        all_training_data['metadata']['total_samples'] = total_samples
        all_training_data['metadata']['category_counts'] = {
            cat: len(items) for cat, items in all_training_data['categories'].items()
        }
        
        # حفظ الملف المجمع
        combined_path = OUTPUT_DIR / "all_training_data.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_training_data, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] Saved {total_samples} combined training samples")
        
        results['steps'].append({
            'name': 'data_consolidation',
            'status': 'success',
            'total_samples': total_samples
        })
        
        # ===========================================================
        # Step 5: Export for Application
        # ===========================================================
        print_step(5, "تصدير البيانات للاستخدام في التطبيق")
        
        # تحديث قاعدة المعرفة
        knowledge_base_path = MODELS_DIR / "knowledge-base.json"
        
        knowledge_base = {
            'version': '2.0.0',
            'updated_at': datetime.now().isoformat(),
            'training_info': {
                'total_samples': total_samples,
                'categories': all_training_data['metadata']['category_counts']
            },
            'error_patterns': all_training_data['categories'].get('error_fixing', [])[:200],
            'code_templates': all_training_data['categories'].get('code_generation', [])[:100],
            'completions': all_training_data['categories'].get('code_completion', [])[:50],
            'qa_knowledge': all_training_data['categories'].get('qa_pairs', [])[:200]
        }
        
        with open(knowledge_base_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] Knowledge base updated: {knowledge_base_path}")
        
        # تحديث المعرفة المتعلمة
        learned_path = DATA_DIR / "learned-knowledge.json"
        
        learned_knowledge = {
            'version': '2.0.0',
            'updated_at': datetime.now().isoformat(),
            'trained_on': results['started_at'],
            'statistics': {
                'total_samples': total_samples,
                'error_patterns': len(all_training_data['categories'].get('error_fixing', [])),
                'code_templates': len(all_training_data['categories'].get('code_generation', [])),
                'qa_pairs': len(all_training_data['categories'].get('qa_pairs', []))
            },
            'languages_covered': ['javascript', 'python', 'react', 'sql', 'html', 'css'],
            'capabilities': [
                'error_detection',
                'error_fixing',
                'code_generation',
                'code_completion',
                'code_explanation',
                'best_practices'
            ]
        }
        
        with open(learned_path, 'w', encoding='utf-8') as f:
            json.dump(learned_knowledge, f, ensure_ascii=False, indent=2)
        
        print(f"📝 تم تحديث المعرفة المتعلمة: {learned_path}")
        
        results['steps'].append({
            'name': 'export_to_app',
            'status': 'success'
        })
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        results['error'] = str(e)
        raise
    
    finally:
        end_time = time.time()
        duration = end_time - start_time
        results['completed_at'] = datetime.now().isoformat()
        results['duration_seconds'] = round(duration, 2)
        
        # حفظ تقرير التدريب
        report_path = OUTPUT_DIR / "training_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print final summary
    print("\n")
    print("=" * 70)
    print("  [SUCCESS] Training Completed!")
    print("=" * 70)
    print(f"  Total Time: {duration:.1f} seconds")
    print(f"  Total Samples: {total_samples}")
    print()
    print("  Files Created:")
    print("    - training/output/all_training_data.json")
    print("    - models/knowledge-base.json")
    print("    - data/learned-knowledge.json")
    print()
    print("  To use in Bi IDE: npm start")
    print("=" * 70)
    print()
    
    return results


def main():
    """نقطة البداية"""
    try:
        run_training()
    except KeyboardInterrupt:
        print("\n\n[WARN] Training stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
