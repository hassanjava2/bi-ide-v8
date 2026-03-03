"""
Knowledge Distillation Pipeline - خط تقطير المعرفة
استخدام النماذج الكبيرة الموجودة (GPT-4/Claude) كمعلّمين لنموذجنا المحلي

الآن (قبل انقطاع النت):
1. استخدم GPT-4/Claude كمعلم
2. اسأله 100,000+ سؤال بكل المجالات (فيزياء، كيمياء، طب، هندسة...)
3. خزّن الأجوبة كبيانات تدريب عالية الجودة
4. درّب نموذجنا المحلي عليها (LoRA/QLoRA)

النتيجة:
- 80% من جودة GPT-4 بنموذج 7B-13B محلي
- بيانات تدريب مجانية عالية الجودة
- أسرع وأرخص بكثير من التدريب من الصفر

⚠️ أولوية مطلقة — كل يوم بدون تنفيذ = بيانات ضائعة للأبد
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
import random
import logging

logger = logging.getLogger(__name__)


class Domain(Enum):
    """المجالات العلمية للتعلم"""
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    MATHEMATICS = "mathematics"
    ENGINEERING = "engineering"
    MEDICINE = "medicine"
    AGRICULTURE = "agriculture"
    ECONOMICS = "economics"
    HISTORY = "history"
    GEOGRAPHY = "geography"
    LINGUISTICS = "linguistics"
    PHILOSOPHY = "philosophy"
    COMPUTER_SCIENCE = "computer_science"
    LAW = "law"
    PSYCHOLOGY = "psychology"
    ART = "art"
    MUSIC = "music"
    LITERATURE = "literature"
    ASTRONOMY = "astronomy"
    GEOLOGY = "geology"
    METEOROLOGY = "meteorology"
    OCEANOGRAPHY = "oceanography"
    ECOLOGY = "ecology"
    ARCHAEOLOGY = "archaeology"
    ANTHROPOLOGY = "anthropology"
    POLITICAL_SCIENCE = "political_science"
    SOCIOLOGY = "sociology"
    EDUCATION = "education"
    BUSINESS = "business"
    MANUFACTURING = "manufacturing"


@dataclass
class QuestionTemplate:
    """قالب سؤال"""
    template: str
    domain: Domain
    difficulty: int  # 1-5
    question_type: str  # "concept", "procedure", "comparison", "application", "analysis"
    variables: List[str] = field(default_factory=list)


@dataclass
class QARecord:
    """سجل سؤال وجواب"""
    record_id: str
    question: str
    answer: str
    domain: Domain
    difficulty: int
    question_type: str
    source_model: str  # GPT-4, Claude, etc.
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    quality_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "instruction": self.question,
            "input": "",
            "output": self.answer,
            "domain": self.domain.value,
            "difficulty": self.difficulty,
            "question_type": self.question_type,
            "source_model": self.source_model,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "verified": self.verified,
            "quality_score": self.quality_score
        }


class QuestionGenerator:
    """مولد أسئلة أوتوماتيكي"""
    
    def __init__(self):
        self.templates: Dict[Domain, List[QuestionTemplate]] = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[Domain, List[QuestionTemplate]]:
        """تهيئة قوالب الأسئلة"""
        templates = {
            Domain.PHYSICS: [
                QuestionTemplate(
                    "Explain the concept of {concept} in physics. Include its mathematical formulation and real-world applications.",
                    Domain.PHYSICS, 2, "concept", ["concept"]
                ),
                QuestionTemplate(
                    "How does {phenomenon} work? Describe the underlying physical principles.",
                    Domain.PHYSICS, 3, "procedure", ["phenomenon"]
                ),
                QuestionTemplate(
                    "Compare {concept1} and {concept2}. What are their similarities and differences?",
                    Domain.PHYSICS, 3, "comparison", ["concept1", "concept2"]
                ),
                QuestionTemplate(
                    "A {object} with mass {mass}kg is moving at {velocity}m/s. Calculate its {property}.",
                    Domain.PHYSICS, 2, "application", ["object", "mass", "velocity", "property"]
                ),
                QuestionTemplate(
                    "Analyze the implications of {theory} for our understanding of {phenomenon}.",
                    Domain.PHYSICS, 5, "analysis", ["theory", "phenomenon"]
                ),
            ],
            Domain.CHEMISTRY: [
                QuestionTemplate(
                    "Describe the chemical reaction between {compound1} and {compound2}. What are the products and conditions required?",
                    Domain.CHEMISTRY, 2, "procedure", ["compound1", "compound2"]
                ),
                QuestionTemplate(
                    "What are the properties and uses of {compound}?",
                    Domain.CHEMISTRY, 1, "concept", ["compound"]
                ),
                QuestionTemplate(
                    "Explain the difference between {concept1} and {concept2} in chemistry.",
                    Domain.CHEMISTRY, 2, "comparison", ["concept1", "concept2"]
                ),
                QuestionTemplate(
                    "How would you synthesize {compound} in the laboratory? Provide step-by-step procedures.",
                    Domain.CHEMISTRY, 4, "application", ["compound"]
                ),
                QuestionTemplate(
                    "Analyze the molecular structure of {compound} and explain how it determines its chemical behavior.",
                    Domain.CHEMISTRY, 4, "analysis", ["compound"]
                ),
            ],
            Domain.ENGINEERING: [
                QuestionTemplate(
                    "Design a {system} capable of {capacity}. Include specifications and material selection.",
                    Domain.ENGINEERING, 4, "application", ["system", "capacity"]
                ),
                QuestionTemplate(
                    "What are the key principles of {engineering_field}?",
                    Domain.ENGINEERING, 2, "concept", ["engineering_field"]
                ),
                QuestionTemplate(
                    "Compare {material1} and {material2} for use in {application}. Which is better and why?",
                    Domain.ENGINEERING, 3, "comparison", ["material1", "material2", "application"]
                ),
                QuestionTemplate(
                    "Describe the manufacturing process for {product}. What equipment and quality controls are needed?",
                    Domain.ENGINEERING, 3, "procedure", ["product"]
                ),
                QuestionTemplate(
                    "Analyze the failure modes of {structure} under {condition} conditions.",
                    Domain.ENGINEERING, 5, "analysis", ["structure", "condition"]
                ),
            ],
            Domain.MEDICINE: [
                QuestionTemplate(
                    "What are the symptoms, causes, and treatments for {disease}?",
                    Domain.MEDICINE, 2, "concept", ["disease"]
                ),
                QuestionTemplate(
                    "Explain the mechanism of action of {drug}.",
                    Domain.MEDICINE, 3, "procedure", ["drug"]
                ),
                QuestionTemplate(
                    "Compare {treatment1} and {treatment2} for {condition}.",
                    Domain.MEDICINE, 3, "comparison", ["treatment1", "treatment2", "condition"]
                ),
                QuestionTemplate(
                    "How would you diagnose {condition} in a patient presenting with {symptoms}?",
                    Domain.MEDICINE, 4, "application", ["condition", "symptoms"]
                ),
                QuestionTemplate(
                    "Analyze the epidemiological factors contributing to the spread of {disease}.",
                    Domain.MEDICINE, 4, "analysis", ["disease"]
                ),
            ],
            Domain.MANUFACTURING: [
                QuestionTemplate(
                    "How is {product} manufactured at industrial scale? Describe the complete production line.",
                    Domain.MANUFACTURING, 3, "procedure", ["product"]
                ),
                QuestionTemplate(
                    "What raw materials are needed to produce {product} and in what quantities per ton?",
                    Domain.MANUFACTURING, 2, "concept", ["product"]
                ),
                QuestionTemplate(
                    "Calculate the energy consumption, labor requirements, and cost breakdown for a {product} factory producing {capacity} tons per day.",
                    Domain.MANUFACTURING, 4, "application", ["product", "capacity"]
                ),
                QuestionTemplate(
                    "Compare {process1} and {process2} for manufacturing {product}. Which is more efficient?",
                    Domain.MANUFACTURING, 3, "comparison", ["process1", "process2", "product"]
                ),
                QuestionTemplate(
                    "Analyze the quality control measures needed in {product} manufacturing to ensure safety and consistency.",
                    Domain.MANUFACTURING, 3, "analysis", ["product"]
                ),
            ],
        }
        
        # Add simple templates for all other domains
        for domain in Domain:
            if domain not in templates:
                templates[domain] = [
                    QuestionTemplate(
                        f"Explain the fundamental concepts of {domain.value}.",
                        domain, 2, "concept", []
                    ),
                    QuestionTemplate(
                        f"What are the most important principles in {domain.value}?",
                        domain, 2, "concept", []
                    ),
                    QuestionTemplate(
                        f"How has {domain.value} evolved over time?",
                        domain, 3, "analysis", []
                    ),
                ]
        
        return templates
    
    def generate_question(self, domain: Domain, difficulty: Optional[int] = None) -> str:
        """توليد سؤال عشوائي"""
        templates = self.templates.get(domain, [])
        if not templates:
            return f"Explain {domain.value}"
        
        template = random.choice(templates)
        
        # If difficulty specified, try to match
        if difficulty:
            matching = [t for t in templates if t.difficulty == difficulty]
            if matching:
                template = random.choice(matching)
        
        # Fill in variables with domain-specific terms
        question = template.template
        
        # Simple variable substitution (would be more sophisticated in real implementation)
        variable_values = self._get_variable_values(domain)
        
        for var in template.variables:
            if var in variable_values:
                question = question.replace(f"{{{var}}}", random.choice(variable_values[var]))
            else:
                question = question.replace(f"{{{var}}}", "example")
        
        return question
    
    def _get_variable_values(self, domain: Domain) -> Dict[str, List[str]]:
        """قيم متغيرة لكل مجال"""
        values = {
            Domain.PHYSICS: {
                "concept": ["entropy", "momentum", "electromagnetic induction", "quantum tunneling", "relativity"],
                "phenomenon": ["superconductivity", "nuclear fission", "wave-particle duality", "black holes"],
                "object": ["projectile", "satellite", "electron", "pendulum", "spring"],
                "property": ["kinetic energy", "potential energy", "momentum", "angular momentum"],
                "theory": ["quantum mechanics", "general relativity", "thermodynamics", "electromagnetism"],
            },
            Domain.CHEMISTRY: {
                "compound": ["sodium chloride", "sulfuric acid", "benzene", "iron oxide", "calcium carbonate"],
                "concept": ["covalent bonding", "catalysis", "electrolysis", "polymerization"],
                "concept1": ["acid", "oxidation", "exothermic", "ionic"],
                "concept2": ["base", "reduction", "endothermic", "covalent"],
            },
            Domain.ENGINEERING: {
                "system": ["bridge", "solar panel array", "water treatment plant", "electric motor", "refrigeration system"],
                "capacity": ["supporting 100 tons", "generating 1MW", "processing 1000 m³/day", "operating at 90% efficiency"],
                "material1": ["steel", "aluminum", "concrete", "titanium"],
                "material2": ["carbon fiber", "wood", "plastic", "composite"],
                "application": ["aircraft construction", "building structures", "marine environments", "high-temperature applications"],
            },
            Domain.MANUFACTURING: {
                "product": ["cement", "steel", "glass", "paper", "fertilizer", "solar panels", "batteries"],
                "capacity": ["100", "500", "1000", "5000"],
                "process1": ["batch processing", "wet method", "manual assembly"],
                "process2": ["continuous processing", "dry method", "automated assembly"],
            },
        }
        
        return values.get(domain, {})
    
    def generate_batch(self, domain: Domain, count: int) -> List[str]:
        """توليد دفعة أسئلة"""
        return [self.generate_question(domain) for _ in range(count)]


class KnowledgeDistillationPipeline:
    """
    خط تقطير المعرفة - يجمع بيانات من النماذج الكبيرة
    """
    
    def __init__(self, output_dir: str = "learning_data/distillation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.question_generator = QuestionGenerator()
        self.records: List[QARecord] = []
        
        # API keys (would be loaded from environment)
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.askaichat_key = os.getenv("ASKAICHAT_API_KEY")  # askaichat.app subscription
        self.askaichat_session = os.getenv("ASKAICHAT_SESSION")  # Session cookie if needed
        
        # Statistics
        self.stats = {
            "total_questions": 0,
            "successful_responses": 0,
            "failed_requests": 0,
            "by_domain": {d.value: 0 for d in Domain},
            "by_model": {"gpt-4": 0, "claude": 0, "askaichat": 0}
        }
        
        logger.info("🎓 Knowledge Distillation Pipeline initialized")
    
    async def query_openai(self, question: str, 
                          model: str = "gpt-4") -> Optional[str]:
        """الاستعلام من OpenAI"""
        if not self.openai_key:
            logger.warning("No OpenAI API key found")
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant providing detailed, accurate, and educational answers. Always be thorough and cite principles when applicable."},
                        {"role": "user", "content": question}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
                
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"OpenAI API error: {response.status}")
                        return None
        
        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            return None
    
    async def query_claude(self, question: str) -> Optional[str]:
        """الاستعلام من Anthropic Claude"""
        if not self.anthropic_key:
            logger.warning("No Anthropic API key found")
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "x-api-key": self.anthropic_key,
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "claude-3-opus-20240229",
                    "max_tokens": 2000,
                    "messages": [
                        {"role": "user", "content": question}
                    ]
                }
                
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["content"][0]["text"]
                    else:
                        logger.error(f"Claude API error: {response.status}")
                        return None
        
        except Exception as e:
            logger.error(f"Error querying Claude: {e}")
            return None
    
    async def query_askaichat(self, question: str) -> Optional[str]:
        """
        الاستعلام من askaichat.app
        
        ⚠️ ملاحظة: askaichat.app لا يملك API رسمي مفتوح.
        هذا يحتاج إما:
        1. استخراج session cookie من المتصفح
        2. أو استخدام browser automation (selenium/playwright)
        3. أو الاتصال بالدعم الفني للموقع للحصول على API access
        
        للآن، نستخدم placeholder يطبع تعليمات الإعداد.
        """
        if not self.askaichat_key and not self.askaichat_session:
            logger.warning("""
⚠️  askaichat.app API key not found!

للاستفادة من اشتراك askaichat.app:

الطريقة 1: استخراج Session Cookie
1. افتح https://askaichat.app/chat في المتصفح
2. افتح DevTools (F12) → Network tab
3. أرسل رسالة وانظر للـ Request
4. انسخ الـ Cookie أو Authorization header
5. عينه كـ: export ASKAICHAT_SESSION="..."

الطريقة 2: Browser Automation
# يحتاج تثبيت playwright
pip install playwright
playwright install

ثم استخدام الكود التالي بدلاً من هذا:
```python
from playwright.async_api import async_playwright

async def query_askaichat_browser(question: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("https://askaichat.app/chat")
        # ... تسجيل دخول وإرسال سؤال
```

الطريقة 3: Claude API (أنصح بها)
Claude API يعمل بشكل موثوق ويملك نفس الجودة (أحياناً أحسن)
احصل على مفتاح من: https://console.anthropic.com/
""")
            return None
        
        # Placeholder - يحتاج تنفيذ حقيقي حسب طريقة الوصول
        logger.info("askaichat.app: Would query here with session/cookie")
        return f"[Placeholder] Answer for: {question[:50]}...\n\n(يحتاج إعداد API access)"
    
    async def collect_single(self, domain: Domain, 
                            model_preference: str = "auto") -> Optional[QARecord]:
        """جمع سؤال وجواب واحد"""
        question = self.question_generator.generate_question(domain)
        
        # Determine which model to use (priority: Claude > askaichat > OpenAI)
        available_models = []
        if self.anthropic_key:
            available_models.append("claude")
        if self.askaichat_key or self.askaichat_session:
            available_models.append("askaichat")
        if self.openai_key:
            available_models.append("gpt-4")
        
        if model_preference == "auto":
            if available_models:
                model = random.choice(available_models)
            else:
                logger.error("No AI APIs configured! Set ANTHROPIC_API_KEY, ASKAICHAT_SESSION, or OPENAI_API_KEY")
                return None
        else:
            model = model_preference
        
        # Query the model
        if model == "gpt-4":
            answer = await self.query_openai(question, "gpt-4")
        elif model == "claude":
            answer = await self.query_claude(question)
        elif model == "askaichat":
            answer = await self.query_askaichat(question)
        else:
            answer = None
        
        if not answer:
            self.stats["failed_requests"] += 1
            return None
        
        # Create record
        record = QARecord(
            record_id=f"{domain.value}_{datetime.now().timestamp()}",
            question=question,
            answer=answer,
            domain=domain,
            difficulty=2,  # Would be determined from template
            question_type="concept",
            source_model=model,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.records.append(record)
        self.stats["successful_responses"] += 1
        self.stats["by_domain"][domain.value] += 1
        self.stats["by_model"][model] += 1
        
        return record
    
    async def collect_batch(self, domain: Domain, count: int, 
                           model: str = "auto") -> List[QARecord]:
        """جمع دفعة أسئلة"""
        logger.info(f"🎯 Collecting {count} samples from {domain.value}")
        
        tasks = [self.collect_single(domain, model) for _ in range(count)]
        results = await asyncio.gather(*tasks)
        
        valid_records = [r for r in results if r is not None]
        logger.info(f"✅ Collected {len(valid_records)}/{count} valid records")
        
        return valid_records
    
    async def run_collection_session(self, target_total: int = 10000,
                                     daily_target: int = 10000) -> Dict[str, Any]:
        """
        جمع يومي مستمر
        
        target_total: الهدف الكلي (100,000)
        daily_target: الهدف اليومي (10,000)
        """
        logger.info(f"🚀 Starting collection session: target={daily_target}")
        
        collected_today = 0
        domain_cycle = list(Domain)
        domain_idx = 0
        
        while collected_today < daily_target:
            domain = domain_cycle[domain_idx % len(domain_cycle)]
            domain_idx += 1
            
            # Collect batch for this domain
            batch_size = min(100, daily_target - collected_today)
            records = await self.collect_batch(domain, batch_size)
            
            collected_today += len(records)
            
            # Save periodically
            if len(self.records) >= 1000:
                await self._save_batch()
            
            # Rate limiting
            await asyncio.sleep(1)
        
        # Final save
        await self._save_batch()
        
        return {
            "collected_today": collected_today,
            "total_collected": self.stats["successful_responses"],
            "stats": self.stats
        }
    
    async def _save_batch(self):
        """حفظ الدفعة للملف"""
        if not self.records:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"distillation_{timestamp}.jsonl"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for record in self.records:
                    f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')
            
            logger.info(f"💾 Saved {len(self.records)} records to {filename}")
            self.records = []
            
        except Exception as e:
            logger.error(f"Error saving batch: {e}")
    
    async def export_for_training(self, output_file: str, 
                                  min_quality: Optional[float] = None) -> int:
        """تصدير للتدريب بصيغة Alpaca"""
        all_records = []
        
        # Load all saved files
        for file in self.output_dir.glob("distillation_*.jsonl"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            all_records.append(data)
                        except:
                            continue
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
        
        # Export in Alpaca format
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in all_records:
                alpaca_record = {
                    "instruction": record["instruction"],
                    "input": "",
                    "output": record["output"],
                    "domain": record.get("domain", "general"),
                    "source": record.get("source_model", "unknown")
                }
                f.write(json.dumps(alpaca_record, ensure_ascii=False) + '\n')
        
        logger.info(f"📤 Exported {len(all_records)} records to {output_file}")
        return len(all_records)
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات"""
        return {
            **self.stats,
            "records_in_memory": len(self.records),
            "output_directory": str(self.output_dir),
            "files_saved": len(list(self.output_dir.glob("distillation_*.jsonl")))
        }


# Global instance
distillation_pipeline = KnowledgeDistillationPipeline()
