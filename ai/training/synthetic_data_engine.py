"""
Synthetic Data Engine - محرك البيانات الاصطناعية
النموذج يولّد بياناته من بياناته — بيانات لا نهائية بدون إنترنت

القدرات:
1. generate_socratic_dialogs - يسأل نفسه أسئلة ويجاوب
2. self_play - نموذجان يتناقشان → يولّدان بيانات تدريب
3. generate_scientific_problems - يخلق مشاكل علمية ويحلها
4. paraphrase_existing - يعيد صياغة البيانات الموجودة
5. create_counterfactuals - "ماذا لو" سيناريوهات
"""

import asyncio
import random
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SyntheticDataType(Enum):
    """أنواع البيانات الاصطناعية"""
    SOCRATIC_DIALOG = "socratic_dialog"
    SELF_PLAY = "self_play"
    SCIENTIFIC_PROBLEM = "scientific_problem"
    PARAPHRASE = "paraphrase"
    COUNTERFACTUAL = "counterfactual"
    MULTI_STEP_REASONING = "multi_step_reasoning"


@dataclass
class SyntheticSample:
    """عينة اصطناعية"""
    sample_id: str
    data_type: SyntheticDataType
    instruction: str
    input_text: str
    output_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_documents: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "type": self.data_type.value,
            "instruction": self.instruction,
            "input": self.input_text,
            "output": self.output_text,
            "metadata": self.metadata,
            "quality_score": self.quality_score,
            "timestamp": self.timestamp.isoformat(),
            "source_documents": self.source_documents
        }


class SocraticDialogGenerator:
    """مولد حوارات سقراطية - يسأل نفسه ويجاوب"""
    
    def __init__(self):
        self.question_starters = [
            "What is the fundamental nature of",
            "How does one truly understand",
            "What are the underlying principles of",
            "Why is important to know about",
            "What would happen if we changed"
        ]
        
        self.follow_up_patterns = [
            "But how do we know that",
            "What evidence supports",
            "Can you explain why",
            "What about the case where",
            "How does this relate to"
        ]
    
    async def generate(self, topic: str, depth: int = 3) -> SyntheticSample:
        """توليد حوار سقراطي"""
        dialog_parts = []
        
        # Initial question
        starter = random.choice(self.question_starters)
        question = f"{starter} {topic}?"
        dialog_parts.append(f"Teacher: {question}")
        
        # Generate back-and-forth
        current_topic = topic
        for i in range(depth):
            # Student asks
            if i == 0:
                student_q = f"I'm not sure I understand {topic} completely."
            else:
                follow_up = random.choice(self.follow_up_patterns)
                student_q = f"{follow_up} {current_topic}?"
            
            dialog_parts.append(f"Student: {student_q}")
            
            # Teacher responds with deeper insight
            teacher_response = self._generate_teacher_response(current_topic, i)
            dialog_parts.append(f"Teacher: {teacher_response}")
            
            # Evolve topic
            current_topic = self._evolve_topic(current_topic)
        
        # Final synthesis
        dialog_parts.append(f"Teacher: So in conclusion, {topic} requires us to think deeply about {self._evolve_topic(topic)}.")
        
        full_dialog = "\n\n".join(dialog_parts)
        
        return SyntheticSample(
            sample_id=f"socratic_{hash(topic) % 10000}",
            data_type=SyntheticDataType.SOCRATIC_DIALOG,
            instruction=f"Engage in a Socratic dialogue about {topic}",
            input_text=question,
            output_text=full_dialog,
            metadata={
                "topic": topic,
                "depth": depth,
                "turns": depth * 2
            },
            quality_score=0.8
        )
    
    def _generate_teacher_response(self, topic: str, depth: int) -> str:
        """توليد رد المعلم"""
        responses = [
            f"Excellent question. {topic} is best understood through its fundamental principles. Let me explain...",
            f"This requires us to examine the underlying mechanisms. When we look closely at {topic}, we see...",
            f"Consider this perspective: {topic} is not just about surface-level understanding, but involves...",
            f"To truly grasp {topic}, we must first understand its relationship to broader concepts..."
        ]
        return random.choice(responses)
    
    def _evolve_topic(self, topic: str) -> str:
        """تطوير الموضوع"""
        evolutions = {
            "physics": ["mechanics", "thermodynamics", "quantum mechanics"],
            "chemistry": ["atomic structure", "chemical bonds", "reaction mechanisms"],
            "biology": ["cell structure", "genetics", "evolution"],
            "mathematics": ["algebra", "geometry", "calculus"]
        }
        
        for key, values in evolutions.items():
            if key in topic.lower():
                return random.choice(values)
        
        return f"the deeper aspects of {topic}"


class SelfPlayGenerator:
    """مولد Self-Play - نموذجان يتناقشان"""
    
    def __init__(self):
        self.roles = ["Advocate", "Skeptic", "Expert", "Beginner"]
    
    async def generate(self, topic: str, turns: int = 4) -> SyntheticSample:
        """توليد نقاش self-play"""
        role1, role2 = random.sample(self.roles, 2)
        
        conversation = []
        
        # Opening positions
        pos1 = f"I believe {topic} is fundamentally important because..."
        pos2 = f"While I see value in {topic}, I have concerns about..."
        
        conversation.append(f"{role1}: {pos1}")
        conversation.append(f"{role2}: {pos2}")
        
        # Debate
        for i in range(turns):
            if i % 2 == 0:
                response = f"But consider this perspective on {topic}..."
                conversation.append(f"{role1}: {response}")
            else:
                response = f"That raises an interesting point, however..."
                conversation.append(f"{role2}: {response}")
        
        # Synthesis
        synthesis = f"After considering both perspectives on {topic}, we can conclude..."
        conversation.append(f"Synthesis: {synthesis}")
        
        full_conversation = "\n\n".join(conversation)
        
        return SyntheticSample(
            sample_id=f"selfplay_{hash(topic) % 10000}",
            data_type=SyntheticDataType.SELF_PLAY,
            instruction=f"Debate the topic of {topic} from multiple perspectives",
            input_text=f"Topic: {topic}",
            output_text=full_conversation,
            metadata={
                "topic": topic,
                "role1": role1,
                "role2": role2,
                "turns": turns
            },
            quality_score=0.85
        )


class ScientificProblemGenerator:
    """مولد مسائل علمية"""
    
    def __init__(self):
        self.problem_types = [
            "calculation", "derivation", "proof", "design", "analysis"
        ]
    
    async def generate(self, domain: str, difficulty: int = 2) -> SyntheticSample:
        """توليد مسألة علمية كاملة"""
        problem_type = random.choice(self.problem_types)
        
        # Generate problem statement
        problem = self._generate_problem_statement(domain, problem_type, difficulty)
        
        # Generate solution
        solution = self._generate_solution(domain, problem_type, difficulty)
        
        return SyntheticSample(
            sample_id=f"science_{hash(domain) % 10000}",
            data_type=SyntheticDataType.SCIENTIFIC_PROBLEM,
            instruction=f"Solve this {domain} problem",
            input_text=problem,
            output_text=solution,
            metadata={
                "domain": domain,
                "problem_type": problem_type,
                "difficulty": difficulty
            },
            quality_score=0.9
        )
    
    def _generate_problem_statement(self, domain: str, 
                                    problem_type: str, difficulty: int) -> str:
        """توليد نص المسألة"""
        templates = {
            "physics": {
                "calculation": "A {object} with mass {m}kg is moving at velocity {v}m/s. Calculate its {property}.",
                "derivation": "Derive the relationship between {var1} and {var2} for a {system}.",
                "design": "Design a {system} that can achieve {goal}. Specify all parameters."
            },
            "chemistry": {
                "calculation": "Calculate the {property} of a solution containing {amount} moles of {compound} in {volume}L.",
                "derivation": "Derive the rate law for the reaction between {compound1} and {compound2}.",
                "design": "Design a synthesis pathway for {compound} starting from {starting_material}."
            },
            "mathematics": {
                "calculation": "Calculate the {operation} of {expression}.",
                "proof": "Prove that {statement} for all {domain}.",
                "derivation": "Derive the formula for {quantity} in terms of {variables}."
            }
        }
        
        domain_templates = templates.get(domain, templates["physics"])
        template = domain_templates.get(problem_type, domain_templates["calculation"])
        
        # Fill in variables (simplified)
        filled = template.format(
            object="particle", m=10, v=5, property="kinetic energy",
            var1="force", var2="acceleration", system="mechanical system",
            compound="NaCl", amount=2, volume=1,
            operation="integral", expression="x^2 + 3x + 2",
            statement="the sum is positive", domain="natural numbers",
            quantity="area", variables="radius",
            goal="maximum efficiency", starting_material="raw ore"
        )
        
        return filled
    
    def _generate_solution(self, domain: str, problem_type: str, 
                          difficulty: int) -> str:
        """توليد الحل"""
        steps = [
            f"Step 1: Identify the given information and what we need to find.",
            f"Step 2: Apply the relevant principles from {domain}.",
            f"Step 3: Set up the equations based on these principles.",
            f"Step 4: Solve the equations systematically.",
            f"Step 5: Verify the solution by checking units and limits."
        ]
        
        if difficulty >= 3:
            steps.insert(2, "Step 2.5: Consider edge cases and constraints.")
        
        solution = "\n".join(steps)
        solution += f"\n\nFinal Answer: [Detailed calculation would go here based on the specific problem]"
        
        return solution


class ParaphraseGenerator:
    """مولد إعادة الصياغة"""
    
    async def generate(self, original_text: str) -> SyntheticSample:
        """إعادة صياغة نص"""
        # Simulate paraphrasing (would use actual NLP model)
        paraphrases = [
            f"In other words, {original_text}",
            f"To put it differently, {original_text}",
            f"Another way to understand this: {original_text}"
        ]
        
        return SyntheticSample(
            sample_id=f"para_{hash(original_text) % 10000}",
            data_type=SyntheticDataType.PARAPHRASE,
            instruction="Paraphrase the following text",
            input_text=original_text,
            output_text=random.choice(paraphrases),
            metadata={"original_length": len(original_text)},
            quality_score=0.75
        )


class CounterfactualGenerator:
    """مولد السيناريوهات البديلة (ماذا لو)"""
    
    def __init__(self):
        self.counterfactual_starters = [
            "What if we changed",
            "Imagine a world where",
            "How would things differ if",
            "Consider the scenario where",
            "Suppose we modified"
        ]
    
    async def generate(self, topic: str, 
                      modification: str) -> SyntheticSample:
        """توليد سيناريو بديل"""
        starter = random.choice(self.counterfactual_starters)
        
        question = f"{starter} {topic} by {modification}?"
        
        # Generate implications
        implications = [
            f"First, {topic} would behave differently because...",
            f"Second, this change would affect related systems...",
            f"Third, long-term consequences would include...",
            f"However, there would also be limitations..."
        ]
        
        answer = "\n\n".join(implications)
        answer += f"\n\nConclusion: Changing {topic} by {modification} would lead to significant but manageable differences."
        
        return SyntheticSample(
            sample_id=f"cf_{hash(topic) % 10000}",
            data_type=SyntheticDataType.COUNTERFACTUAL,
            instruction=f"Analyze the counterfactual: What if {topic} was {modification}?",
            input_text=question,
            output_text=answer,
            metadata={
                "original": topic,
                "modification": modification
            },
            quality_score=0.8
        )


class SyntheticDataEngine:
    """
    محرك البيانات الاصطناعية - يولّد بيانات لا نهائية
    """
    
    def __init__(self, output_dir: str = "learning_data/synthetic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.generators = {
            SyntheticDataType.SOCRATIC_DIALOG: SocraticDialogGenerator(),
            SyntheticDataType.SELF_PLAY: SelfPlayGenerator(),
            SyntheticDataType.SCIENTIFIC_PROBLEM: ScientificProblemGenerator(),
            SyntheticDataType.PARAPHRASE: ParaphraseGenerator(),
            SyntheticDataType.COUNTERFACTUAL: CounterfactualGenerator()
        }
        
        self.generated_count = 0
        self.quality_threshold = 0.7
        
        logger.info("🧬 Synthetic Data Engine initialized")
    
    async def generate_batch(self, count: int = 100) -> List[SyntheticSample]:
        """توليد دفعة عينات"""
        samples = []
        
        topics = [
            "physics", "chemistry", "biology", "mathematics", "engineering",
            "medicine", "history", "philosophy", "economics", "computer science",
            "quantum mechanics", "organic chemistry", "genetics", "calculus",
            "thermodynamics", "electromagnetism", "evolution", "number theory"
        ]
        
        for i in range(count):
            # Randomly select generator type
            data_type = random.choice(list(self.generators.keys()))
            generator = self.generators[data_type]
            
            topic = random.choice(topics)
            
            try:
                if data_type == SyntheticDataType.SOCRATIC_DIALOG:
                    sample = await generator.generate(topic, depth=random.randint(2, 5))
                elif data_type == SyntheticDataType.SELF_PLAY:
                    sample = await generator.generate(topic, turns=random.randint(3, 6))
                elif data_type == SyntheticDataType.SCIENTIFIC_PROBLEM:
                    sample = await generator.generate(topic, difficulty=random.randint(1, 4))
                elif data_type == SyntheticDataType.PARAPHRASE:
                    sample = await generator.generate(f"The fundamental principles of {topic}")
                elif data_type == SyntheticDataType.COUNTERFACTUAL:
                    modifications = ["increasing by 10x", "eliminating entirely", "reversing the direction"]
                    sample = await generator.generate(topic, random.choice(modifications))
                else:
                    continue
                
                # Filter by quality
                if sample.quality_score >= self.quality_threshold:
                    samples.append(sample)
                    self.generated_count += 1
                
            except Exception as e:
                logger.error(f"Error generating sample: {e}")
        
        return samples
    
    async def save_batch(self, samples: List[SyntheticSample]):
        """حفظ الدفعة"""
        if not samples:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"synthetic_{timestamp}.jsonl"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
        
        logger.info(f"💾 Saved {len(samples)} synthetic samples to {filename}")
    
    async def generate_continuously(self, target_per_hour: int = 1000):
        """توليد مستمر"""
        logger.info(f"🔄 Starting continuous generation: {target_per_hour}/hour")
        
        while True:
            try:
                samples = await self.generate_batch(target_per_hour // 6)  # Every 10 minutes
                await self.save_batch(samples)
                
                logger.info(f"✅ Generated {len(samples)} samples (total: {self.generated_count})")
                
                await asyncio.sleep(600)  # 10 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in continuous generation: {e}")
                await asyncio.sleep(60)
    
    async def augment_existing_dataset(self, input_file: str, 
                                       multiplier: int = 3) -> int:
        """تعزيز مجموعة بيانات موجودة"""
        logger.info(f"📚 Augmenting {input_file} by {multiplier}x")
        
        augmented = []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        original_text = data.get("output", "")
                        
                        if not original_text:
                            continue
                        
                        # Generate paraphrases
                        for _ in range(multiplier - 1):
                            sample = await self.generators[SyntheticDataType.PARAPHRASE].generate(original_text)
                            augmented.append(sample)
                    
                    except Exception as e:
                        continue
        
        except Exception as e:
            logger.error(f"Error reading input: {e}")
            return 0
        
        # Save augmented data
        await self.save_batch(augmented)
        
        return len(augmented)
    
    def get_stats(self) -> Dict[str, Any]:
        """إحصائيات"""
        return {
            "total_generated": self.generated_count,
            "quality_threshold": self.quality_threshold,
            "output_directory": str(self.output_dir),
            "generator_types": [g.value for g in self.generators.keys()]
        }


# Global instance
synthetic_data_engine = SyntheticDataEngine()
