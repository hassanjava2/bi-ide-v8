"""
Context Awareness Module
Manage context window and summarize long conversations for Council AI
"""

import torch
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from collections import deque


@dataclass
class ContextWindow:
    """Represents a window of context."""
    messages: List[Dict[str, Any]]
    token_count: int
    relevance_score: float
    timestamp: datetime
    summary: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'messages': self.messages,
            'token_count': self.token_count,
            'relevance_score': self.relevance_score,
            'timestamp': self.timestamp.isoformat(),
            'summary': self.summary
        }


@dataclass
class ContextSummary:
    """Summary of a conversation segment."""
    content: str
    key_points: List[str]
    topics: List[str]
    decisions: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'key_points': self.key_points,
            'topics': self.topics,
            'decisions': self.decisions,
            'timestamp': self.timestamp.isoformat()
        }


class ContextSummarizer:
    """Summarize long contexts using extractive or abstractive methods."""
    
    def __init__(self, model=None, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.max_summary_length = 200
    
    def summarize(
        self,
        messages: List[Dict[str, Any]],
        method: str = 'extractive',
        max_length: int = 200
    ) -> ContextSummary:
        """
        Summarize a list of messages.
        
        Args:
            messages: List of message dictionaries
            method: 'extractive', 'abstractive', or 'hybrid'
            max_length: Maximum summary length
            
        Returns:
            ContextSummary object
        """
        if method == 'extractive':
            return self._extractive_summarize(messages, max_length)
        elif method == 'abstractive' and self.model:
            return self._abstractive_summarize(messages, max_length)
        else:
            return self._hybrid_summarize(messages, max_length)
    
    def _extractive_summarize(
        self,
        messages: List[Dict[str, Any]],
        max_length: int
    ) -> ContextSummary:
        """Extract key sentences as summary."""
        # Combine all content
        all_text = ' '.join([msg.get('content', '') for msg in messages])
        
        # Simple extractive summarization based on sentence importance
        sentences = self._split_sentences(all_text)
        
        if len(sentences) <= 3:
            summary_text = all_text
        else:
            # Score sentences
            word_freq = {}
            for sentence in sentences:
                for word in sentence.lower().split():
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            sentence_scores = []
            for sentence in sentences:
                score = sum(word_freq.get(word.lower(), 0) for word in sentence.split())
                sentence_scores.append((sentence, score))
            
            # Select top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in sentence_scores[:min(3, len(sentences))]]
            
            # Restore original order
            summary_text = ' '.join([s for s in sentences if s in top_sentences])
        
        # Extract key points
        key_points = self._extract_key_points(messages)
        
        # Extract topics
        topics = self._extract_topics(all_text)
        
        # Extract decisions
        decisions = self._extract_decisions(messages)
        
        return ContextSummary(
            content=summary_text[:max_length],
            key_points=key_points,
            topics=topics,
            decisions=decisions,
            timestamp=datetime.now()
        )
    
    def _abstractive_summarize(
        self,
        messages: List[Dict[str, Any]],
        max_length: int
    ) -> ContextSummary:
        """Generate abstractive summary using model."""
        # Placeholder for model-based summarization
        # In practice, this would use a T5, BART, or similar model
        
        all_text = ' '.join([msg.get('content', '') for msg in messages])
        
        # For now, fall back to extractive
        return self._extractive_summarize(messages, max_length)
    
    def _hybrid_summarize(
        self,
        messages: List[Dict[str, Any]],
        max_length: int
    ) -> ContextSummary:
        """Combine extractive and abstractive approaches."""
        # Extractive first
        extractive = self._extractive_summarize(messages, max_length // 2)
        
        # Then abstractive if model available
        if self.model:
            abstractive = self._abstractive_summarize(messages, max_length // 2)
            extractive.content = abstractive.content + " " + extractive.content
        
        return extractive
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_key_points(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract key points from messages."""
        key_points = []
        
        for msg in messages:
            content = msg.get('content', '')
            
            # Look for code blocks
            if '```' in content:
                key_points.append("Contains code example")
            
            # Look for questions
            if '?' in content:
                key_points.append(f"Question asked: {content[:50]}...")
            
            # Look for important keywords
            if any(kw in content.lower() for kw in ['important', 'key', 'critical', 'main']):
                key_points.append(f"Important point: {content[:50]}...")
        
        return key_points[:5]  # Limit to 5 key points
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text."""
        # Simple keyword-based topic extraction
        topic_keywords = {
            'python': ['python', 'pytorch', 'django', 'flask'],
            'javascript': ['javascript', 'js', 'react', 'node', 'frontend'],
            'machine_learning': ['ml', 'model', 'training', 'neural', 'ai'],
            'database': ['sql', 'database', 'postgres', 'mysql', 'query'],
            'devops': ['docker', 'kubernetes', 'deployment', 'ci/cd'],
        }
        
        text_lower = text.lower()
        found_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found_topics.append(topic)
        
        return found_topics
    
    def _extract_decisions(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract decisions made during conversation."""
        decisions = []
        
        decision_keywords = ['decide', 'decision', 'choose', 'selected', 'will use', 'going with']
        
        for msg in messages:
            content = msg.get('content', '').lower()
            
            for keyword in decision_keywords:
                if keyword in content:
                    # Extract sentence containing decision
                    sentences = self._split_sentences(content)
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            decisions.append(sentence.strip())
                            break
        
        return decisions[:3]


class ContextManager:
    """
    Manages context window for Council AI.
    Handles context truncation, summarization, and retrieval.
    """
    
    def __init__(
        self,
        max_context_tokens: int = 4096,
        summarizer: Optional[ContextSummarizer] = None,
        tokenizer = None
    ):
        self.max_context_tokens = max_context_tokens
        self.summarizer = summarizer or ContextSummarizer()
        self.tokenizer = tokenizer
        
        # Context storage
        self.active_context: deque = deque(maxlen=100)
        self.summarized_segments: List[ContextSummary] = []
        
        # Statistics
        self.total_messages_processed = 0
        self.total_summaries_created = 0
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Add message to context.
        
        Returns:
            Updated context state
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.active_context.append(message)
        self.total_messages_processed += 1
        
        # Check if context needs summarization
        current_tokens = self._estimate_tokens()
        
        if current_tokens > self.max_context_tokens * 0.8:
            self._summarize_oldest_segment()
        
        return {
            'message_added': True,
            'current_tokens': current_tokens,
            'active_messages': len(self.active_context),
            'summaries': len(self.summarized_segments)
        }
    
    def get_context(
        self,
        max_tokens: Optional[int] = None,
        include_summaries: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get current context for model input.
        
        Args:
            max_tokens: Maximum tokens to include
            include_summaries: Whether to include summarized segments
            
        Returns:
            List of context messages
        """
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        context = []
        
        # Add summaries first (oldest)
        if include_summaries and self.summarized_segments:
            for summary in self.summarized_segments[-3:]:  # Last 3 summaries
                context.append({
                    'role': 'system',
                    'content': f"[Previous conversation summary]: {summary.content}",
                    'is_summary': True
                })
        
        # Add active context (most recent)
        current_tokens = self._estimate_context_tokens(context)
        
        for message in reversed(self.active_context):
            msg_tokens = self._estimate_message_tokens(message)
            
            if current_tokens + msg_tokens > max_tokens:
                break
            
            context.insert(len(self.summarized_segments) if include_summaries else 0, message)
            current_tokens += msg_tokens
        
        return context
    
    def get_full_context(self) -> Dict[str, Any]:
        """Get complete context including all segments."""
        return {
            'active_context': list(self.active_context),
            'summarized_segments': [s.to_dict() for s in self.summarized_segments],
            'total_messages': self.total_messages_processed,
            'total_summaries': self.total_summaries_created
        }
    
    def clear_context(self) -> None:
        """Clear all context."""
        self.active_context.clear()
        self.summarized_segments.clear()
        self.total_messages_processed = 0
        self.total_summaries_created = 0
    
    def search_context(
        self,
        query: str,
        search_summaries: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search context for relevant information.
        
        Args:
            query: Search query
            search_summaries: Whether to search in summaries
            
        Returns:
            List of relevant messages/summaries
        """
        results = []
        query_lower = query.lower()
        
        # Search active context
        for message in self.active_context:
            if query_lower in message.get('content', '').lower():
                results.append(message)
        
        # Search summaries
        if search_summaries:
            for summary in self.summarized_segments:
                if query_lower in summary.content.lower():
                    results.append({
                        'role': 'system',
                        'content': f"[Summary]: {summary.content}",
                        'is_summary': True
                    })
        
        return results
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        return {
            'active_messages': len(self.active_context),
            'summarized_segments': len(self.summarized_segments),
            'estimated_tokens': self._estimate_tokens(),
            'total_messages_processed': self.total_messages_processed,
            'total_summaries_created': self.total_summaries_created
        }
    
    def _estimate_tokens(self) -> int:
        """Estimate total tokens in context."""
        tokens = 0
        
        # Active context
        for message in self.active_context:
            tokens += self._estimate_message_tokens(message)
        
        # Summaries
        for summary in self.summarized_segments:
            tokens += len(summary.content.split())  # Rough estimate
        
        return tokens
    
    def _estimate_message_tokens(self, message: Dict[str, Any]) -> int:
        """Estimate tokens in a message."""
        content = message.get('content', '')
        
        if self.tokenizer:
            return len(self.tokenizer.encode(content))
        else:
            # Rough estimate: ~1.3 tokens per word
            return int(len(content.split()) * 1.3)
    
    def _estimate_context_tokens(self, context: List[Dict]) -> int:
        """Estimate tokens in context list."""
        return sum(self._estimate_message_tokens(msg) for msg in context)
    
    def _summarize_oldest_segment(self) -> None:
        """Summarize oldest portion of active context."""
        if len(self.active_context) < 5:
            return
        
        # Take oldest 20% of messages
        num_to_summarize = max(5, len(self.active_context) // 5)
        messages_to_summarize = []
        
        for _ in range(num_to_summarize):
            if self.active_context:
                messages_to_summarize.append(self.active_context.popleft())
        
        # Create summary
        summary = self.summarizer.summarize(messages_to_summarize)
        self.summarized_segments.append(summary)
        self.total_summaries_created += 1
        
        print(f"Created summary: {summary.content[:100]}...")


class CouncilContextManager(ContextManager):
    """
    Extended context manager for Council AI with agent-specific context.
    """
    
    def __init__(
        self,
        max_context_tokens: int = 4096,
        max_agent_context: int = 2048,
        summarizer: Optional[ContextSummarizer] = None,
        tokenizer = None
    ):
        super().__init__(max_context_tokens, summarizer, tokenizer)
        self.max_agent_context = max_agent_context
        
        # Agent-specific contexts
        self.agent_contexts: Dict[str, ContextManager] = {}
        
        # Shared knowledge
        self.shared_knowledge: Dict[str, Any] = {}
    
    def create_agent_context(self, agent_id: str) -> ContextManager:
        """Create context manager for specific agent."""
        agent_manager = ContextManager(
            max_context_tokens=self.max_agent_context,
            summarizer=self.summarizer,
            tokenizer=self.tokenizer
        )
        self.agent_contexts[agent_id] = agent_manager
        return agent_manager
    
    def get_agent_context(self, agent_id: str) -> Optional[ContextManager]:
        """Get context manager for agent."""
        return self.agent_contexts.get(agent_id)
    
    def add_to_agent_context(
        self,
        agent_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add message to agent-specific context."""
        if agent_id not in self.agent_contexts:
            self.create_agent_context(agent_id)
        
        self.agent_contexts[agent_id].add_message(role, content, metadata)
    
    def get_consolidated_context(
        self,
        active_agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get consolidated context for council.
        
        Returns:
            Context with shared and agent-specific information
        """
        context = {
            'shared_context': self.get_context(),
            'shared_knowledge': self.shared_knowledge,
            'agent_contexts': {}
        }
        
        # Add agent contexts
        for agent_id, agent_manager in self.agent_contexts.items():
            context['agent_contexts'][agent_id] = {
                'context': agent_manager.get_context(max_tokens=self.max_agent_context // 2),
                'stats': agent_manager.get_context_stats()
            }
        
        # Add active agent full context
        if active_agent_id and active_agent_id in self.agent_contexts:
            context['active_agent_context'] = self.agent_contexts[active_agent_id].get_context()
        
        return context
    
    def share_knowledge(self, key: str, value: Any) -> None:
        """Share knowledge across all agents."""
        self.shared_knowledge[key] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_shared_knowledge(self, key: str) -> Optional[Any]:
        """Get shared knowledge."""
        entry = self.shared_knowledge.get(key)
        return entry['value'] if entry else None


def create_context_manager(
    max_tokens: int = 4096,
    use_summarization: bool = True
) -> ContextManager:
    """Create context manager."""
    summarizer = ContextSummarizer() if use_summarization else None
    return ContextManager(
        max_context_tokens=max_tokens,
        summarizer=summarizer
    )


if __name__ == '__main__':
    print("Context Awareness Module Demo")
    print("="*50)
    
    # Create context manager
    manager = create_context_manager(max_tokens=1000)
    
    # Add messages
    for i in range(20):
        manager.add_message(
            'user' if i % 2 == 0 else 'assistant',
            f'Message number {i} with some content about Python programming and AI development.'
        )
    
    # Get stats
    stats = manager.get_context_stats()
    print(f"Context Stats:")
    print(f"  Active messages: {stats['active_messages']}")
    print(f"  Summaries: {stats['summarized_segments']}")
    print(f"  Estimated tokens: {stats['estimated_tokens']}")
    
    # Get context
    context = manager.get_context()
    print(f"\nRetrieved context: {len(context)} messages")
    
    # Search
    results = manager.search_context('Python')
    print(f"Search results: {len(results)} matches")
