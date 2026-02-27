"""
Vector Database Integration
ChromaDB/FAISS based vector storage for semantic search and Council AI integration
"""

import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import hashlib


@dataclass
class EmbeddingDocument:
    """Document with embedding vector."""
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'embedding': self.embedding.tolist(),
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class BaseVectorStore:
    """Base class for vector stores."""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.documents: Dict[str, EmbeddingDocument] = {}
    
    def store_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """Store embedding in vector store."""
        raise NotImplementedError
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[EmbeddingDocument]:
        """Search for similar embeddings."""
        raise NotImplementedError
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document from store."""
        raise NotImplementedError
    
    def get_document(self, doc_id: str) -> Optional[EmbeddingDocument]:
        """Get document by ID."""
        return self.documents.get(doc_id)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID for document."""
        return hashlib.md5(f"{text}_{datetime.now().timestamp()}".encode()).hexdigest()


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(self, embedding_dim: int = 768, index_type: str = 'flat'):
        super().__init__(embedding_dim)
        self.index_type = index_type
        self.index = None
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self._init_index()
    
    def _init_index(self):
        """Initialize FAISS index."""
        try:
            import faiss
            
            if self.index_type == 'flat':
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine with normalized vectors)
            elif self.index_type == 'ivf':
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
            elif self.index_type == 'hnsw':
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            
            print(f"FAISS index initialized: {self.index_type}")
            
        except ImportError:
            print("FAISS not available, using numpy fallback")
            self.index = None
    
    def store_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """Store embedding in FAISS index."""
        if doc_id is None:
            doc_id = self._generate_id(text)
        
        # Normalize embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        # Create document
        doc = EmbeddingDocument(
            id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        # Store document
        self.documents[doc_id] = doc
        
        # Add to index
        if self.index is not None:
            import faiss
            
            idx = len(self.id_to_index)
            self.id_to_index[doc_id] = idx
            self.index_to_id[idx] = doc_id
            
            # Reshape for FAISS
            vec = embedding.reshape(1, -1).astype('float32')
            
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                self.index.train(vec)
            
            self.index.add(vec)
        
        return doc_id
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[EmbeddingDocument]:
        """Search for similar embeddings using FAISS."""
        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        if self.index is not None and len(self.documents) > 0:
            import faiss
            
            # Search in FAISS
            query = query_embedding.reshape(1, -1).astype('float32')
            distances, indices = self.index.search(query, min(k * 2, len(self.documents)))
            
            results = []
            for idx in indices[0]:
                if idx == -1:
                    continue
                
                doc_id = self.index_to_id.get(int(idx))
                if doc_id and doc_id in self.documents:
                    doc = self.documents[doc_id]
                    
                    # Apply metadata filter
                    if filter_metadata and not self._matches_filter(doc, filter_metadata):
                        continue
                    
                    results.append(doc)
                    
                    if len(results) >= k:
                        break
            
            return results
        else:
            # Fallback to numpy
            return self._numpy_search(query_embedding, k, filter_metadata)
    
    def _numpy_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        filter_metadata: Optional[Dict]
    ) -> List[EmbeddingDocument]:
        """Fallback numpy-based search."""
        scores = []
        
        for doc_id, doc in self.documents.items():
            # Apply metadata filter
            if filter_metadata and not self._matches_filter(doc, filter_metadata):
                continue
            
            similarity = self._cosine_similarity(query_embedding, doc.embedding)
            scores.append((doc, similarity))
        
        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scores[:k]]
    
    def _matches_filter(self, doc: EmbeddingDocument, filter_metadata: Dict) -> bool:
        """Check if document matches metadata filter."""
        for key, value in filter_metadata.items():
            if doc.metadata.get(key) != value:
                return False
        return True
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document (note: FAISS doesn't support deletion, mark as deleted)."""
        if doc_id in self.documents:
            self.documents[doc_id].metadata['deleted'] = True
            return True
        return False
    
    def save(self, filepath: str) -> None:
        """Save index and documents."""
        import faiss
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(filepath) + '.faiss')
        
        # Save documents
        docs_data = {k: v.to_dict() for k, v in self.documents.items()}
        with open(str(filepath) + '.json', 'w') as f:
            json.dump(docs_data, f)
        
        # Save mappings
        mappings = {
            'id_to_index': self.id_to_index,
            'index_to_id': self.index_to_id
        }
        with open(str(filepath) + '_mappings.json', 'w') as f:
            json.dump(mappings, f)
    
    def load(self, filepath: str) -> None:
        """Load index and documents."""
        import faiss
        
        filepath = Path(filepath)
        
        # Load FAISS index
        if (filepath.parent / (filepath.name + '.faiss')).exists():
            self.index = faiss.read_index(str(filepath) + '.faiss')
        
        # Load documents
        with open(str(filepath) + '.json', 'r') as f:
            docs_data = json.load(f)
            
        self.documents = {}
        for doc_id, doc_data in docs_data.items():
            self.documents[doc_id] = EmbeddingDocument(
                id=doc_data['id'],
                text=doc_data['text'],
                embedding=np.array(doc_data['embedding']),
                metadata=doc_data['metadata'],
                timestamp=datetime.fromisoformat(doc_data['timestamp'])
            )
        
        # Load mappings
        with open(str(filepath) + '_mappings.json', 'r') as f:
            mappings = json.load(f)
            self.id_to_index = {k: int(v) for k, v in mappings['id_to_index'].items()}
            self.index_to_id = {int(k): v for k, v in mappings['index_to_id'].items()}


class ChromaDBStore(BaseVectorStore):
    """ChromaDB-based vector store."""
    
    def __init__(self, collection_name: str = 'council_memory', embedding_dim: int = 768):
        super().__init__(embedding_dim)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._init_chroma()
    
    def _init_chroma(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="data/chromadb"
            ))
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"ChromaDB collection initialized: {self.collection_name}")
            
        except ImportError:
            print("ChromaDB not available")
            self.client = None
    
    def store_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """Store embedding in ChromaDB."""
        if doc_id is None:
            doc_id = self._generate_id(text)
        
        # Create document
        doc = EmbeddingDocument(
            id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        self.documents[doc_id] = doc
        
        # Add to ChromaDB
        if self.collection is not None:
            # Prepare metadata (ChromaDB requires simple types)
            chroma_metadata = {
                'text': text[:1000],  # Truncate for storage
                'timestamp': doc.timestamp.isoformat(),
                **{k: str(v) for k, v in (metadata or {}).items()}
            }
            
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding.tolist()],
                metadatas=[chroma_metadata]
            )
        
        return doc_id
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[EmbeddingDocument]:
        """Search for similar embeddings in ChromaDB."""
        if self.collection is not None:
            # Prepare where clause for filtering
            where_clause = None
            if filter_metadata:
                where_clause = {k: str(v) for k, v in filter_metadata.items()}
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where_clause
            )
            
            documents = []
            for i, doc_id in enumerate(results['ids'][0]):
                if doc_id in self.documents:
                    documents.append(self.documents[doc_id])
            
            return documents
        else:
            # Fallback
            return self._numpy_search(query_embedding, k, filter_metadata)
    
    def _numpy_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        filter_metadata: Optional[Dict]
    ) -> List[EmbeddingDocument]:
        """Fallback numpy-based search."""
        scores = []
        
        for doc_id, doc in self.documents.items():
            if filter_metadata:
                match = all(doc.metadata.get(k) == v for k, v in filter_metadata.items())
                if not match:
                    continue
            
            similarity = self._cosine_similarity(query_embedding, doc.embedding)
            scores.append((doc, similarity))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scores[:k]]
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document from ChromaDB."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            
            if self.collection is not None:
                self.collection.delete(ids=[doc_id])
            
            return True
        return False


class VectorStore:
    """
    Unified vector store interface with embedding generation.
    Integrates with Council AI for memory retrieval.
    """
    
    def __init__(
        self,
        backend: str = 'faiss',
        embedding_dim: int = 768,
        embedding_model = None
    ):
        """
        Initialize vector store.
        
        Args:
            backend: 'faiss' or 'chromadb'
            embedding_dim: Embedding dimension
            embedding_model: Model for generating embeddings
        """
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        
        # Initialize backend
        if backend == 'faiss':
            self._backend = FAISSVectorStore(embedding_dim)
        elif backend == 'chromadb':
            self._backend = ChromaDBStore(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self.embedding_model:
            # Use provided model
            return self.embedding_model.encode(text)
        else:
            # Use simple fallback (random for demo)
            # In production, use sentence-transformers or similar
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.randn(self.embedding_dim)
            return embedding / np.linalg.norm(embedding)
    
    def store(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Store text with automatically generated embedding.
        
        Returns:
            Document ID
        """
        embedding = self.embed_text(text)
        return self._backend.store_embedding(text, embedding, metadata, doc_id)
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar texts.
        
        Returns:
            List of results with text and similarity score
        """
        query_embedding = self.embed_text(query)
        results = self._backend.similarity_search(query_embedding, k, filter_metadata)
        
        return [
            {
                'id': doc.id,
                'text': doc.text,
                'metadata': doc.metadata,
                'similarity': self._backend._cosine_similarity(query_embedding, doc.embedding),
                'timestamp': doc.timestamp.isoformat()
            }
            for doc in results
        ]
    
    def get_relevant_context(
        self,
        query: str,
        k: int = 3,
        min_similarity: float = 0.5
    ) -> str:
        """
        Get relevant context string for query.
        
        Returns:
            Concatenated relevant texts
        """
        results = self.search(query, k=k)
        
        # Filter by similarity
        relevant = [r for r in results if r['similarity'] >= min_similarity]
        
        if not relevant:
            return ""
        
        # Combine texts
        context_parts = []
        for i, result in enumerate(relevant, 1):
            context_parts.append(f"[{i}] {result['text']}")
        
        return "\n\n".join(context_parts)
    
    def delete(self, doc_id: str) -> bool:
        """Delete document."""
        return self._backend.delete_document(doc_id)
    
    def save(self, filepath: str) -> None:
        """Save store."""
        if hasattr(self._backend, 'save'):
            self._backend.save(filepath)
    
    def load(self, filepath: str) -> None:
        """Load store."""
        if hasattr(self._backend, 'load'):
            self._backend.load(filepath)


# Council AI Integration
def enhance_council_context(
    vector_store: VectorStore,
    council_context: Dict[str, Any],
    query: str,
    k: int = 3
) -> Dict[str, Any]:
    """
    Enhance Council AI context with vector search results.
    
    Args:
        vector_store: Vector store instance
        council_context: Current council context
        query: Current query
        k: Number of relevant memories to retrieve
        
    Returns:
        Enhanced context with relevant memories
    """
    # Search for relevant memories
    relevant_memories = vector_store.search(query, k=k)
    
    # Add to context
    if 'relevant_memories' not in council_context:
        council_context['relevant_memories'] = []
    
    for memory in relevant_memories:
        council_context['relevant_memories'].append({
            'text': memory['text'],
            'similarity': memory['similarity'],
            'metadata': memory['metadata']
        })
    
    return council_context


if __name__ == '__main__':
    print("Vector DB Module Demo")
    print("="*50)
    
    # Create vector store
    store = VectorStore(backend='faiss', embedding_dim=128)
    
    # Store some documents
    docs = [
        "Python is a versatile programming language",
        "Machine learning uses neural networks",
        "Docker containers are used for deployment",
        "Arabic is a Semitic language",
        "React is a JavaScript library for building UIs"
    ]
    
    for i, doc in enumerate(docs):
        store.store(doc, metadata={'index': i, 'category': 'general'})
    
    print(f"Stored {len(docs)} documents")
    
    # Search
    query = "programming languages"
    results = store.search(query, k=3)
    
    print(f"\nSearch results for '{query}':")
    for r in results:
        print(f"  - {r['text'][:50]}... (similarity: {r['similarity']:.3f})")
    
    # Get relevant context
    context = store.get_relevant_context(query, k=2)
    print(f"\nRelevant context:\n{context}")
