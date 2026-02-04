"""Retrieval orchestration module - FIXED VERSION."""
from typing import List, Dict, Any, Tuple, Union
from benchmark.embeddings import EmbeddingModel
from benchmark.vector_dbs.chroma import ChromaVectorDB
from benchmark.vector_dbs.weaviate import WeaviateVectorDB
from utils.timer import Timer
from utils.logger import logger

VectorDB = Union[ChromaVectorDB, WeaviateVectorDB]

class RetrievalSystem:
    def __init__(self, embedding_model: EmbeddingModel, vector_db: VectorDB):
        self.embedding_model = embedding_model
        self.vector_db = vector_db

    def _ensure_float(self, val: Any) -> float:
        """Ensure value is a float, handling various input types."""
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            # Remove 'ms' suffix if present and convert
            cleaned = val.replace("ms", "").strip()
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                return 0.0
        return 0.0

    def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index documents with proper float handling."""
        doc_ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        # Time the embedding process
        timer_embed = Timer()
        timer_embed.start()
        embeddings = self.embedding_model.embed_documents(texts)
        embed_time = self._ensure_float(timer_embed.stop())

        # Time the vector database insertion
        raw_db_time = self.vector_db.insert_documents(
            doc_ids=doc_ids,
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas
        )
        db_time = self._ensure_float(raw_db_time)

        total_time = embed_time + db_time

        return {
            "embedding_time_ms": embed_time,
            "indexing_time_ms": db_time,
            "total_time_ms": total_time,
            "document_count": len(documents)
        }

    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[str], List[str], List[float], Dict[str, float]]:
        """Retrieve documents with proper float handling."""
        timer_embed = Timer()
        timer_embed.start()
        query_embedding = self.embedding_model.embed_query(query)
        q_embed_time = self._ensure_float(timer_embed.stop())

        doc_ids, texts, scores, raw_search_time = self.vector_db.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        search_time = self._ensure_float(raw_search_time)

        total_time = q_embed_time + search_time

        metrics = {
            "query_embedding_time_ms": q_embed_time,
            "vector_search_time_ms": search_time,
            "total_retrieval_time_ms": total_time,
            "results_count": len(doc_ids)
        }
        return doc_ids, texts, scores, metrics

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "embedding_model": self.embedding_model.name,
            "embedding_dimension": self.embedding_model.get_dimension(),
            "vector_db": self.vector_db.name,
            "vector_db_stats": self.vector_db.get_stats()
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.vector_db.cleanup()