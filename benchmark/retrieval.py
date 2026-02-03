"""Retrieval orchestration module."""
from typing import List, Dict, Any, Tuple, Union
from benchmark.embeddings import EmbeddingModel
from benchmark.vector_dbs.chroma import ChromaVectorDB
from benchmark.vector_dbs.weaviate import WeaviateVectorDB
from utils.timer import Timer
from utils.logger import logger


VectorDB = Union[ChromaVectorDB, WeaviateVectorDB]


class RetrievalSystem:
    """Orchestrates embedding and vector search operations."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_db: VectorDB
    ):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
    
    def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Index documents into vector database.
        
        Returns:
            Dictionary with timing metrics
        """
        logger.stage(f"Indexing {len(documents)} documents")
        
        # Extract document data
        doc_ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Embed documents
        timer_embed = Timer()
        timer_embed.start()
        embeddings = self.embedding_model.embed_documents(texts)
        embedding_time = timer_embed.stop()
        
        logger.metric("embedding_time_ms", f"{embedding_time:.2f}")
        
        # Insert into vector DB
        indexing_time = self.vector_db.insert_documents(
            doc_ids=doc_ids,
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas
        )
        
        logger.metric("indexing_time_ms", f"{indexing_time:.2f}")
        
        return {
            "embedding_time_ms": embedding_time,
            "indexing_time_ms": indexing_time,
            "total_time_ms": embedding_time + indexing_time,
            "document_count": len(documents)
        }
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> Tuple[List[str], List[str], List[float], Dict[str, float]]:
        """Retrieve relevant documents for a query.
        
        Returns:
            Tuple of (doc_ids, texts, scores, timing_metrics)
        """
        # Embed query
        timer_embed = Timer()
        timer_embed.start()
        query_embedding = self.embedding_model.embed_query(query)
        query_embed_time = timer_embed.stop()
        
        # Search vector DB
        doc_ids, texts, scores, search_time = self.vector_db.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        metrics = {
            "query_embedding_time_ms": query_embed_time,
            "vector_search_time_ms": search_time,
            "total_retrieval_time_ms": query_embed_time + search_time,
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