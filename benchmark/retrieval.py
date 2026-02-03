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
    ) -> Dict[str, Any]:
        """Index documents into vector database."""
        logger.stage(f"Indexing {len(documents)} documents")
        
        doc_ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # 1. Capture embedding time as float
        timer_embed = Timer()
        timer_embed.start()
        embeddings = self.embedding_model.embed_documents(texts)
        embedding_time = float(timer_embed.stop())
        
        # Log for display but keep the float variable for math
        logger.metric("embedding_time_ms", f"{embedding_time:.2f}")
        
        # 2. Capture indexing time
        # Ensure your VectorDB.insert_documents returns a float!
        raw_indexing_time = self.vector_db.insert_documents(
            doc_ids=doc_ids,
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas
        )
        
        try:
            indexing_time = float(raw_indexing_time)
        except (TypeError, ValueError):
            logger.error(f"VectorDB returned non-numeric time: {raw_indexing_time}")
            indexing_time = 0.0
        
        logger.metric("indexing_time_ms", f"{indexing_time:.2f}")
        
        return {
            "embedding_time_ms": embedding_time,
            "indexing_time_ms": indexing_time,
            "total_time_ms": embedding_time + indexing_time, # Now safe
            "document_count": len(documents)
        }
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> Tuple[List[str], List[str], List[float], Dict[str, float]]:
        """Retrieve relevant documents for a query."""
        timer_embed = Timer()
        timer_embed.start()
        query_embedding = self.embedding_model.embed_query(query)
        query_embed_time = float(timer_embed.stop())
        
        # 3. Ensure search returns a numeric search_time
        doc_ids, texts, scores, raw_search_time = self.vector_db.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        search_time = float(raw_search_time) if raw_search_time is not None else 0.0
        
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