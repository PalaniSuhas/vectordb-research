"""ChromaDB vector database implementation."""
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from utils.timer import Timer
from utils.logger import logger


class ChromaVectorDB:
    """ChromaDB implementation for vector storage and retrieval."""
    
    def __init__(self, collection_name: str, dimension: int):
        """Initialize ChromaDB client."""
        self.collection_name = collection_name
        self.dimension = dimension
        
        # Create ephemeral client (in-memory)
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Delete collection if exists
        try:
            self.client.delete_collection(name=collection_name)
        except:
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"dimension": dimension}
        )
        
        logger.info(f"ChromaDB collection '{collection_name}' created (dimension={dimension})")
    
    def insert_documents(
        self,
        doc_ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> float:
        """Insert documents into ChromaDB.
        
        Returns:
            Indexing time in milliseconds
        """
        timer = Timer()
        timer.start()
        
        try:
            self.collection.add(
                ids=doc_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            elapsed = timer.stop()
            logger.metric(f"ChromaDB_indexing_time", f"{elapsed:.2f}ms")
            
            return elapsed
        except Exception as e:
            logger.error(f"ChromaDB insert failed: {str(e)}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3
    ) -> Tuple[List[str], List[str], List[float], float]:
        """Search for similar documents.
        
        Returns:
            Tuple of (doc_ids, texts, scores, query_latency_ms)
        """
        timer = Timer()
        timer.start()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "distances", "metadatas"]
            )
            
            elapsed = timer.stop()
            logger.metric(f"ChromaDB_search_latency", f"{elapsed:.2f}ms")
            
            # Extract results
            doc_ids = results['ids'][0] if results['ids'] else []
            texts = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            # Convert distances to similarity scores (ChromaDB uses L2 distance)
            # Lower distance = higher similarity
            scores = [1.0 / (1.0 + dist) for dist in distances]
            
            return doc_ids, texts, scores, elapsed
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        count = self.collection.count()
        return {
            "document_count": count,
            "dimension": self.dimension,
            "collection_name": self.collection_name
        }
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"ChromaDB collection '{self.collection_name}' deleted")
        except:
            pass
    
    @property
    def name(self) -> str:
        return "ChromaDB"