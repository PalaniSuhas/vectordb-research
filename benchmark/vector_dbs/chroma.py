"""ChromaDB vector database implementation."""
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from utils.timer import Timer
from utils.logger import logger

class ChromaVectorDB:
    """ChromaDB implementation for vector storage and retrieval."""
    
    def __init__(self, collection_name: str, dimension: int):
        self.collection_name = collection_name
        self.dimension = dimension
        self.client = chromadb.EphemeralClient(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
        
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB collection '{collection_name}' created (dim: {dimension})")

    def insert_documents(
        self,
        doc_ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> float:
        timer = Timer()
        timer.start()
        
        try:
            self.collection.add(
                ids=doc_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            elapsed = timer.stop()  # This is a float
            logger.metric("ChromaDB_indexing_time", f"{elapsed:.2f}ms")
            return float(elapsed) 
        except Exception as e:
            logger.error(f"ChromaDB insert failed: {str(e)}")
            raise

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3
    ) -> Tuple[List[str], List[str], List[float], float]:
        timer = Timer()
        timer.start()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "distances", "metadatas"]
            )
            elapsed = timer.stop()
            
            doc_ids = results.get('ids', [[]])[0]
            texts = results.get('documents', [[]])[0]
            distances = results.get('distances', [[]])[0]
            scores = [max(0.0, 1.0 - dist) for dist in distances]
            
            return doc_ids, texts, scores, float(elapsed)
        except Exception as e:
            logger.error(f"ChromaDB search failed: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {"document_count": count, "dimension": self.dimension, "metric": "cosine"}

    def cleanup(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

    @property
    def name(self) -> str:
        return "ChromaDB"