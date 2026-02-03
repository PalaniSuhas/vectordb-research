"""Weaviate vector database implementation."""
from typing import List, Dict, Any, Tuple
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from utils.timer import Timer
from utils.logger import logger


class WeaviateVectorDB:
    """Weaviate implementation for vector storage and retrieval."""
    
    def __init__(self, collection_name: str, dimension: int):
        """Initialize Weaviate client (embedded mode)."""
        self.collection_name = collection_name
        self.dimension = dimension
        
        try:
            # Use embedded Weaviate
            self.client = weaviate.connect_to_embedded(
                version="1.24.4",
                environment_variables={
                    "ENABLE_MODULES": "text2vec-transformers",
                    "TRANSFORMERS_INFERENCE_API": "http://t2v-transformers:8080"
                },
                headers={}
            )
            
            # Delete collection if exists
            if self.client.collections.exists(collection_name):
                self.client.collections.delete(collection_name)
            
            # Create collection with vector configuration
            self.collection = self.client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT),
                    Property(name="topic", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),  # We provide vectors
            )
            
            logger.info(f"Weaviate collection '{collection_name}' created (dimension={dimension})")
            
        except Exception as e:
            logger.error(f"Weaviate initialization failed: {str(e)}")
            # Fallback: use in-memory simple storage
            self.client = None
            self.collection = None
            self._fallback_storage = []
            self._fallback_embeddings = []
            logger.info("Using fallback in-memory storage")
    
    def insert_documents(
        self,
        doc_ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> float:
        """Insert documents into Weaviate.
        
        Returns:
            Indexing time in milliseconds
        """
        timer = Timer()
        timer.start()
        
        try:
            if self.collection is not None:
                # Weaviate mode
                with self.collection.batch.dynamic() as batch:
                    for i, (doc_id, embedding, text, metadata) in enumerate(
                        zip(doc_ids, embeddings, texts, metadatas)
                    ):
                        batch.add_object(
                            properties={
                                "text": text,
                                "doc_id": doc_id,
                                "category": metadata.get("category", ""),
                                "topic": metadata.get("topic", "")
                            },
                            vector=embedding
                        )
            else:
                # Fallback mode
                for doc_id, embedding, text, metadata in zip(doc_ids, embeddings, texts, metadatas):
                    self._fallback_storage.append({
                        "doc_id": doc_id,
                        "text": text,
                        "metadata": metadata
                    })
                    self._fallback_embeddings.append(embedding)
            
            elapsed = timer.stop()
            logger.metric(f"Weaviate_indexing_time", f"{elapsed:.2f}ms")
            
            return elapsed
            
        except Exception as e:
            logger.error(f"Weaviate insert failed: {str(e)}")
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
            if self.collection is not None:
                # Weaviate mode
                response = self.collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=top_k,
                    return_metadata=MetadataQuery(distance=True)
                )
                
                doc_ids = []
                texts = []
                scores = []
                
                for obj in response.objects:
                    doc_ids.append(obj.properties.get("doc_id", ""))
                    texts.append(obj.properties.get("text", ""))
                    # Convert distance to similarity (lower distance = higher similarity)
                    distance = obj.metadata.distance if obj.metadata.distance else 1.0
                    scores.append(1.0 / (1.0 + distance))
                
            else:
                # Fallback mode: compute cosine similarity
                import numpy as np
                
                query_vec = np.array(query_embedding)
                similarities = []
                
                for emb in self._fallback_embeddings:
                    doc_vec = np.array(emb)
                    similarity = np.dot(query_vec, doc_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
                    )
                    similarities.append(similarity)
                
                # Get top-k indices
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                doc_ids = [self._fallback_storage[i]["doc_id"] for i in top_indices]
                texts = [self._fallback_storage[i]["text"] for i in top_indices]
                scores = [similarities[i] for i in top_indices]
            
            elapsed = timer.stop()
            logger.metric(f"Weaviate_search_latency", f"{elapsed:.2f}ms")
            
            return doc_ids, texts, scores, elapsed
            
        except Exception as e:
            logger.error(f"Weaviate search failed: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if self.collection is not None:
            count = len(list(self.collection.iterator()))
        else:
            count = len(self._fallback_storage)
        
        return {
            "document_count": count,
            "dimension": self.dimension,
            "collection_name": self.collection_name
        }
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.client is not None:
                if self.client.collections.exists(self.collection_name):
                    self.client.collections.delete(self.collection_name)
                self.client.close()
                logger.info(f"Weaviate collection '{self.collection_name}' deleted")
        except:
            pass
    
    @property
    def name(self) -> str:
        return "Weaviate"