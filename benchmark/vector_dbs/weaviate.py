"""Weaviate vector database implementation - FIXED for v1.27.0+."""
from typing import List, Dict, Any, Tuple
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from utils.timer import Timer
from utils.logger import logger

class WeaviateVectorDB:
    def __init__(self, collection_name: str, dimension: int):
        self.collection_name = collection_name
        self.dimension = dimension
        self._use_fallback = False
        self._fallback_storage = []
        self._fallback_embeddings = []
        
        try:
            # Try to connect to embedded Weaviate with latest version
            self.client = weaviate.connect_to_embedded(
                version="1.27.0",  # Updated to 1.27.0
                environment_variables={
                    "ENABLE_MODULES": "backup-filesystem,text2vec-openai,text2vec-cohere,text2vec-huggingface,ref2vec-centroid,generative-openai,qna-openai"
                }
            )
            
            # Delete existing collection if it exists
            if self.client.collections.exists(collection_name):
                self.client.collections.delete(collection_name)
            
            # Create new collection
            self.collection = self.client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="doc_id", data_type=DataType.TEXT)
                ],
                vectorizer_config=Configure.Vectorizer.none(),
            )
            logger.info(f"Weaviate collection '{collection_name}' created (dim: {dimension})")
            
        except Exception as e:
            logger.error(f"Weaviate initialization failed: {str(e)}")
            logger.error("Falling back to in-memory numpy-based vector search")
            self.client = None
            self.collection = None
            self._use_fallback = True

    def insert_documents(self, doc_ids, embeddings, texts, metadatas) -> float:
        timer = Timer()
        timer.start()
        
        try:
            if self._use_fallback:
                # Use fallback storage
                for d_id, emb, txt, meta in zip(doc_ids, embeddings, texts, metadatas):
                    self._fallback_storage.append({"doc_id": d_id, "text": txt, "metadata": meta})
                    self._fallback_embeddings.append(emb)
                logger.info(f"Inserted {len(doc_ids)} documents into fallback storage")
            else:
                # Use Weaviate
                with self.collection.batch.dynamic() as batch:
                    for d_id, emb, txt, meta in zip(doc_ids, embeddings, texts, metadatas):
                        batch.add_object(
                            properties={"text": txt, "doc_id": d_id},
                            vector=emb
                        )
                logger.info(f"Inserted {len(doc_ids)} documents into Weaviate")
            
            elapsed = timer.stop()
            logger.metric("Weaviate_indexing_time", f"{elapsed:.2f}ms")
            return float(elapsed)
            
        except Exception as e:
            logger.error(f"Weaviate insert failed: {str(e)}")
            raise

    def search(self, query_embedding, top_k=3) -> Tuple[List[str], List[str], List[float], float]:
        timer = Timer()
        timer.start()
        
        try:
            if self._use_fallback:
                # Use numpy-based similarity search
                import numpy as np
                
                if not self._fallback_embeddings:
                    return [], [], [], float(timer.stop())
                
                q_vec = np.array(query_embedding)
                
                # Compute cosine similarities
                similarities = []
                for emb in self._fallback_embeddings:
                    e_vec = np.array(emb)
                    similarity = np.dot(q_vec, e_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(e_vec))
                    similarities.append(similarity)
                
                # Get top k indices
                top_k = min(top_k, len(similarities))
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                doc_ids = [self._fallback_storage[i]["doc_id"] for i in top_indices]
                texts = [self._fallback_storage[i]["text"] for i in top_indices]
                scores = [similarities[i] for i in top_indices]
                
            else:
                # Use Weaviate
                res = self.collection.query.near_vector(
                    near_vector=query_embedding, 
                    limit=top_k, 
                    return_metadata=MetadataQuery(distance=True)
                )
                
                doc_ids, texts, scores = [], [], []
                for obj in res.objects:
                    doc_ids.append(obj.properties.get("doc_id", ""))
                    texts.append(obj.properties.get("text", ""))
                    dist = obj.metadata.distance if obj.metadata.distance else 1.0
                    # Convert distance to similarity score
                    scores.append(1.0 / (1.0 + dist))
            
            elapsed = timer.stop()
            logger.metric("Weaviate_search_time", f"{elapsed:.2f}ms")
            return doc_ids, texts, scores, float(elapsed)
            
        except Exception as e:
            logger.error(f"Weaviate search failed: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        if self._use_fallback:
            return {
                "document_count": len(self._fallback_storage),
                "dimension": self.dimension,
                "mode": "fallback_numpy"
            }
        else:
            try:
                count = self.collection.aggregate.over_all(total_count=True)
                return {
                    "document_count": count.total_count if count.total_count else 0,
                    "dimension": self.dimension,
                    "mode": "weaviate_embedded"
                }
            except Exception:
                return {
                    "document_count": "N/A",
                    "dimension": self.dimension,
                    "mode": "weaviate_embedded"
                }

    def cleanup(self) -> None:
        if self.client:
            try:
                if self.collection and self.client.collections.exists(self.collection_name):
                    self.client.collections.delete(self.collection_name)
                self.client.close()
            except Exception as e:
                logger.error(f"Weaviate cleanup error: {str(e)}")

    @property
    def name(self) -> str:
        if self._use_fallback:
            return "Weaviate-Fallback"
        return "Weaviate"