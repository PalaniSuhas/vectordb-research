"""Weaviate vector database implementation."""
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
        try:
            self.client = weaviate.connect_to_embedded(version="1.24.4")
            if self.client.collections.exists(collection_name):
                self.client.collections.delete(collection_name)
            
            self.collection = self.client.collections.create(
                name=collection_name,
                properties=[Property(name="text", data_type=DataType.TEXT),
                            Property(name="doc_id", data_type=DataType.TEXT)],
                vectorizer_config=Configure.Vectorizer.none(),
            )
        except Exception as e:
            logger.error(f"Weaviate initialization failed: {str(e)}")
            self.client = None
            self.collection = None
            self._fallback_storage = []
            self._fallback_embeddings = []

    def insert_documents(self, doc_ids, embeddings, texts, metadatas) -> float:
        timer = Timer()
        timer.start()
        try:
            if self.collection:
                with self.collection.batch.dynamic() as batch:
                    for d_id, emb, txt, meta in zip(doc_ids, embeddings, texts, metadatas):
                        batch.add_object(properties={"text": txt, "doc_id": d_id}, vector=emb)
            else:
                for d_id, emb, txt, meta in zip(doc_ids, embeddings, texts, metadatas):
                    self._fallback_storage.append({"doc_id": d_id, "text": txt})
                    self._fallback_embeddings.append(emb)
            
            elapsed = timer.stop()
            return float(elapsed)
        except Exception as e:
            logger.error(f"Weaviate insert failed: {str(e)}")
            raise

    def search(self, query_embedding, top_k=3) -> Tuple[List[str], List[str], List[float], float]:
        timer = Timer()
        timer.start()
        try:
            if self.collection:
                res = self.collection.query.near_vector(
                    near_vector=query_embedding, limit=top_k, 
                    return_metadata=MetadataQuery(distance=True)
                )
                doc_ids, texts, scores = [], [], []
                for obj in res.objects:
                    doc_ids.append(obj.properties.get("doc_id", ""))
                    texts.append(obj.properties.get("text", ""))
                    dist = obj.metadata.distance if obj.metadata.distance else 1.0
                    scores.append(1.0 / (1.0 + dist))
            else:
                import numpy as np
                q_vec = np.array(query_embedding)
                sims = [np.dot(q_vec, e)/(np.linalg.norm(q_vec)*np.linalg.norm(e)) for e in self._fallback_embeddings]
                top_idx = np.argsort(sims)[-top_k:][::-1]
                doc_ids = [self._fallback_storage[i]["doc_id"] for i in top_idx]
                texts = [self._fallback_storage[i]["text"] for i in top_idx]
                scores = [sims[i] for i in top_idx]
            
            elapsed = timer.stop()
            return doc_ids, texts, scores, float(elapsed)
        except Exception as e:
            logger.error(f"Weaviate search failed: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        return {"document_count": len(self._fallback_storage) if not self.collection else "N/A"}

    def cleanup(self) -> None:
        if self.client: self.client.close()

    @property
    def name(self) -> str:
        return "Weaviate"