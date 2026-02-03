"""Embedding model implementations."""
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import openai
from sentence_transformers import SentenceTransformer
from utils.timer import Timer
from utils.logger import logger


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get model name."""
        pass


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI text-embedding-3-large model."""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = "text-embedding-3-large"
        self._dimension = 3072
        self.max_batch_size = 100
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            logger.info(f"Embedding batch {i//self.max_batch_size + 1}...")
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        timer = Timer()
        timer.start()
        
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model_name
            )
            embedding = response.data[0].embedding
            
            elapsed = timer.stop()
            logger.metric(f"{self.name}_query_embedding_time", f"{elapsed:.2f}ms")
            
            return embedding
        except Exception as e:
            logger.error(f"OpenAI query embedding failed: {str(e)}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    @property
    def name(self) -> str:
        return "text-embedding-3-large"


class BGEEmbedding(EmbeddingModel):
    """BGE-Large or BGE-M3 embedding model."""
    
    def __init__(self, model_variant: str = "BAAI/bge-large-en-v1.5"):
        self.model_variant = model_variant
        self.model = SentenceTransformer(model_variant)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        timer = Timer()
        timer.start()
        
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            result = [emb.tolist() for emb in embeddings]
            
            elapsed = timer.stop()
            logger.metric(f"{self.name}_batch_embedding_time", f"{elapsed:.2f}ms")
            
            return result
        except Exception as e:
            logger.error(f"BGE embedding failed: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        timer = Timer()
        timer.start()
        
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            result = embedding.tolist()
            
            elapsed = timer.stop()
            logger.metric(f"{self.name}_query_embedding_time", f"{elapsed:.2f}ms")
            
            return result
        except Exception as e:
            logger.error(f"BGE query embedding failed: {str(e)}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    @property
    def name(self) -> str:
        return "bge-large-en-v1.5"


class MPNetEmbedding(EmbeddingModel):
    """All-MPNet-Base-v2 embedding model."""
    
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        timer = Timer()
        timer.start()
        
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            result = [emb.tolist() for emb in embeddings]
            
            elapsed = timer.stop()
            logger.metric(f"{self.name}_batch_embedding_time", f"{elapsed:.2f}ms")
            
            return result
        except Exception as e:
            logger.error(f"MPNet embedding failed: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        timer = Timer()
        timer.start()
        
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            result = embedding.tolist()
            
            elapsed = timer.stop()
            logger.metric(f"{self.name}_query_embedding_time", f"{elapsed:.2f}ms")
            
            return result
        except Exception as e:
            logger.error(f"MPNet query embedding failed: {str(e)}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    @property
    def name(self) -> str:
        return "all-mpnet-base-v2"


def get_embedding_model(model_name: str, api_key: str = None) -> EmbeddingModel:
    """Factory function to get embedding model."""
    if model_name == "text-embedding-3-large":
        if not api_key:
            raise ValueError("OpenAI API key required")
        return OpenAIEmbedding(api_key)
    elif model_name == "bge-large-en-v1.5":
        return BGEEmbedding()
    elif model_name == "all-mpnet-base-v2":
        return MPNetEmbedding()
    else:
        raise ValueError(f"Unknown embedding model: {model_name}")