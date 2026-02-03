"""Google Gemini LLM implementation using the modern google-genai SDK."""
from typing import List, Dict, Any
from google import genai
from google.genai import types
import json
from utils.timer import Timer
from utils.logger import logger

class GeminiLLM:
    """Google Gemini model for evaluation and generation."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer using Gemini."""
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            return f"Error: {str(e)}"
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_texts: List[str],
        retrieved_scores: List[float]
    ) -> Dict[str, Any]:
        """Evaluate retrieval quality."""
        timer = Timer()
        timer.start()
        
        context = "\n\n".join([f"Doc {i+1}:\n{t}" for i, t in enumerate(retrieved_texts)])
        prompt = f"Evaluate RAG retrieval for Query: {query} with Context: {context}. Return JSON with context_relevance, answer_completeness, faithfulness, overall_quality."
        
        # Check for Gemini 3 or 2.5 thinking capabilities
        is_thinking_model = any(m in self.model_name for m in ["gemini-3", "gemini-2.5"])
        
        try:
            config = types.GenerateContentConfig(
                response_mime_type="application/json"
            )
            
            if is_thinking_model:
                # Gemini 3 uses thinking_level; 2.5 uses thinking_budget
                if "gemini-3" in self.model_name:
                    config.thinking_config = types.ThinkingConfig(thinking_level="high")
                else:
                    config.thinking_config = types.ThinkingConfig(thinking_budget=4096)
            else:
                config.temperature = 0.0

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            elapsed = timer.stop()
            result = json.loads(response.text)
            result["evaluation_time_ms"] = elapsed
            result["model"] = self.model_name
            return result
        except Exception as e:
            logger.error(f"Gemini evaluation failed for {self.model_name}: {str(e)}")
            return {"error": str(e), "overall_quality": 0.0, "model": self.model_name}

    @property
    def name(self) -> str:
        return f"Gemini-{self.model_name}"