"""Google Gemini LLM implementation using the modern google-genai SDK."""
from typing import List, Dict, Any
from google import genai
from google.genai import types
import json
from utils.timer import Timer
from utils.logger import logger

class GeminiLLM:
    """Google Gemini model for evaluation using the new google-genai SDK."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        # Initializing the new Client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_texts: List[str],
        retrieved_scores: List[float]
    ) -> Dict[str, Any]:
        """Evaluate retrieval quality."""
        timer = Timer()
        timer.start()
        
        context = "\n\n".join([
            f"Document {i+1} (score: {score:.3f}):\n{text}"
            for i, (text, score) in enumerate(zip(retrieved_texts, retrieved_scores))
        ])
        
        prompt = f"""You are an evaluation system that assesses retrieval quality.
Query: {query}
Retrieved Context: {context}

Evaluate based on:
1. context_relevance (0.0-1.0)
2. answer_completeness (0.0-1.0)
3. faithfulness (0.0-1.0)

Return ONLY JSON:
{{
  "context_relevance": <score>,
  "answer_completeness": <score>,
  "faithfulness": <score>,
  "overall_quality": <average>
}}"""
        
        try:
            # Using the new generate_content API with response_mime_type
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            
            elapsed = timer.stop()
            result = json.loads(response.text)
            
            # Ensure all keys exist
            for key in ["context_relevance", "answer_completeness", "faithfulness"]:
                result[key] = max(0.0, min(1.0, float(result.get(key, 0.0))))
            
            if "overall_quality" not in result:
                result["overall_quality"] = (result["context_relevance"] + result["answer_completeness"] + result["faithfulness"]) / 3
            
            result["evaluation_time_ms"] = elapsed
            result["model"] = self.model_name
            
            logger.metric(f"Gemini_evaluation_time", f"{elapsed:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Gemini evaluation failed: {str(e)}")
            return {
                "context_relevance": 0.0,
                "answer_completeness": 0.0,
                "faithfulness": 0.0,
                "overall_quality": 0.0,
                "evaluation_time_ms": timer.stop() if timer.start_time else 0,
                "model": self.model_name,
                "error": str(e)
            }

    @property
    def name(self) -> str:
        return f"Gemini-{self.model_name}"