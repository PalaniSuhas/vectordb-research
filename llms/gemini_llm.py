"""Google Gemini LLM implementation for evaluation."""
from typing import List, Dict, Any
import google.generativeai as genai
import json
from utils.timer import Timer
from utils.logger import logger


class GeminiLLM:
    """Google Gemini model for evaluation."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_texts: List[str],
        retrieved_scores: List[float]
    ) -> Dict[str, Any]:
        """Evaluate retrieval quality.
        
        Returns:
            Dictionary with evaluation scores
        """
        timer = Timer()
        timer.start()
        
        # Construct evaluation prompt
        context = "\n\n".join([
            f"Document {i+1} (score: {score:.3f}):\n{text}"
            for i, (text, score) in enumerate(zip(retrieved_texts, retrieved_scores))
        ])
        
        prompt = f"""You are an evaluation system that assesses retrieval quality.

Query: {query}

Retrieved Context:
{context}

Evaluate the retrieval based on these metrics:

1. context_relevance: How relevant are the retrieved documents to the query? (0.0 to 1.0)
2. answer_completeness: Do the documents contain enough information to answer the query? (0.0 to 1.0)
3. faithfulness: Are the documents factually accurate and consistent? (0.0 to 1.0)

CRITICAL INSTRUCTIONS:
- Base your scores ONLY on the retrieved context provided
- Do NOT invent facts or make assumptions
- Return ONLY valid JSON with no additional text
- Use decimal numbers between 0.0 and 1.0

Return ONLY this JSON structure:
{{
  "context_relevance": <score>,
  "answer_completeness": <score>,
  "faithfulness": <score>,
  "overall_quality": <average of above three scores>
}}"""
        
        try:
            generation_config = {
                "temperature": 0.0,
                "response_mime_type": "application/json"
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            elapsed = timer.stop()
            
            # Parse response
            result = json.loads(response.text)
            
            # Validate scores
            for key in ["context_relevance", "answer_completeness", "faithfulness", "overall_quality"]:
                if key not in result:
                    result[key] = 0.0
                else:
                    result[key] = max(0.0, min(1.0, float(result[key])))
            
            # Add metadata
            result["evaluation_time_ms"] = elapsed
            result["model"] = self.model_name
            
            logger.metric(f"Gemini_evaluation_time", f"{elapsed:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini evaluation failed: {str(e)}")
            # Return default scores on error
            return {
                "context_relevance": 0.0,
                "answer_completeness": 0.0,
                "faithfulness": 0.0,
                "overall_quality": 0.0,
                "evaluation_time_ms": timer.elapsed,
                "model": self.model_name,
                "error": str(e)
            }
    
    @property
    def name(self) -> str:
        return f"Gemini-{self.model_name}"