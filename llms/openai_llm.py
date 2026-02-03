"""OpenAI LLM implementation for evaluation."""
from typing import List, Dict, Any
import openai
import json
from utils.timer import Timer
from utils.logger import logger


class OpenAILLM:
    """OpenAI GPT model for evaluation."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict evaluation system. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            elapsed = timer.stop()
            
            # Parse response
            result_text = response.usage.completion_tokens if response.usage else 0
            result = json.loads(response.choices[0].message.content)
            
            # Validate scores
            for key in ["context_relevance", "answer_completeness", "faithfulness", "overall_quality"]:
                if key not in result:
                    result[key] = 0.0
                else:
                    result[key] = max(0.0, min(1.0, float(result[key])))
            
            # Add metadata
            result["evaluation_time_ms"] = elapsed
            result["model"] = self.model
            
            logger.metric(f"OpenAI_evaluation_time", f"{elapsed:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI evaluation failed: {str(e)}")
            # Return default scores on error
            return {
                "context_relevance": 0.0,
                "answer_completeness": 0.0,
                "faithfulness": 0.0,
                "overall_quality": 0.0,
                "evaluation_time_ms": timer.elapsed,
                "model": self.model,
                "error": str(e)
            }
    
    @property
    def name(self) -> str:
        return f"OpenAI-{self.model}"