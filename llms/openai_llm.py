"""OpenAI LLM implementation for evaluation and generation."""
from typing import List, Dict, Any
import openai
import json
from utils.timer import Timer
from utils.logger import logger

class OpenAILLM:
    """OpenAI GPT model for evaluation and chat."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer based on provided context."""
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer the question based only on the context provided."
        
        # GPT-5 models do not support temperature values other than 1
        is_gpt5 = "gpt-5" in self.model
        
        try:
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
            if not is_gpt5:
                params["temperature"] = 0.7
                
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
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
        
        context = "\n\n".join([
            f"Document {i+1} (score: {score:.3f}):\n{text}"
            for i, (text, score) in enumerate(zip(retrieved_texts, retrieved_scores))
        ])
        
        prompt = f"""You are an evaluation system that assesses retrieval quality.
Query: {query}
Retrieved Context:
{context}

Evaluate based on: context_relevance, answer_completeness, faithfulness (0.0 to 1.0).
Return ONLY JSON.
"""
        is_gpt5 = "gpt-5" in self.model
        
        try:
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            }
            
            # GPT-5 family logic
            if is_gpt5:
                # Temperature 0.0 is unsupported; omit it to use default 1.0
                # Use reasoning_effort for models that support it
                if "codex" not in self.model.lower():
                    params["reasoning_effort"] = "medium"
            else:
                params["temperature"] = 0.0

            response = self.client.chat.completions.create(**params)
            
            elapsed = timer.stop()
            result = json.loads(response.choices[0].message.content)
            
            result["evaluation_time_ms"] = elapsed
            result["model"] = self.model
            return result
        except Exception as e:
            logger.error(f"OpenAI evaluation failed for {self.model}: {str(e)}")
            return {"error": str(e), "overall_quality": 0.0, "model": self.model}

    @property
    def name(self) -> str:
        return f"OpenAI-{self.model}"