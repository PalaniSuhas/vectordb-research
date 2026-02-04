"""Google Gemini LLM implementation using the modern google-genai SDK - FIXED."""
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
        prompt = f"""Evaluate RAG retrieval quality.

Query: {query}

Retrieved Context:
{context}

Provide a JSON evaluation with these numeric scores (0.0 to 1.0):
- context_relevance: How relevant is the retrieved context to the query?
- answer_completeness: How complete would an answer be based on this context?
- faithfulness: How well does the context support answering the query?
- overall_quality: Overall retrieval quality score

Return ONLY a valid JSON object, no other text."""
        
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
            
            # Handle response - for thinking models, response.text contains the final output
            # The thinking process is in response.candidates[0].content.parts but we only need the final text
            response_text = response.text
            
            # Clean up the response text - remove markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Ensure all required fields exist with defaults
            result.setdefault("context_relevance", 0.0)
            result.setdefault("answer_completeness", 0.0)
            result.setdefault("faithfulness", 0.0)
            result.setdefault("overall_quality", 0.0)
            
            result["evaluation_time_ms"] = float(elapsed)
            result["model"] = self.model_name
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Gemini JSON parse error for {self.model_name}: {str(e)}")
            logger.error(f"Response text was: {response_text if 'response_text' in locals() else 'N/A'}")
            return {
                "error": f"JSON parse error: {str(e)}",
                "context_relevance": 0.0,
                "answer_completeness": 0.0,
                "faithfulness": 0.0,
                "overall_quality": 0.0,
                "model": self.model_name,
                "evaluation_time_ms": float(timer.stop()) if timer.start_time else 0.0
            }
        except Exception as e:
            logger.error(f"Gemini evaluation failed for {self.model_name}: {str(e)}")
            return {
                "error": str(e),
                "context_relevance": 0.0,
                "answer_completeness": 0.0,
                "faithfulness": 0.0,
                "overall_quality": 0.0,
                "model": self.model_name,
                "evaluation_time_ms": 0.0
            }

    @property
    def name(self) -> str:
        return f"Gemini-{self.model_name}"