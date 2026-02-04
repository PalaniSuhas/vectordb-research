"""Evaluation orchestration module - COMPREHENSIVE FIX."""
from typing import List, Dict, Any, Union
from llms.openai_llm import OpenAILLM
from llms.gemini_llm import GeminiLLM
from utils.logger import logger

LLMEvaluator = Union[OpenAILLM, GeminiLLM]


class EvaluationPipeline:
    """Orchestrates LLM-based evaluation of retrieval results."""
    
    def __init__(self, llm_evaluators: List[LLMEvaluator]):
        self.llm_evaluators = llm_evaluators
    
    def evaluate_query(
        self,
        query: str,
        retrieved_texts: List[str],
        retrieved_scores: List[float]
    ) -> Dict[str, Any]:
        """Evaluate retrieval for a single query using all LLMs.
        
        Returns:
            Dictionary with evaluation results from all LLMs
        """
        results = {}
        
        for evaluator in self.llm_evaluators:
            logger.info(f"Evaluating with {evaluator.name}")
            
            evaluation = evaluator.evaluate_retrieval(
                query=query,
                retrieved_texts=retrieved_texts,
                retrieved_scores=retrieved_scores
            )
            
            results[evaluator.name] = evaluation
        
        return results
    
    def evaluate_all_queries(
        self,
        queries: List[Dict[str, Any]],
        retrieval_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate all queries.
        
        Args:
            queries: List of query dictionaries
            retrieval_results: List of retrieval result dictionaries
        
        Returns:
            Aggregated evaluation results
        """
        all_evaluations = []
        
        for query_data, retrieval_data in zip(queries, retrieval_results):
            query_text = query_data["text"]
            retrieved_texts = retrieval_data["retrieved_texts"]
            retrieved_scores = retrieval_data["retrieved_scores"]
            
            logger.stage(f"Evaluating query: {query_text}")
            
            evaluation = self.evaluate_query(
                query=query_text,
                retrieved_texts=retrieved_texts,
                retrieved_scores=retrieved_scores
            )
            
            all_evaluations.append({
                "query": query_text,
                "query_id": query_data["id"],
                "evaluations": evaluation,
                "retrieval_metrics": retrieval_data["metrics"]
            })
        
        # Aggregate results
        return self._aggregate_evaluations(all_evaluations)
    
    def _aggregate_evaluations(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate evaluation results across all queries.
        
        FIXED: Properly handles error dictionaries from failed LLM evaluations
        
        Returns:
            Aggregated metrics
        """
        # Initialize aggregation structure
        aggregated = {
            "query_evaluations": evaluations,
            "summary": {}
        }
        
        # Aggregate by LLM
        for evaluator in self.llm_evaluators:
            llm_name = evaluator.name
            
            # Collect all scores for this LLM
            context_relevance_scores = []
            answer_completeness_scores = []
            faithfulness_scores = []
            overall_quality_scores = []
            evaluation_times = []
            successful_evaluations = 0
            
            for eval_data in evaluations:
                llm_eval = eval_data["evaluations"].get(llm_name, {})
                
                if "error" in llm_eval:
                    logger.info(f"Skipping failed evaluation for {llm_name}: {llm_eval.get('error')}")
                    continue
                
                # Helper to handle cases where LLM returns {"score": 0.8} instead of 0.8
                def safe_float(key):
                    val = llm_eval.get(key, 0.0)
                    if isinstance(val, dict):
                        # Extract 'score' or 'value', defaulting to 0.0 if not found
                        return float(val.get("score", val.get("value", 0.0)))
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return 0.0

                # Use the helper for all metric fields
                context_relevance_scores.append(safe_float("context_relevance"))
                answer_completeness_scores.append(safe_float("answer_completeness"))
                faithfulness_scores.append(safe_float("faithfulness"))
                overall_quality_scores.append(safe_float("overall_quality"))
                
                evaluation_times.append(safe_float("evaluation_time_ms"))
                successful_evaluations += 1
            
            # Calculate averages only if we have successful evaluations
            if successful_evaluations > 0:
                aggregated["summary"][llm_name] = {
                    "avg_context_relevance": sum(context_relevance_scores) / successful_evaluations,
                    "avg_answer_completeness": sum(answer_completeness_scores) / successful_evaluations,
                    "avg_faithfulness": sum(faithfulness_scores) / successful_evaluations,
                    "avg_overall_quality": sum(overall_quality_scores) / successful_evaluations,
                    "avg_evaluation_time_ms": sum(evaluation_times) / successful_evaluations,
                    "total_queries_evaluated": successful_evaluations,
                    "failed_evaluations": len(evaluations) - successful_evaluations
                }
            else:
                # All evaluations failed for this LLM
                logger.error(f"All evaluations failed for {llm_name}")
                aggregated["summary"][llm_name] = {
                    "avg_context_relevance": 0.0,
                    "avg_answer_completeness": 0.0,
                    "avg_faithfulness": 0.0,
                    "avg_overall_quality": 0.0,
                    "avg_evaluation_time_ms": 0.0,
                    "total_queries_evaluated": 0,
                    "failed_evaluations": len(evaluations),
                    "error": "All evaluations failed"
                }
        
        return aggregated