"""Evaluation orchestration module."""
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
            
            for eval_data in evaluations:
                llm_eval = eval_data["evaluations"].get(llm_name, {})
                
                context_relevance_scores.append(llm_eval.get("context_relevance", 0.0))
                answer_completeness_scores.append(llm_eval.get("answer_completeness", 0.0))
                faithfulness_scores.append(llm_eval.get("faithfulness", 0.0))
                overall_quality_scores.append(llm_eval.get("overall_quality", 0.0))
                evaluation_times.append(llm_eval.get("evaluation_time_ms", 0.0))
            
            # Calculate averages
            aggregated["summary"][llm_name] = {
                "avg_context_relevance": sum(context_relevance_scores) / len(context_relevance_scores) if context_relevance_scores else 0.0,
                "avg_answer_completeness": sum(answer_completeness_scores) / len(answer_completeness_scores) if answer_completeness_scores else 0.0,
                "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
                "avg_overall_quality": sum(overall_quality_scores) / len(overall_quality_scores) if overall_quality_scores else 0.0,
                "avg_evaluation_time_ms": sum(evaluation_times) / len(evaluation_times) if evaluation_times else 0.0,
                "total_queries_evaluated": len(evaluations)
            }
        
        return aggregated