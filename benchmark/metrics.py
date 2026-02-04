"""Metrics aggregation and result formatting - FIXED."""
from typing import Dict, Any, List
import json
from pathlib import Path
from utils.logger import logger


class MetricsAggregator:
    """Aggregates and formats benchmark metrics."""
    
    @staticmethod
    def _ensure_float(val: Any) -> float:
        """Ensure value is a float."""
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            cleaned = val.replace("ms", "").strip()
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                return 0.0
        return 0.0
    
    @staticmethod
    def aggregate_benchmark_results(
        embedding_model: str,
        vector_db: str,
        indexing_metrics: Dict[str, Any],
        retrieval_metrics: List[Dict[str, Any]],
        evaluation_results: Dict[str, Any],
        system_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate all metrics into final result structure.
        
        Returns:
            Complete benchmark results
        """
        # Sanitize indexing metrics
        clean_indexing = {
            "embedding_time_ms": MetricsAggregator._ensure_float(indexing_metrics.get("embedding_time_ms", 0)),
            "indexing_time_ms": MetricsAggregator._ensure_float(indexing_metrics.get("indexing_time_ms", 0)),
            "total_time_ms": MetricsAggregator._ensure_float(indexing_metrics.get("total_time_ms", 0)),
            "document_count": int(indexing_metrics.get("document_count", 0))
        }
        
        # Calculate average retrieval metrics
        avg_retrieval = MetricsAggregator._average_retrieval_metrics(retrieval_metrics)
        
        # Build result structure
        result = {
            "configuration": {
                "embedding_model": embedding_model,
                "vector_database": vector_db,
                "embedding_dimension": system_info.get("embedding_dimension", 0)
            },
            "indexing_metrics": clean_indexing,
            "retrieval_metrics": avg_retrieval,
            "llm_evaluation_scores": evaluation_results.get("summary", {}),
            "detailed_query_results": evaluation_results.get("query_evaluations", []),
            "system_info": system_info
        }
        
        return result
    
    @staticmethod
    def _average_retrieval_metrics(
        retrieval_metrics: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate average retrieval metrics across all queries.
        
        Returns:
            Dictionary with averaged metrics
        """
        if not retrieval_metrics:
            return {
                "avg_query_embedding_time_ms": 0.0,
                "avg_vector_search_time_ms": 0.0,
                "avg_total_retrieval_time_ms": 0.0,
                "total_queries": 0
            }
        
        # Collect all metrics with type sanitization
        query_embed_times = []
        vector_search_times = []
        total_retrieval_times = []
        
        for metrics in retrieval_metrics:
            query_embed_times.append(
                MetricsAggregator._ensure_float(metrics.get("query_embedding_time_ms", 0))
            )
            vector_search_times.append(
                MetricsAggregator._ensure_float(metrics.get("vector_search_time_ms", 0))
            )
            total_retrieval_times.append(
                MetricsAggregator._ensure_float(metrics.get("total_retrieval_time_ms", 0))
            )
        
        return {
            "avg_query_embedding_time_ms": sum(query_embed_times) / len(query_embed_times),
            "avg_vector_search_time_ms": sum(vector_search_times) / len(vector_search_times),
            "avg_total_retrieval_time_ms": sum(total_retrieval_times) / len(total_retrieval_times),
            "total_queries": len(retrieval_metrics)
        }
    
    @staticmethod
    def save_results(
        results: Dict[str, Any],
        output_path: str
    ) -> None:
        """Save results to JSON file.
        
        Args:
            results: Results dictionary
            output_path: Path to output file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    @staticmethod
    def save_combined_results(
        all_results: List[Dict[str, Any]],
        output_path: str
    ) -> None:
        """Save combined results from multiple benchmark runs.
        
        Args:
            all_results: List of result dictionaries
            output_path: Path to output file
        """
        combined = {
            "benchmark_runs": all_results,
            "total_runs": len(all_results),
            "summary": MetricsAggregator._create_summary(all_results)
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(combined, f, indent=2)
        
        logger.info(f"Combined results saved to {output_path}")
    
    @staticmethod
    def _create_summary(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics across all benchmark runs.
        
        Returns:
            Summary dictionary
        """
        summary = {
            "configurations_tested": [],
            "best_overall_quality": None,
            "fastest_retrieval": None,
            "fastest_indexing": None
        }
        
        best_quality_score = -1.0
        fastest_retrieval_time = float('inf')
        fastest_indexing_time = float('inf')
        
        for result in all_results:
            config = result.get("configuration", {})
            config_str = f"{config.get('embedding_model')} + {config.get('vector_database')}"
            summary["configurations_tested"].append(config_str)
            
            # Find best quality
            llm_scores = result.get("llm_evaluation_scores", {})
            for llm_name, scores in llm_scores.items():
                quality = MetricsAggregator._ensure_float(scores.get("avg_overall_quality", 0))
                if quality > best_quality_score:
                    best_quality_score = quality
                    summary["best_overall_quality"] = {
                        "configuration": config_str,
                        "llm": llm_name,
                        "score": quality
                    }
            
            # Find fastest retrieval
            retrieval = result.get("retrieval_metrics", {})
            avg_time = MetricsAggregator._ensure_float(
                retrieval.get("avg_total_retrieval_time_ms", float('inf'))
            )
            if avg_time < fastest_retrieval_time:
                fastest_retrieval_time = avg_time
                summary["fastest_retrieval"] = {
                    "configuration": config_str,
                    "time_ms": avg_time
                }
            
            # Find fastest indexing
            indexing = result.get("indexing_metrics", {})
            total_time = MetricsAggregator._ensure_float(
                indexing.get("total_time_ms", float('inf'))
            )
            if total_time < fastest_indexing_time:
                fastest_indexing_time = total_time
                summary["fastest_indexing"] = {
                    "configuration": config_str,
                    "time_ms": total_time
                }
        
        return summary