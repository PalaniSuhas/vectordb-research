"""Streamlit UI for Embedding & Vector Database Benchmarking."""
import streamlit as st
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from benchmark.embeddings import get_embedding_model
from benchmark.vector_dbs.chroma import ChromaVectorDB
from benchmark.vector_dbs.weaviate import WeaviateVectorDB
from benchmark.retrieval import RetrievalSystem
from benchmark.evaluation import EvaluationPipeline
from benchmark.metrics import MetricsAggregator
from llms.openai_llm import OpenAILLM
from llms.gemini_llm import GeminiLLM
from utils.logger import logger

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Embedding Benchmark System",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Embedding & Vector DB Benchmark System")
st.markdown("**Canonical Embedding Strategy** - Retrieval decoupled from generation")

# Initialize session state
if 'benchmark_results' not in st.session_state:
    st.session_state.benchmark_results = []
if 'running' not in st.session_state:
    st.session_state.running = False


def load_dataset() -> Dict[str, Any]:
    """Load benchmark dataset."""
    dataset_path = project_root / "data" / "benchmark_docs.json"
    with open(dataset_path, 'r') as f:
        return json.load(f)


def run_single_benchmark(
    embedding_model_name: str,
    vector_db_name: str,
    llm_names: List[str],
    openai_key: str,
    gemini_key: str,
    dataset: Dict[str, Any]
) -> Dict[str, Any]:
    """Run a single benchmark configuration."""
    
    logger.stage(f"Benchmark: {embedding_model_name} + {vector_db_name}")
    
    # Initialize embedding model
    embedding_model = get_embedding_model(embedding_model_name, api_key=openai_key)
    logger.info(f"Embedding model: {embedding_model.name} (dim={embedding_model.get_dimension()})")
    
    # Initialize vector database
    collection_name = f"bench_{embedding_model_name.replace('-', '_')}"
    
    if vector_db_name == "ChromaDB":
        vector_db = ChromaVectorDB(
            collection_name=collection_name,
            dimension=embedding_model.get_dimension()
        )
    else:  # Weaviate
        vector_db = WeaviateVectorDB(
            collection_name=collection_name,
            dimension=embedding_model.get_dimension()
        )
    
    # Create retrieval system
    retrieval_system = RetrievalSystem(
        embedding_model=embedding_model,
        vector_db=vector_db
    )
    
    # Index documents
    logger.stage("Indexing Documents")
    indexing_metrics = retrieval_system.index_documents(dataset["documents"])
    
    # Run queries
    logger.stage("Running Queries")
    retrieval_results = []
    
    for query_data in dataset["queries"]:
        query_text = query_data["text"]
        logger.info(f"Query: {query_text}")
        
        doc_ids, texts, scores, metrics = retrieval_system.retrieve(
            query=query_text,
            top_k=3
        )
        
        retrieval_results.append({
            "query_id": query_data["id"],
            "query_text": query_text,
            "retrieved_doc_ids": doc_ids,
            "retrieved_texts": texts,
            "retrieved_scores": scores,
            "metrics": metrics
        })
    
    # Initialize LLM evaluators
    logger.stage("LLM Evaluation")
    evaluators = []
    
    if "OpenAI GPT" in llm_names and openai_key:
        evaluators.append(OpenAILLM(api_key=openai_key))
    
    if "Gemini" in llm_names and gemini_key:
        evaluators.append(GeminiLLM(api_key=gemini_key))
    
    # Run evaluation
    evaluation_pipeline = EvaluationPipeline(llm_evaluators=evaluators)
    evaluation_results = evaluation_pipeline.evaluate_all_queries(
        queries=dataset["queries"],
        retrieval_results=retrieval_results
    )
    
    # Get system info
    system_info = retrieval_system.get_system_info()
    
    # Aggregate results
    final_results = MetricsAggregator.aggregate_benchmark_results(
        embedding_model=embedding_model_name,
        vector_db=vector_db_name,
        indexing_metrics=indexing_metrics,
        retrieval_metrics=[r["metrics"] for r in retrieval_results],
        evaluation_results=evaluation_results,
        system_info=system_info
    )
    
    # Cleanup
    retrieval_system.cleanup()
    
    return final_results


def run_benchmark(
    embedding_models: List[str],
    vector_dbs: List[str],
    llm_names: List[str],
    openai_key: str,
    gemini_key: str
):
    """Run complete benchmark across all configurations."""
    
    st.session_state.running = True
    dataset = load_dataset()
    
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_configs = len(embedding_models) * len(vector_dbs)
    current_config = 0
    
    for embedding_model in embedding_models:
        for vector_db in vector_dbs:
            current_config += 1
            progress = current_config / total_configs
            progress_bar.progress(progress)
            status_text.text(f"Running: {embedding_model} + {vector_db} ({current_config}/{total_configs})")
            
            try:
                result = run_single_benchmark(
                    embedding_model_name=embedding_model,
                    vector_db_name=vector_db,
                    llm_names=llm_names,
                    openai_key=openai_key,
                    gemini_key=gemini_key,
                    dataset=dataset
                )
                all_results.append(result)
                
            except Exception as e:
                st.error(f"Error in {embedding_model} + {vector_db}: {str(e)}")
                logger.error(f"Benchmark failed: {str(e)}")
    
    # Save results
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "final_benchmark_results.json"
    
    MetricsAggregator.save_combined_results(all_results, str(output_path))
    
    st.session_state.benchmark_results = all_results
    st.session_state.running = False
    
    progress_bar.progress(1.0)
    status_text.text("✅ Benchmark Complete!")
    
    st.success(f"Results saved to: {output_path}")


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys
    st.subheader("API Keys")
    openai_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password"
    )
    gemini_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password"
    )
    
    st.divider()
    
    # Vector Database Selection
    st.subheader("Vector Database")
    vector_dbs = st.multiselect(
        "Select Vector Databases",
        ["ChromaDB", "Weaviate"],
        default=["ChromaDB"]
    )
    
    # Embedding Model Selection
    st.subheader("Embedding Models")
    embedding_models = st.multiselect(
        "Select Embedding Models",
        [
            "text-embedding-3-large",
            "bge-large-en-v1.5",
            "all-mpnet-base-v2"
        ],
        default=["all-mpnet-base-v2"]
    )
    
    # LLM Selection
    st.subheader("Evaluation LLMs")
    llm_models = st.multiselect(
        "Select LLMs for Evaluation",
        ["OpenAI GPT", "Gemini"],
        default=["OpenAI GPT"]
    )
    
    st.divider()
    
    # Run Button
    run_button = st.button(
        "🚀 Run Benchmark",
        disabled=st.session_state.running or not vector_dbs or not embedding_models or not llm_models,
        type="primary",
        use_container_width=True
    )
    
    if run_button:
        if not openai_key and ("OpenAI GPT" in llm_models or "text-embedding-3-large" in embedding_models):
            st.error("OpenAI API key required for selected models")
        elif not gemini_key and "Gemini" in llm_models:
            st.error("Gemini API key required")
        else:
            run_benchmark(
                embedding_models=embedding_models,
                vector_dbs=vector_dbs,
                llm_names=llm_models,
                openai_key=openai_key,
                gemini_key=gemini_key
            )

# Main Panel
if not st.session_state.benchmark_results:
    st.info("👈 Configure your benchmark settings in the sidebar and click 'Run Benchmark'")
    
    with st.expander("ℹ️ About this System"):
        st.markdown("""
        ### Canonical Embedding Strategy
        
        This benchmark system follows strict principles:
        
        1. **Decoupled Architecture**: Retrieval (embedding + vector search) is separate from generation (LLM)
        2. **One Vector Store per Model**: Each embedding model gets its own vector database instance
        3. **Text-Only LLM Input**: LLMs only receive retrieved text, never vectors
        4. **Quantitative Metrics**: All results are numerical scores, no subjective prose
        5. **JSON Output**: Final results are structured JSON for programmatic analysis
        
        ### Benchmark Flow
        
        For each configuration (embedding model × vector database):
        1. Embed documents using selected embedding model
        2. Index into selected vector database
        3. Run standardized queries
        4. Retrieve top-K documents
        5. Evaluate with multiple LLMs
        6. Aggregate metrics
        
        ### Metrics Collected
        
        - **Indexing**: Embedding time, indexing time
        - **Retrieval**: Query embedding time, vector search time
        - **Quality**: Context relevance, completeness, faithfulness (per LLM)
        """)

else:
    st.header("📊 Benchmark Results")
    
    # Summary Statistics
    total_runs = len(st.session_state.benchmark_results)
    st.metric("Total Configurations Tested", total_runs)
    
    # Display each result
    for idx, result in enumerate(st.session_state.benchmark_results):
        config = result.get("configuration", {})
        
        with st.expander(
            f"**{config.get('embedding_model')} + {config.get('vector_database')}**",
            expanded=(idx == 0)
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Indexing Metrics")
                indexing = result.get("indexing_metrics", {})
                st.json(indexing)
                
                st.subheader("Retrieval Metrics")
                retrieval = result.get("retrieval_metrics", {})
                st.json(retrieval)
            
            with col2:
                st.subheader("LLM Evaluation Scores")
                llm_scores = result.get("llm_evaluation_scores", {})
                st.json(llm_scores)
    
    # Download button
    results_json = json.dumps(st.session_state.benchmark_results, indent=2)
    st.download_button(
        label="📥 Download Results JSON",
        data=results_json,
        file_name="benchmark_results.json",
        mime="application/json"
    )