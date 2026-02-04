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

# PDF loading function
import pdfplumber
from pathlib import Path as PathlibPath

def load_pdf_documents(pdf_path: str):
    """Load and parse PDF into document chunks."""
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"Loading PDF: {pdf_path} ({len(pdf.pages)} pages)")
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    documents.append({
                        "id": f"doc_page_{page_num}",
                        "text": text.strip(),
                        "metadata": {
                            "source": PathlibPath(pdf_path).name,
                            "page": page_num,
                            "total_pages": len(pdf.pages)
                        }
                    })
                    logger.info(f"Extracted page {page_num}: {len(text)} chars")
            logger.info(f"Total documents extracted: {len(documents)}")
    except Exception as e:
        logger.error(f"Failed to load PDF {pdf_path}: {str(e)}")
        raise
    return documents

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Embedding Benchmark System",
    layout="wide"
)

st.title("Embedding & Vector DB Benchmark System")
st.markdown("**Canonical Embedding Strategy** - Retrieval decoupled from generation")

# Initialize session state
if 'benchmark_results' not in st.session_state:
    st.session_state.benchmark_results = []
if 'running' not in st.session_state:
    st.session_state.running = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'retrieval_system' not in st.session_state:
    st.session_state.retrieval_system = None


# All available models for automatic benchmarking
ALL_EMBEDDING_MODELS = [
    "text-embedding-3-large",
    "bge-large-en-v1.5",
    "all-mpnet-base-v2"
]

ALL_VECTOR_DBS = [
    "ChromaDB",
    "Weaviate"
]

ALL_LLM_MODELS = {
    "OpenAI": [
        "gpt-5.2",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano"
    ],
    "Gemini": [
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro"
    ]
}

# Benchmark questions from the PDF
BENCHMARK_QUESTIONS = [
    "Is a perfectly rational agent always intelligent, and can an intelligent agent be irrational?",
    "In what situations do symbolic (knowledge-based) approaches outperform purely data-driven learning systems, and why?",
    "Why is probability theory considered the only coherent framework for reasoning under uncertainty, and what are its practical limitations?",
    "When should an AI agent rely more on planning and search rather than learning from data, and how can the two be effectively combined?",
    "Can the rational-agent framework adequately address ethical behavior and value alignment in advanced AI systems? If not, what is missing?"
]


def load_dataset() -> Dict[str, Any]:
    """Load benchmark dataset from PDF."""
    pdf_path = project_root / "data" / "demo pdf.pdf"
    
    if not pdf_path.exists():
        st.error(f"PDF file not found: {pdf_path}")
        return {"documents": [], "queries": []}
    
    try:
        # Extract text from PDF
        documents = load_pdf_documents(str(pdf_path))
        
        # Create queries from benchmark questions
        queries = []
        for i, question in enumerate(BENCHMARK_QUESTIONS):
            queries.append({
                "id": f"q_{i+1}",
                "text": question
            })
        
        return {
            "documents": documents,
            "queries": queries
        }
    except Exception as e:
        logger.error(f"Failed to load PDF: {str(e)}")
        st.error(f"Error loading PDF: {str(e)}")
        return {"documents": [], "queries": []}


def run_single_benchmark(
    embedding_model_name: str,
    vector_db_name: str,
    llm_models: List[str],
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
    collection_name = f"bench_{embedding_model_name.replace('-', '_').replace('.', '_')}"
    
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
            "metrics": metrics,
            "answer": ""  # Will be filled by LLM
        })
    
    # Initialize LLM evaluators
    logger.stage("LLM Evaluation")
    evaluators = []
    
    # Add all specified LLM models
    for llm_model in llm_models:
        if llm_model.startswith("gpt") and openai_key:
            evaluators.append(OpenAILLM(api_key=openai_key, model=llm_model))
        elif llm_model.startswith("gemini") and gemini_key:
            evaluators.append(GeminiLLM(api_key=gemini_key, model=llm_model))
    
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


def run_full_benchmark(openai_key: str, gemini_key: str):
    """Run complete benchmark across ALL configurations automatically."""
    
    st.session_state.running = True
    dataset = load_dataset()
    
    if not dataset["documents"]:
        st.error("No documents loaded from PDF. Cannot run benchmark.")
        st.session_state.running = False
        return
    
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Flatten all LLM models
    all_llms = []
    for provider_models in ALL_LLM_MODELS.values():
        all_llms.extend(provider_models)
    
    total_configs = len(ALL_EMBEDDING_MODELS) * len(ALL_VECTOR_DBS)
    current_config = 0
    
    for embedding_model in ALL_EMBEDDING_MODELS:
        for vector_db in ALL_VECTOR_DBS:
            current_config += 1
            progress = current_config / total_configs
            progress_bar.progress(progress)
            status_text.text(f"Running: {embedding_model} + {vector_db} ({current_config}/{total_configs})")
            
            try:
                result = run_single_benchmark(
                    embedding_model_name=embedding_model,
                    vector_db_name=vector_db,
                    llm_models=all_llms,
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
    
    # Get LLM analysis of best configuration
    analysis = generate_benchmark_analysis(all_results, openai_key, gemini_key)
    
    combined_results = {
        "benchmark_runs": all_results,
        "total_runs": len(all_results),
        "summary": MetricsAggregator._create_summary(all_results),
        "llm_analysis": analysis
    }
    
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    st.session_state.benchmark_results = combined_results
    st.session_state.running = False
    
    progress_bar.progress(1.0)
    status_text.text(" Benchmark Complete!")
    
    st.success(f"Results saved to: {output_path}")


def generate_benchmark_analysis(results: List[Dict], openai_key: str, gemini_key: str) -> str:
    """Use LLM to analyze benchmark results and recommend best configuration."""
    
    if not results:
        return "No results to analyze."
    
    # Create summary for LLM
    summary_text = "Benchmark Results Summary:\n\n"
    
    for result in results:
        config = result.get("configuration", {})
        config_str = f"{config.get('embedding_model')} + {config.get('vector_database')}"
        
        summary_text += f"\nConfiguration: {config_str}\n"
        summary_text += f"Indexing time: {result.get('indexing_metrics', {}).get('total_time_ms', 0):.2f}ms\n"
        summary_text += f"Avg retrieval time: {result.get('retrieval_metrics', {}).get('avg_total_retrieval_time_ms', 0):.2f}ms\n"
        
        llm_scores = result.get('llm_evaluation_scores', {})
        for llm_name, scores in llm_scores.items():
            summary_text += f"  {llm_name} - Overall Quality: {scores.get('avg_overall_quality', 0):.3f}\n"
    
    # Use GPT to analyze
    try:
        if openai_key:
            llm = OpenAILLM(api_key=openai_key, model="gpt-4o-mini")
            
            prompt = f"""{summary_text}

Based on these benchmark results, provide a comprehensive analysis:
1. Which configuration performed best overall and why?
2. What are the trade-offs between speed and quality?
3. Which embedding model + vector database combination would you recommend for production use?
4. Are there any surprising results or patterns?

Provide a clear, detailed analysis in 3-4 paragraphs."""
            
            import openai
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI systems analyst providing technical recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
    except Exception as e:
        logger.error(f"Analysis generation failed: {str(e)}")
        return f"Analysis generation failed: {str(e)}"
    
    return "No analysis available."


def init_test_chatbot(embedding_model_name: str, vector_db_name: str, llm_model: str, 
                      openai_key: str, gemini_key: str):
    """Initialize retrieval system for testing chatbot."""
    
    dataset = load_dataset()
    
    if not dataset["documents"]:
        st.error("No documents loaded from PDF.")
        return False
    
    try:
        # Initialize embedding model
        embedding_model = get_embedding_model(embedding_model_name, api_key=openai_key)
        
        # Initialize vector database
        collection_name = f"test_{embedding_model_name.replace('-', '_').replace('.', '_')}"
        
        if vector_db_name == "ChromaDB":
            vector_db = ChromaVectorDB(
                collection_name=collection_name,
                dimension=embedding_model.get_dimension()
            )
        else:
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
        retrieval_system.index_documents(dataset["documents"])
        
        # Store in session state
        st.session_state.retrieval_system = retrieval_system
        st.session_state.test_llm_model = llm_model
        st.session_state.openai_key = openai_key
        st.session_state.gemini_key = gemini_key
        
        return True
        
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        logger.error(f"Chatbot init failed: {str(e)}")
        return False


def chat_with_retrieval(query: str):
    """Process a chat query using the retrieval system."""
    
    if not st.session_state.retrieval_system:
        return "Retrieval system not initialized. Please click 'Initialize Test Chatbot' first."
    
    try:
        # Retrieve relevant documents
        doc_ids, texts, scores, metrics = st.session_state.retrieval_system.retrieve(
            query=query,
            top_k=3
        )
        
        # Create context from retrieved documents
        context = "\n\n".join([
            f"[Document {i+1} (relevance: {score:.3f})]\n{text}"
            for i, (text, score) in enumerate(zip(texts, scores))
        ])
        
        # Generate answer using LLM
        llm_model = st.session_state.test_llm_model
        
        if llm_model.startswith("gpt"):
            llm = OpenAILLM(api_key=st.session_state.openai_key, model=llm_model)
        else:
            llm = GeminiLLM(api_key=st.session_state.gemini_key, model=llm_model)
        
        answer = llm.generate_answer(query, context)
        
        return f"**Answer:**\n{answer}\n\n**Retrieved Context:**\n{context}"
        
    except Exception as e:
        logger.error(f"Chat query failed: {str(e)}")
        return f"Error processing query: {str(e)}"


# Sidebar Configuration
with st.sidebar:
    st.header(" Configuration")
    
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
    
    # Mode selection
    mode = st.radio(
        "Select Mode",
        ["Auto Benchmark", "Test Chatbot"],
        index=0
    )
    
    if mode == "Auto Benchmark":
        st.info("""
        **Auto Benchmark Mode**
        
        This will automatically test ALL combinations:
        - 3 Embedding Models
        - 2 Vector Databases  
        - 10 LLM Models
        
        Total configurations: 6
        Each tested with 5 queries and 10 LLM evaluators.
        """)
        
        run_button = st.button(
            "Run Full Benchmark",
            disabled=st.session_state.running or not openai_key,
            type="primary",
            use_container_width=True
        )
        
        if run_button:
            if not openai_key:
                st.error("OpenAI API key required")
            else:
                run_full_benchmark(openai_key, gemini_key)
    
    else:  # Test Chatbot mode
        st.subheader("Chatbot Configuration")
        
        test_embedding = st.selectbox(
            "Embedding Model",
            ALL_EMBEDDING_MODELS,
            index=2
        )
        
        test_vector_db = st.selectbox(
            "Vector Database",
            ALL_VECTOR_DBS,
            index=0
        )
        
        # Flatten LLM options for selectbox
        llm_options = []
        for provider, models in ALL_LLM_MODELS.items():
            llm_options.extend(models)
        
        test_llm = st.selectbox(
            "LLM Model",
            llm_options,
            index=0
        )
        
        init_button = st.button(
            "Initialize Test Chatbot",
            type="primary",
            use_container_width=True
        )
        
        if init_button:
            with st.spinner("Initializing chatbot..."):
                success = init_test_chatbot(
                    test_embedding,
                    test_vector_db,
                    test_llm,
                    openai_key,
                    gemini_key
                )
                if success:
                    st.success(" Chatbot initialized!")


# Main Panel
if mode == "Test Chatbot":
    st.header("Test Chatbot")
    
    if st.session_state.retrieval_system:
        st.success(f" Chatbot active with {st.session_state.test_llm_model}")
        
        # Chat interface
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the document..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_with_retrieval(prompt)
                st.markdown(response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    else:
        st.info(" Configure and initialize the test chatbot in the sidebar")

elif not st.session_state.benchmark_results:
    st.info(" Click 'Run Full Benchmark' in the sidebar to start automatic testing")
    
    with st.expander(" About Auto Benchmark"):
        st.markdown("""
        ### Automatic Benchmark Testing
        
        This system will automatically test:
        
        **Embedding Models:**
        - text-embedding-3-large (OpenAI)
        - bge-large-en-v1.5 (Open Source)
        - all-mpnet-base-v2 (Open Source)
        
        **Vector Databases:**
        - ChromaDB
        - Weaviate
        
        **LLM Evaluators (10 models):**
        - gpt-5.2, gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.1-codex-max
        - gemini-3-pro, gemini-3-flash, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro
        
        **Benchmark Questions:**
        - 5 deep questions about AI rationality, symbolic AI, probability theory, planning vs learning, and ethics
        
        **Results Include:**
        - Timing metrics (indexing, retrieval)
        - Quality scores from all LLMs
        - Generated answers for each question
        - LLM analysis of best configuration
        """)

else:
    st.header(" Benchmark Results")
    
    results = st.session_state.benchmark_results
    
    # Summary Statistics
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Configurations", results.get("total_runs", 0))
    
    summary = results.get("summary", {})
    
    with col2:
        best = summary.get("best_overall_quality", {})
        if best:
            st.metric("Best Quality", f"{best.get('score', 0):.3f}", 
                     delta=best.get('configuration', ''))
    
    with col3:
        fastest = summary.get("fastest_retrieval", {})
        if fastest:
            st.metric("Fastest Retrieval", f"{fastest.get('time_ms', 0):.2f}ms",
                     delta=fastest.get('configuration', ''))
    
    # LLM Analysis
    if "llm_analysis" in results:
        st.subheader(" LLM Analysis & Recommendation")
        st.markdown(results["llm_analysis"])
    
    st.divider()
    
    # Detailed Results
    st.subheader("Detailed Results")
    
    for idx, result in enumerate(results.get("benchmark_runs", [])):
        config = result.get("configuration", {})
        
        with st.expander(
            f"**{config.get('embedding_model')} + {config.get('vector_database')}**",
            expanded=(idx == 0)
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Indexing Metrics**")
                indexing = result.get("indexing_metrics", {})
                st.json(indexing)
                
                st.markdown("**Retrieval Metrics**")
                retrieval = result.get("retrieval_metrics", {})
                st.json(retrieval)
            
            with col2:
                st.markdown("**LLM Evaluation Scores**")
                llm_scores = result.get("llm_evaluation_scores", {})
                st.json(llm_scores)
            
            # Show detailed query results
            st.markdown("**Query Results with Answers**")
            detailed = result.get("detailed_query_results", [])
            for query_result in detailed:
                with st.expander(f"Query: {query_result.get('query', '')}"):
                    st.markdown(f"**Retrieved Documents:**")
                    retrieval_data = query_result.get('retrieval_metrics', {})
                    st.json(retrieval_data)
                    
                    st.markdown(f"**LLM Evaluations:**")
                    evals = query_result.get('evaluations', {})
                    st.json(evals)
    
    # Download button
    results_json = json.dumps(results, indent=2)
    st.download_button(
        label=" Download Full Results JSON",
        data=results_json,
        file_name="benchmark_results.json",
        mime="application/json"
    )