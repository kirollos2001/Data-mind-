import os
import faiss
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from functools import lru_cache
import hashlib

# Load environment variables
load_dotenv()

# -----------------------------
# 1) Configuration & Setup
# -----------------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", # Faster than 2.5-pro for most tasks
    google_api_key=api_key,
    temperature=0
)

# Paths - use absolute paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "rag_output")
FAISS_INDEX_PATH = os.path.join(RAG_OUTPUT_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(RAG_OUTPUT_DIR, "chunks.txt")

# Performance Settings
ENABLE_ROUTER = False  # Set to True to enable router (adds ~1.5s latency)
ENABLE_GRADING = False  # Set to True to enable LLM grading (adds ~1.5s latency)
DISTANCE_THRESHOLD = 0.7  # For score-based filtering (lower = more strict)
MAX_RETRIES = 1
TOP_K = 5

# Load FAISS Index and Chunks
print("Loading FAISS index and chunks...")
try:
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    chunks = []
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
            raw_chunks = content.split("--- Chunk ")
            for rc in raw_chunks:
                if not rc.strip(): continue
                try:
                    newline_idx = rc.find("\n")
                    if newline_idx != -1:
                        chunk_text = rc[newline_idx+1:].strip()
                        chunks.append(chunk_text)
                except:
                    pass
    print(f"✅ Loaded {index.ntotal} vectors and {len(chunks)} chunks.")
except Exception as e:
    print(f"❌ Error loading RAG data: {e}")
    index = None
    chunks = []

# Embedding cache
@lru_cache(maxsize=100)
def get_query_embedding_cached(query: str) -> np.ndarray:
    """Cache query embeddings to avoid redundant API calls."""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        return np.array([result['embedding']])
    except Exception as e:
        print(f"⚠️ Error embedding query: {e}")
        return None

# -----------------------------
# 2) State Definition
# -----------------------------
class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    doc_scores: List[float]  # Added for score tracking
    retry_count: int

# -----------------------------
# 3) Nodes (OPTIMIZED)
# -----------------------------

def router_node(state: AgentState):
    """
    OPTIONAL: Decides whether to use RAG or not.
    DISABLED by default for speed (saves ~1.5s).
    """
    if not ENABLE_ROUTER:
        return {"generation": None, "documents": []}
    
    print("--- ROUTER ---")
    question = state["question"]
    
    system_prompt = (
        "You are an expert at routing user questions. "
        "Return 'USE_RAG' if the question is about documents/reports, or 'NO_RAG' for greetings/simple questions."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    decision = response.content.strip().upper()
    
    if "USE_RAG" in decision:
        return {"generation": None, "documents": []}
    else:
        return {"generation": "NO_RAG_NEEDED"}

def retrieve_node(state: AgentState):
    """
    OPTIMIZED: Retrieves documents from FAISS with score-based filtering.
    Speed: ~0.1s (very fast!)
    """
    print("--- RETRIEVE ---")
    question = state["question"]
    
    if index is None or len(chunks) == 0:
        print("⚠️ Index not loaded.")
        return {"documents": [], "doc_scores": []}

    # Get cached embedding (fast if repeated query)
    query_embedding = get_query_embedding_cached(question)
    
    if query_embedding is None:
        return {"documents": [], "doc_scores": []}
    
    try:
        # FAISS search (extremely fast)
        k = TOP_K
        D, I = index.search(query_embedding, k)
        
        # Filter by distance threshold (replaces slow LLM grading)
        retrieved_docs = []
        scores = []
        
        for distance, idx in zip(D[0], I[0]):
            if idx < len(chunks):
                # Lower distance = better match
                # Normalize to similarity score (inverse of distance)
                similarity = 1 / (1 + distance)
                
                if similarity >= DISTANCE_THRESHOLD:
                    retrieved_docs.append(chunks[idx])
                    scores.append(similarity)
        
        print(f"✅ Retrieved {len(retrieved_docs)} docs (threshold: {DISTANCE_THRESHOLD})")
        return {"documents": retrieved_docs, "doc_scores": scores}
        
    except Exception as e:
        print(f"❌ Error in retrieval: {e}")
        return {"documents": [], "doc_scores": []}

def grade_documents_node(state: AgentState):
    """
    OPTIMIZED: Uses score-based filtering instead of LLM.
    DISABLED by default for speed (saves ~1.5s).
    """
    if not ENABLE_GRADING:
        # Skip LLM grading, rely on score threshold
        documents = state["documents"]
        if documents:
            print(f"--- SKIP GRADING (using scores) ---")
            return {"documents": documents}
        else:
            print(f"--- NO DOCS AFTER FILTERING ---")
            current_retry = state.get("retry_count", 0)
            return {"documents": [], "retry_count": current_retry + 1}
    
    # Original LLM-based grading (slow)
    print("--- GRADE DOCUMENTS (LLM) ---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {"documents": [], "retry_count": state.get("retry_count", 0)}

    context_str = "\n\n".join(documents[:3])
    
    system_prompt = (
        "You are a grader. If the documents are relevant to the question, respond 'yes', else 'no'."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {question}\n\nDocuments:\n{context_str}")
    ]
    
    response = llm.invoke(messages)
    grade = response.content.strip().lower()
    
    if "yes" in grade:
        print("--- DECISION: RELEVANT ---")
        return {"documents": documents}
    else:
        print("--- DECISION: NOT RELEVANT ---")
        current_retry = state.get("retry_count", 0)
        return {"documents": [], "retry_count": current_retry + 1}

def rewrite_query_node(state: AgentState):
    """
    Rewrites the query to improve retrieval.
    """
    print("--- REWRITE QUERY ---")
    question = state["question"]
    
    system_prompt = (
        "Rewrite this question to be more specific and optimized for document retrieval. "
        "Keep it concise."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    better_question = response.content.strip()
    print(f"📝 Rewritten: {better_question}")
    
    # Clear cache for new query
    get_query_embedding_cached.cache_clear()
    
    return {"question": better_question}

def generate_node(state: AgentState):
    """
    Generates the final answer.
    """
    print("--- GENERATE ---")
    question = state["question"]
    documents = state["documents"]
    generation_status = state.get("generation")
    
    if generation_status == "NO_RAG_NEEDED":
        messages = [HumanMessage(content=question)]
        response = llm.invoke(messages)
        return {"generation": response.content}
    
    # RAG answer
    if not documents:
        # No documents found
        return {"generation": "I don't have enough information in the documents to answer this question."}
    
    context = "\n\n".join(documents)
    system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer, say so. "
        "Be concise and accurate."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {question}\n\nContext:\n{context}")
    ]
    
    response = llm.invoke(messages)
    return {"generation": response.content}

# -----------------------------
# 4) Graph Construction
# -----------------------------

def route_decision(state: AgentState):
    if ENABLE_ROUTER and state.get("generation") == "NO_RAG_NEEDED":
        return "generate_direct"
    return "retrieve"

def grade_decision(state: AgentState):
    if state["documents"]:
        return "generate"
    
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "generate_giveup"
    
    return "rewrite"

workflow = StateGraph(AgentState)

# Add nodes
if ENABLE_ROUTER:
    workflow.add_node("router", router_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate", generate_node)

# Add edges
if ENABLE_ROUTER:
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "generate_direct": "generate",
            "retrieve": "retrieve"
        }
    )
else:
    workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    grade_decision,
    {
        "generate": "generate",
        "rewrite": "rewrite_query",
        "generate_giveup": "generate"
    }
)

workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# -----------------------------
# 5) Main Execution
# -----------------------------
if __name__ == "__main__":
    import sys
    import time
    
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = "What is the revenue growth in the report?"
        
    print(f"\n🤖 User Question: {user_input}\n")
    print(f"⚙️  Config: Router={ENABLE_ROUTER}, Grading={ENABLE_GRADING}, Threshold={DISTANCE_THRESHOLD}\n")
    
    inputs = {"question": user_input, "retry_count": 0}
    
    try:
        start_time = time.time()
        
        # Run agent
        final_state = app.invoke(inputs)
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("💡 Final Answer:")
        print(final_state["generation"])
        print("="*60)
        print(f"⏱️  Total time: {elapsed:.2f}s")
        
        if final_state.get("documents"):
            print(f"📚 Retrieved {len(final_state['documents'])} relevant documents")
        
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        import traceback
        traceback.print_exc()
