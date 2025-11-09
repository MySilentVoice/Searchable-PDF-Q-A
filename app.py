# app.py - Optimized Streamlit RAG app (no doc selector)
# MiniLM (local embeddings) + Chroma (local) + Gemini 2.5 Flash (generation)
import os
import time
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

# LangChain-style imports (keep as in your environment)
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration ---
load_dotenv()
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_K = int(os.getenv("DEFAULT_K", "3"))
MAX_SAFE_CHARS = int(os.getenv("MAX_SAFE_CHARS", "30000"))  # crude guard on concatenated context size

st.set_page_config(page_title="PDF Q&A (RAG)", layout="wide")


# -------------------------
# Helpers: invocation + retrieval fallback
# -------------------------
def safe_invoke_chain(chain_obj, query, attempts=3):
    """Try common chain invocation methods with small backoff; return chain output."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            if hasattr(chain_obj, "invoke"):
                return chain_obj.invoke({"query": query})
            if hasattr(chain_obj, "run"):
                return chain_obj.run(query)
            return chain_obj({"query": query})
        except Exception as e:
            last_exc = e
            time.sleep(0.5 * attempt)
    raise last_exc


def retrieve_docs_with_fallback(retriever, vectorstore, query, k=3, source_filter=None):
    """
    Attempt common retriever APIs, then fallback to vectorstore similarity_search.
    If source_filter is provided (filename string), try to apply it where the backend supports it.
    Returns list of Document-like objects with .page_content and .metadata.
    """
    # Try common retriever method names first
    retriever_method_names = [
        "get_relevant_documents",
        "get_relevant_documents_with_score",
        "get_relevant_documents_and_scores",
        "get_relevant_documents_from_query",
    ]
    for name in retriever_method_names:
        method = getattr(retriever, name, None)
        if callable(method):
            try:
                return method(query)
            except TypeError:
                # method may accept (query, k)
                try:
                    return method(query, k=k)
                except Exception:
                    pass
            except Exception:
                # skip to next fallback
                pass

    # Fallback to vectorstore similarity APIs (Chroma wrappers vary)
    # Try similarity_search with optional filter kwarg
    try:
        if hasattr(vectorstore, "similarity_search"):
            try:
                if source_filter:
                    return vectorstore.similarity_search(query, k=k, filter={"source": source_filter})
            except TypeError:
                # wrapper doesn't accept filter kwarg
                pass
            return vectorstore.similarity_search(query, k=k)
    except Exception:
        pass

    # similarity_search_with_score -> convert (doc, score) -> docs
    try:
        if hasattr(vectorstore, "similarity_search_with_score"):
            try:
                if source_filter:
                    pairs = vectorstore.similarity_search_with_score(query, k=k, filter={"source": source_filter})
                else:
                    pairs = vectorstore.similarity_search_with_score(query, k=k)
                return [doc for doc, _score in pairs]
            except TypeError:
                pairs = vectorstore.similarity_search_with_score(query, k=k)
                return [doc for doc, _score in pairs]
    except Exception:
        pass

    raise RuntimeError("No supported retrieval API found on retriever/vectorstore. Check your langchain/chroma versions.")


# -------------------------
# Setup resources (cached)
# -------------------------
@st.cache_resource
def setup_rag_system(default_k=DEFAULT_K):
    """Initialize embedding model, vectorstore, retriever, LLM, and QA chain (cached once)."""
    # Ensure DB exists
    if not os.path.exists(PERSIST_DIRECTORY):
        st.error(f"Vector DB not found at {PERSIST_DIRECTORY}. Run ingest.py to create it.")
        return None

    # Embeddings (must match ingest)
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        st.error(f"Failed to initialize embeddings ({EMBEDDING_MODEL_NAME}): {e}")
        return None

    # Load Chroma vectorstore
    try:
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
    except Exception as e:
        st.error(f"Failed to load Chroma vectorstore: {e}")
        return None

    # LLM init (explicit api_key)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY missing in environment (.env). Add your Gemini API key.")
        return None

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Gemini LLM: {e}")
        return None

    # Retriever (basic; we may recreate per-query for filters)
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": default_k})
    except Exception:
        # Some wrappers differ; try best-effort
        try:
            retriever = vectorstore.as_retriever()
        except Exception:
            retriever = None

    # Build QA chain: prefer map_rerank, fallback to stuff
    try:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="map_rerank", retriever=retriever, return_source_documents=True)
    except Exception:
        try:
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        except Exception as e:
            st.error(f"Failed to create QA chain: {e}")
            return None

    return {
        "embedding_model": embedding_model,
        "vectorstore": vectorstore,
        "retriever": retriever,
        "llm": llm,
        "qa_chain": qa_chain,
    }


# -------------------------
# UI
# -------------------------
st.title("📄 PDF Document Q&A (RAG System)")
st.caption("Classic LangChain | Retrieval: MiniLM | Generation: Gemini 2.5 Flash")

state = setup_rag_system()
if state is None:
    st.stop()

vectorstore = state["vectorstore"]
base_retriever = state["retriever"]
llm = state["llm"]
qa_pipeline = state["qa_chain"]

# User input
user_query = st.text_input(
    "Ask a question about the documents:",
    placeholder="e.g., What does the first paper say about energy efficiency?"
)

if not user_query:
    st.info("Enter a question to query the documents.")
    st.stop()

# Auto-detect "first paper" / "second paper" mentions in the query (heuristic)
q_lower = user_query.lower()
auto_filter = None
if "first paper" in q_lower or "paper 1" in q_lower or "paper1" in q_lower:
    auto_filter = "paper1.pdf"
elif "second paper" in q_lower or "paper 2" in q_lower or "paper2" in q_lower:
    auto_filter = "paper2.pdf"

# Without a selector, filter_source is driven only by auto-detect; default => search all docs
filter_source = auto_filter if auto_filter is not None else None

# Perform retrieval (with fallback) to estimate context size and possibly adjust k
k_to_use = DEFAULT_K
try:
    retriever = base_retriever
    docs = retrieve_docs_with_fallback(retriever, vectorstore, user_query, k=k_to_use, source_filter=filter_source)
except Exception as e:
    st.error(f"Retrieval failed: {e}")
    st.stop()

# Estimate context size; if too large, re-run retrieval with smaller k and rebuild the chain
total_chars = sum(len(getattr(d, "page_content", "") or "") for d in docs)
if total_chars > MAX_SAFE_CHARS and len(docs) > 1:
    reduced_k = max(1, k_to_use - 1)
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": reduced_k})
        qa_pipeline = RetrievalQA.from_chain_type(llm=llm, chain_type="map_rerank", retriever=retriever, return_source_documents=True)
        docs = retrieve_docs_with_fallback(retriever, vectorstore, user_query, k=reduced_k, source_filter=filter_source)
    except Exception:
        # ignore and continue with original docs if any step fails
        pass

# Invoke chain and measure time
start = time.time()
with st.spinner("Generating answer..."):
    try:
        result = safe_invoke_chain(qa_pipeline, user_query)
        response_time = time.time() - start
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        if "quota" in str(e).lower():
            st.warning("Possible Gemini quota/billing issue. Check Google AI Studio billing.")
        st.stop()

# Normalize result
if isinstance(result, str):
    answer_text = result
    source_documents = []
else:
    answer_text = result.get("result") or result.get("answer") or result.get("output") or ""
    source_documents = result.get("source_documents") or result.get("source_docs") or []

# Display: Final answer, context used (source chunks), response time
st.markdown("---")
st.subheader("✅ Final Answer")
st.write(answer_text or "_(no answer returned)_")

st.subheader("📖 Context Used (Source Chunks)")
if source_documents:
    for i, doc in enumerate(source_documents):
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        st.markdown(f"**Chunk {i+1}** (Source: `{os.path.basename(src)}`, Page: `{page}`)")
        snippet = (doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else "")) if getattr(doc, "page_content", None) else ""
        st.code(snippet, language="text")
else:
    # Fallback: show retrieved docs used earlier
    if docs:
        st.info("Chain did not return source_documents; showing retrieved chunks used as fallback.")
        for i, d in enumerate(docs[:DEFAULT_K]):
            src = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page", "N/A")
            st.markdown(f"**Chunk {i+1}** (Source: `{os.path.basename(src)}`, Page: `{page}`)")
            snippet = (d.page_content[:400] + ("..." if len(d.page_content) > 400 else "")) if getattr(d, "page_content", None) else ""
            st.code(snippet, language="text")
    else:
        st.info("No context chunks found for this query.")

st.markdown(f"***Query Response Time: {response_time:.2f} seconds***")
