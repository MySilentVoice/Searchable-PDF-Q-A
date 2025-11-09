# 📄 PDF Document Q&A (RAG System)

Ask questions about research papers using AI. Built with **LangChain**, **ChromaDB**, **MiniLM**, and **Gemini 2.5 Flash**.

---

## 🚀 Quick Start

1. **Create & activate a virtual environment** (Python 3.11 recommended)

**Windows (PowerShell)**
```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
macOS / Linux

bash
python3.11 -m venv venv
source venv/bin/activate
Install dependencies

bash
pip install -r requirements.txt
Configure API key

Create a .env file in the project root:

GOOGLE_API_KEY=your_gemini_api_key_here
Get the key from Google AI Studio (or the provider you use).

Add PDFs
Place your PDF files in the pdf_qa_project/ folder. Example names:

paper1.pdf

paper2.pdf

Build the vector database

bash
Copy code
python ingest.py
Run the Streamlit app

bash
Copy code
streamlit run app.py
❓ Example questions
"Summarize the key contributions"

"What is Chain-of-Thought faithfulness?"

The app auto-detects phrases like "first paper" or "second paper" in the query and will try to filter accordingly.

🧰 Tech Stack
Embeddings: MiniLM (local via sentence-transformers)

Vector DB: ChromaDB (persisted locally)

LLM: Gemini 2.5 Flash (via Google Generative API)

Orchestration: LangChain-style retrieval + RAG chain

Frontend: Streamlit

✅ Features
✅ Semantic search across PDFs

✅ Shows source chunks used by the chain

✅ Query response time tracking

✅ Smart document filtering (heuristic)

✅ Local embeddings (no extra API costs for embeddings)

Author: Manish Rawat
Task: Searchable PDF Q&A