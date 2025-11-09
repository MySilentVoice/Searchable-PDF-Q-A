📄 PDF Document Q&A (RAG System)
A Retrieval-Augmented Generation (RAG) application that lets you ask questions about research papers. Built with LangChain, ChromaDB, MiniLM embeddings, and Google Gemini 2.5 Flash.

🚀 Features

🧠 RAG Pipeline - Semantic search + AI generation
📚 Multi-document support - Query across multiple PDFs
🔍 Smart filtering - Auto-detects "first paper" / "second paper" mentions
⚡ Local embeddings - Free MiniLM model (no API costs)
💾 Persistent storage - ChromaDB vector database
🎯 Context display - See exact chunks used for answers
⏱️ Performance metrics - Query response time tracking


🛠️ Tech Stack
ComponentPurposeLangChainRAG orchestration frameworkMiniLMLocal text embeddings (sentence-transformers)ChromaDBVector database for semantic searchGemini 2.5 FlashLLM for answer generationStreamlitWeb interfacePyPDFPDF text extraction

📁 Project Structure
PYTHON PROJECT/
├── app.py                    # Streamlit Q&A interface
├── ingest.py                # Vector DB creation script
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── .env                     # Environment variables (API keys)
├── .gitignore              # Git ignore rules
├── chroma_db/              # Generated vector database
├── pdf_qa_project/         # PDF documents folder
│   ├── paper1.pdf          # First research paper
│   └── paper2.pdf          # Second research paper
└── venv/                   # Virtual environment (after setup)

⚙️ Installation
1️⃣ Create virtual environment
bashpython -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
2️⃣ Install dependencies
bashpip install -r requirements.txt
Note: On Windows, if torch fails to install:
bashpip install torch --index-url https://download.pytorch.org/whl/cpu
3️⃣ Set up environment variables
Create a .env file in the project root:
envGOOGLE_API_KEY=your_gemini_api_key_here
PERSIST_DIRECTORY=./chroma_db
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DEFAULT_K=3
MAX_SAFE_CHARS=30000
Get your Gemini API key: Google AI Studio
4️⃣ Add your PDF files
Download the required papers:

Paper 1 → Save as paper1.pdf
Paper 2 → Save as paper2.pdf

Place both PDFs in the pdf_qa_project/ folder (create it if it doesn't exist):
bashmkdir pdf_qa_project
# Then move your downloaded PDFs into this folder

🎯 Usage
Step 1: Build the Vector Database
Run once to process and embed the PDFs:
bashpython ingest.py
This will:

Load both PDF documents
Split text into 1000-character chunks (200 overlap)
Generate embeddings using MiniLM
Store vectors in ./chroma_db/

Expected output:
Loading documents...
Splitting 50 pages into chunks...
Total chunks created: 245
Creating and persisting vector store...
✅ Vector store created successfully
Step 2: Launch the Q&A Interface
bashstreamlit run app.py
Open your browser to http://localhost:8501
Step 3: Ask Questions
Select document:

"All" - Search across both papers
"paper1.pdf" - Query only the first paper
"paper2.pdf" - Query only the second paper

Auto-detection: The system automatically detects phrases like:

"What does the first paper say about..."
"According to the second paper..."

Example queries:

"Summarize the key contributions"
"What is Chain-of-Thought faithfulness?"
"Compare the methodologies used in both papers"
"What datasets were used in the first paper?"


🧠 How It Works
Architecture
User Query
    ↓
[Document Selector + Auto-Detection]
    ↓
[Retriever] → Finds top-K relevant chunks (MiniLM embeddings)
    ↓
[Context Assembly] → Combines retrieved chunks
    ↓
[Gemini 2.5 Flash] → Generates answer from context
    ↓
[Display] → Shows answer + source chunks + timing
Key Components

Text Chunking - RecursiveCharacterTextSplitter breaks PDFs into overlapping segments
Embeddings - MiniLM converts text to 384-dimensional vectors
Vector Search - ChromaDB finds semantically similar chunks
LLM Generation - Gemini synthesizes final answer from context
Source Attribution - Displays exact chunks and page numbers used


🔧 Error Handling Features
The code includes robust error handling:

✅ Multiple retrieval fallbacks - Tries various LangChain APIs
✅ Context size management - Auto-reduces k if context too large
✅ Chain invocation flexibility - Supports invoke/run/call methods
✅ Quota detection - Alerts for Gemini API limits
✅ Missing file checks - Clear messages for setup issues


📊 Sample Output
✅ Final Answer:
Chain-of-Thought faithfulness evaluates whether the reasoning steps 
accurately reflect the model's internal decision-making process...

📖 Context Used (Source Chunks)
Chunk 1 (Source: paper2.pdf, Page: 9)
"Prior research has proposed metrics to evaluate various aspects..."

Query Response Time: 2.34 seconds

🐛 Troubleshooting
IssueSolutionModuleNotFoundErrorRun pip install -r requirements.txtGOOGLE_API_KEY missingCheck .env file spelling and restartFailed to load ChromaRun python ingest.py firstQuota exceededCheck Google AI Studio billingTorch install fails (Windows)Use CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpuImport errors with langchainRemove old langchain-classic if present
Common Fix: Clean Install
bashpip uninstall langchain langchain-community langchain-core -y
pip install -r requirements.txt --force-reinstall

🎓 Project Context
Task: Fresher Interview - Task 1 (Mandatory)
Goal: Build a searchable PDF Q&A system demonstrating:

Document processing and chunking
Embedding generation and vector storage
Semantic retrieval
LLM-based answer generation
Clean UI with source attribution


📝 Configuration Options
Edit .env to customize:
env# Increase for more context (slower, more comprehensive)
DEFAULT_K=5

# Reduce if hitting context limits
MAX_SAFE_CHARS=20000

# Switch embedding model (not recommended unless needed)
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

📄 License
Educational/Research use. Built for technical assessment purposes.
Author: Manish Rawat
Date: November 2024