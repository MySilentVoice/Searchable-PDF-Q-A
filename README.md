📄 PDF Document Q&A (RAG System)
Ask questions about research papers using AI. Built with LangChain, ChromaDB, MiniLM, and Gemini 2.5 Flash.

🚀 Quick Start
1. Install Dependencies
bash python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
2. Configure API Key
Create .env file:
envGOOGLE_API_KEY=your_gemini_api_key_here
Get your key: Google AI Studio
3. Add PDFs
Download and place in pdf_qa_project/ folder:

Paper 1 → paper1.pdf
Paper 2 → paper2.pdf

4. Build Vector Database
bashpython ingest.py
5. Run the App
bash streamlit run app.py

📁 Project Structure
PYTHON PROJECT/
├── app.py              # Streamlit interface
├── ingest.py          # Vector DB creation
├── requirements.txt   # Dependencies
├── .env              # API keys
├── chroma_db/        # Vector database (generated)
├── pdf_qa_project/   # Your PDFs go here
└── venv/             # Virtual environment

Example questions:

"Summarize the key contributions"
"What is Chain-of-Thought faithfulness?"

The app auto-detects mentions like "first paper" or "second paper" in your query.

🛠️ Tech Stack
ComponentPurposeMiniLMFree local embeddingsChromaDBVector databaseGemini 2.5 FlashAnswer generationLangChainRAG orchestrationStreamlitWeb interface

📊 Features

✅ Semantic search across PDFs
✅ Shows source chunks used
✅ Query response time tracking
✅ Smart document filtering
✅ Local embeddings (no extra API costs)


Author: Manish Rawat | Task: Searchable PDF Q&A