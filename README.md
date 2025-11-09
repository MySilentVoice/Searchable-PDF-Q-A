# 📄 PDF Document Q&A (RAG System)

A Retrieval-Augmented Generation (RAG) application built with **LangChain**, **Chroma**, **MiniLM embeddings**, and **Google Gemini 2.5 Flash**.  
It allows you to ask natural language questions about research papers (PDFs), retrieves relevant chunks using local embeddings, and generates concise answers using Gemini.

---

## 🚀 Features

- 🧠 **Retrieval-Augmented Generation (RAG)** pipeline  
- 🗂️ Automatic **PDF ingestion**, splitting, and embedding storage using **Chroma**  
- 🔍 **MiniLM** (Sentence Transformers) for efficient, free local embeddings  
- 💬 **Gemini 2.5 Flash** for high-quality answer generation  
- ⚡ **Streamlit** UI for simple interaction  
- 💾 Local persistent **vector database** (`./chroma_db`)  
- 🔐 Environment-based API configuration via `.env` file  
- 🧩 Modular and fully local (no external dependencies beyond Gemini API)

---

## 🧰 Tech Stack

| Component | Description |
|------------|--------------|
| **LangChain (modular)** | Framework for chaining retrieval + LLMs |
| **Sentence-Transformers (MiniLM)** | Local embedding model for text similarity |
| **ChromaDB** | Lightweight vector database for storing embeddings |
| **Google Gemini 2.5 Flash** | LLM used to generate final answers |
| **Streamlit** | Frontend web app for user queries |
| **Python-dotenv** | Loads environment variables securely |

---

## 📂 Project Structure

Python Project/
├── app.py # Streamlit app (retrieval + generation)
├── ingest.py # Creates vector DB from PDFs
├── utils.py (optional) # Helper functions (if used)
├── requirements.txt # Python dependencies
├── .env # Environment variables (Gemini API key, etc.)
├── chroma_db/ # Persistent Chroma vector store
├── pdf_qa_project/ # Folder containing paper1.pdf, paper2.pdf
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Setup & Installation

### 1️⃣ Clone or download the project

```bash
git clone <your_repo_url>
cd PDF-QA
2️⃣ Create and activate a virtual environment
bash
Copy code
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Add your environment variables
Create a .env file in the project root:

bash
Copy code
GOOGLE_API_KEY=your_google_api_key_here
PERSIST_DIRECTORY=./chroma_db
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
You can get your Gemini API key from Google AI Studio → API Keys.

5️⃣ Add your PDF files
Place your PDFs inside the pdf_qa_project/ folder and rename them:

bash
Copy code
pdf_qa_project/paper1.pdf
pdf_qa_project/paper2.pdf
🧩 Step 1: Build the Vector Database
Run the ingestion script once to process and embed the PDFs.

bash
Copy code
python ingest.py
✅ This script:

Reads both PDFs

Splits text into chunks

Embeds using MiniLM

Saves vectors to chroma_db/

💬 Step 2: Run the Streamlit Q&A App
bash
Copy code
streamlit run app.py
Once it starts, open the local URL (usually http://localhost:8501).
You can now type questions like:

"Summarize the key contributions of the first paper."
"What methods are compared in the second paper?"
"How do the authors evaluate performance improvements?"

🧠 How It Works (Under the Hood)
PDF Loading & Chunking (ingest.py)

Loads PDFs using PyPDFLoader

Splits text into overlapping chunks (1000 characters, 200 overlap)

Embeddings

Converts chunks into dense vectors using MiniLM (sentence-transformers/all-MiniLM-L6-v2)

Vector Store (Chroma)

Stores and indexes embeddings for semantic search

Retrieval

Retrieves top-K chunks relevant to your question

Generation (Gemini)

Passes context + question to Gemini 2.5 Flash for answer synthesis

Streamlit Frontend

Displays the final answer, source chunks, and response time

🧾 Example Output
vbnet
Copy code
Question: Summarize the key contributions.

✅ Final Answer:
The key contributions include Arushi Somani conducting RL ablation experiments on CoT scoring with preference models...
📖 Context Used (Source Chunks):
Chunk 1: paper2.pdf, page 12
Chunk 2: paper2.pdf, page 12
...
🧰 Troubleshooting
Issue	Fix
ModuleNotFoundError (langchain_*)	Run pip install -r requirements.txt again
GOOGLE_API_KEY missing	Check .env spelling and restart Streamlit
Failed to load Chroma vectorstore	Run python ingest.py first
Quota exceeded / API limit	Check Google AI Studio billing / quota page
Torch install fails on Windows	Try: pip install torch --index-url https://download.pytorch.org/whl/cpu

🧑‍💻 Author
Manish Rawat
AI/ML Enthusiast — Building intelligent retrieval systems with LangChain and open-source embeddings.

🏁 License
This project is open-source and free to use for educational and research purposes.