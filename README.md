# Custom Q&A Chatbot with RAG 
This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on uploaded documents. The system processes PDFs/HTML files, converts them into embeddings, stores them in a vector database, and retrieves relevant context to generate responses using a large language model (LLM).

The chatbot was implemented three different ways to compare approaches:
1. OpenAI API implementation
2. Google Gemini API exploration
3. Open-source local implementation using Ollama

The application interface is built using Streamlit, allowing users to upload documents and interact with the chatbot through a web interface.

## Project Structure
```
pdf-rag-qa-chatbot/
│
├── data/
│   ├── ads_data_html/
│   │   ├── ...
│   │   └── note.svg
│   ├── Ads cookbook.pdf
│   ├── Quick_Install-Windows.html
│   ├── Quick_Install-Linux.html
│   └── Quick_Install_and_License_Setup.html
│
├── src/
│   ├── app_openai.py          # OpenAI API implementation
│   ├── app_gemini.py          # Gemini API implementation (experimental)
│   ├── app_opensource.py      # Local open-source implementation using Ollama
│   └── htmlTemplates.py       # HTML/CSS templates for the chat interface
│
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## System Architecture
The chatbot follows a Retrieval-Augmented Generation (RAG) pipeline:
1. Upload Documents (PDF or HTML)
2. Extract Text
3. Split Text into Chunks
4. Generate Embeddings
5. Store Embeddings in a FAISS Vector Database
6. User Query → Embedding
7. Retrieve Relevant Text Chunks
8. LLM Generates Context-Aware Response



## Installation 
1. Clone the Repository:
```
git clone <repo-url>
cd <repo-folder>
```
2. Create a Python Environment (Recommended)
```
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

3. Install Python Dependencies
```
pip install -r requirements.txt
```

## Running the Chatbot 
The project contains three implementations. Run the desired version with Streamlit.

### 1. OpenAI API Implementation 

**Step 1: Obtain an OpenAI API Key** 

Add to a `.env` file (in the root directory of the project):
```
OPENAI_API_KEY= # Your OpenAI API key
```

**Step 2: Run the Application**
```
cd src
streamlit run app_openai.py
```

**Models Used**
* Embedding model: `text-embedding-3-small`
* LLM: `gpt-4o-mini`

### 2. Gemini API Implementation

**Step 1: Obtain a Gemini API Key** 

Add to `.env` file (in the root directory of the project):
```
GEMINI_API_KEY=your_api_key_here
```

**Step 2: Run the Application**
```
cd src
streamlit run app_gemini.py
```

**Models Used**
* Embedding model: `gemini-embedding-2-preview`
* LLM: `gemini-2.5-flash`

### 3. Open-Source Local Implementation (Ollama)
This version runs entirely locally without external APIs.

**Step 1: Install Ollama** 

Download and install from: 

https://ollama.com

On Mac or Linux:
```
brew install ollama
```

**Step 2: Download the Model**
```
ollama pull gemma3:1b
```

**Step 3: Run the Application**
```
cd src
streamlit run app_opensource.py
```

**Models Used**
* Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
* LLM: `gemma3:1b` (via Ollama)

## Usage
1. Upload one or more PDF or HTML documents using the sidebar.
2. Click Process to create embeddings and build the vector database.
3. Enter questions in the chat input field.
4. The chatbot will retrieve relevant document sections and generate responses.

## Technologies Used
* Python
* Streamlit
* LangChain
* FAISS Vector Database
* OpenAI API
* Google Gemini API
* Ollama (Local LLM Runtime)
* SentenceTransformers
