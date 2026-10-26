# 🦜️ LangChain: Models, Multi-Provider Chat & Semantic Search

A practical, hands-on repository exploring core LangChain primitives — spanning legacy LLMs, unified multi-provider Chat Models (OpenAI, Anthropic, Google Gemini, and Hugging Face), vector embeddings, and semantic document similarity search from scratch.

---

## 📌 Project Overview

This repository breaks down the core building blocks required to build modern AI and RAG (Retrieval-Augmented Generation) applications:

1. **LLMs vs. Chat Models**: Understanding the transition from raw text-completion LLMs to structured message-based chat models.
2. **Provider Agnosticism**: Writing clean, swappable code across the major AI ecosystems:
   - **OpenAI** (`gpt-4o-mini`, `gpt-4o`)
   - **Anthropic** (`claude-3-5-sonnet`, `claude-3-haiku`)
   - **Google Generative AI** (`gemini-1.5-flash`, `gemini-pro`)
   - **Hugging Face** (`TinyLlama-1.1B-Chat-v1.0` via both Cloud API and local pipeline)
3. **Embeddings & Vector Mechanics**: Generating vector representations for single queries and batch documents, working with custom dimensions, and running local sentence transformers offline.
4. **Semantic Document Similarity**: Building a vector search and ranking engine from scratch using cosine similarity with `scikit-learn` and `numpy`.

---

## 📂 Repository Structure

```plaintext
LangChain/
│
├── models/
│   ├── .env.example                     # Sample configuration for environment variables
│   │
│   ├── 1.LLM/
│   │   └── 1_llm_demo.py                # Legacy completion LLM invocation using OpenAI
│   │
│   ├── 2.ChatModels/
│   │   ├── 1_chatmodel_openai.py        # OpenAI ChatModel with temperature & token limits
│   │   ├── 2_chatmodel_antropic.py      # Anthropic Claude ChatModel integration
│   │   ├── 3_chatmodel_google.py        # Google Gemini ChatModel integration
│   │   ├── 4_chatmodel_hf_api.py        # Hugging Face ChatModel via Serverless Inference API
│   │   └── 5_chatmodel_hf_local.py      # Hugging Face ChatModel running 100% locally
│   │
│   └── 3.EmbeddedModels/
│       ├── 1_embedding_openai_query.py  # Single query embedding with OpenAI text-embedding-3
│       ├── 2_embedding_openai_docs.py   # Batch document embeddings with OpenAI
│       ├── 3_embedding_hf_local.py      # Local offline embedding via Sentence Transformers
│       └── 4_document_similarity.py     # End-to-end semantic similarity search engine
│
├── .gitignore                           # Excludes venv, cache, and sensitive .env files
├── requirements.txt                     # All project dependencies
└── README.md                            # Project documentation
```

---

## 🔍 Modules Breakdown

### 1. Base LLMs (`models/1.LLM/`)
- Demonstrates basic single-prompt completion via `langchain_openai.OpenAI`.
- Serves as the baseline comparison to appreciate why modern LLM application development has shifted toward Chat Models.

### 2. Multi-Provider Chat Models (`models/2.ChatModels/`)
Explore how LangChain standardizes the interface across distinct provider APIs:
- **OpenAI**: Configures hyper-parameters (`temperature=0.3`, `max_completion_tokens=100`) for deterministic responses.
- **Anthropic Claude**: Connects with Anthropic's reasoning models.
- **Google Gemini**: Uses `langchain_google_genai` for low-latency generative tasks.
- **Hugging Face Cloud API**: Leverages `HuggingFaceEndpoint` to interact with open-source models like `TinyLlama` without local GPU resources.
- **Hugging Face Local Pipeline**: Uses `HuggingFacePipeline` and PyTorch/Transformers to run open-weights models offline on your machine with zero API cost.

### 3. Embeddings & Semantic Search (`models/3.EmbeddedModels/`)
The foundational mechanics powering vector databases and RAG systems:
- **OpenAI Embeddings**: Working with `text-embedding-3-large`, configuring custom embedding dimensions (e.g., `dimensions=32` or `300`).
- **Batch Processing**: Converting lists of documents into high-dimensional vector representations in a single call with `embed_documents()`.
- **Local Embeddings**: Running `sentence-transformers/all-MiniLM-L6-v2` locally via `langchain_huggingface` for privacy-first, zero-cost vector generation.
- **Document Similarity Engine (`4_document_similarity.py`)**:
  - Encodes a corpus of sample documents into vector space.
  - Generates the vector embedding of a user query.
  - Computes pairwise **Cosine Similarity** using `scikit-learn`.
  - Dynamically sorts and retrieves the most relevant document alongside its confidence score.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher installed.
- API keys for the providers you plan to run (OpenAI, Google Gemini, Anthropic, or Hugging Face).

### 2. Clone the Repository & Set Up Virtual Environment

```bash
# Clone the repository (or navigate into directory)
cd LangChain

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file inside the `models/` directory (you can copy `models/.env.example`):

```bash
cp models/.env.example models/.env
```

Add your API keys:
```env
OPENAI_API_KEY="your-openai-api-key"
GOOGLE_API_KEY="your-google-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
HUGGINGFACEHUB_ACCESS_TOKEN="your-huggingface-token"
```

---

## 💻 Running the Code

### Run Base LLM Demo
```bash
python models/1.LLM/1_llm_demo.py
```

### Run Chat Models
```bash
# OpenAI
python models/2.ChatModels/1_chatmodel_openai.py

# Anthropic Claude
python models/2.ChatModels/2_chatmodel_antropic.py

# Google Gemini
python models/2.ChatModels/3_chatmodel_google.py

# Hugging Face Inference API
python models/2.ChatModels/4_chatmodel_hf_api.py

# Hugging Face Local Model (Downloads TinyLlama on first run)
python models/2.ChatModels/5_chatmodel_hf_local.py
```

### Run Embedding & Similarity Search Scripts
```bash
# Single Query Embedding
python models/3.EmbeddedModels/1_embedding_openai_query.py

# Batch Document Embeddings
python models/3.EmbeddedModels/2_embedding_openai_docs.py

# Local Hugging Face Embeddings
python models/3.EmbeddedModels/3_embedding_hf_local.py

# Document Similarity Matcher (Semantic Search)
python models/3.EmbeddedModels/4_document_similarity.py
```

---

## 🧠 Key Takeaways

- **Uniformity Across LLM Vendors**: LangChain's unified `.invoke()` API simplifies swapping models from OpenAI to Claude or Gemini with minimal refactoring.
- **Cloud vs. Local Execution**: Cloud APIs provide instant scalability, while local Hugging Face pipelines grant complete privacy and zero inference costs.
- **The Core of RAG**: Embeddings transform unstructured language into numerical vectors where semantic meaning translates to spatial proximity. Measuring that proximity with cosine similarity is the foundational step behind modern search engines and RAG pipelines.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
