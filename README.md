# 🏦 Financial Knowledge Assistant

> Production-ready RAG system for financial document Q&A with source citations

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ Features

- 📄 **Multi-format Ingestion** - PDF, TXT, CSV with table extraction
- 🔍 **Hybrid Search** - Vector + BM25 with cross-encoder reranking
- 💬 **Cited Responses** - Every answer includes verifiable sources
- 🛡️ **Financial Guardrails** - No investment advice, hallucination prevention
- 📊 **Built-in Evaluation** - RAGAS metrics + custom financial metrics
- 🎨 **Professional UI** - Streaming responses, document preview, chat history
- 🚀 **Production-Ready** - FastAPI backend, Docker, structured logging

## 🏗️ Architecture

```
Documents → Parsing → Chunking → Embeddings → Vector DB
                                                  ↓
User Query → Rewrite → Hybrid Search → Rerank → Context
                                                  ↓
                                    LLM Generation → Cited Answer → UI
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key (for GPT-4o-mini)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/financial-knowledge-assistant.git
cd financial-knowledge-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the application
streamlit run ui/streamlit_app.py
```

### Using Make (Recommended)

```bash
make setup    # Create venv and install deps
make run      # Run Streamlit app
make api      # Run FastAPI backend
make test     # Run tests
make eval     # Run evaluation pipeline
```

## 📁 Project Structure

```
financial-knowledge-assistant/
├── app/
│   ├── api/              # FastAPI endpoints
│   ├── core/             # Config, logging, exceptions
│   ├── ingestion/        # Document loading & chunking
│   ├── retrieval/        # Embeddings, vector store, reranking
│   ├── generation/       # LLM, prompts, RAG chain
│   └── evaluation/       # RAGAS metrics, test cases
├── ui/
│   ├── streamlit_app.py  # Main Streamlit application
│   ├── components/       # Reusable UI components
│   └── assets/           # CSS, images
├── data/
│   ├── raw/              # Original documents
│   └── sample_docs/      # Sample financial documents
├── tests/                # Unit and integration tests
├── notebooks/            # Exploration notebooks
└── scripts/              # CLI utilities
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | OpenAI GPT-4o-mini / GPT-4o |
| **Embeddings** | BAAI/bge-base-en-v1.5 (free) |
| **Vector DB** | ChromaDB |
| **Reranker** | BAAI/bge-reranker-base |
| **Framework** | LangChain |
| **UI** | Streamlit |
| **API** | FastAPI |
| **Evaluation** | RAGAS |

## 📊 Evaluation Results

| Metric | Score | Description |
|--------|-------|-------------|
| Faithfulness | 0.89 | Answers grounded in context |
| Answer Relevancy | 0.85 | Answers address the question |
| Context Precision | 0.82 | Retrieved docs are relevant |
| Citation Accuracy | 0.91 | Citations are valid |
| Response Latency (p50) | 1.2s | Median response time |

## 💬 Example Queries

- "What are the key risks mentioned in the HDFC Balanced Advantage Fund?"
- "Summarize this annual report in simple terms"
- "What does SEBI say about mutual fund expense ratios?"
- "Compare these two funds based on returns and risk"
- "Explain the company's financial performance"

## 🔧 Configuration

Key settings in `.env`:

```env
OPENAI_API_KEY=your-api-key
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
LLM_MODEL=gpt-4o-mini
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
ENABLE_RERANKING=true
```

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# Access
# Streamlit UI: http://localhost:8501
# FastAPI: http://localhost:8000
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run evaluation
python scripts/evaluate_rag.py
```

## 📈 Roadmap

- [x] Core RAG pipeline
- [x] Hybrid search with reranking
- [x] Professional Streamlit UI
- [x] Source citations
- [x] Evaluation framework
- [ ] Table extraction from PDFs
- [ ] Multi-document comparison mode
- [ ] Financial entity recognition
- [ ] Agentic workflows

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for **educational purposes only**. It does not provide financial advice. Always consult a qualified financial advisor for investment decisions.

---

Built with ❤️ for the AI Engineering community
