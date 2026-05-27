# 🧠 Multimodal RAG System

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> A production-ready **Multimodal Retrieval-Augmented Generation (RAG)** system that supports **PDFs, plain text, Markdown, and images** — enabling natural language Q&A over heterogeneous document collections.

---

## 📌 Overview

Traditional RAG systems are limited to text. This system extends the paradigm to support **multiple modalities**:

| Modality | Handling |
|---|---|
| 📄 PDF | Per-page text extraction via PyPDF2 |
| 📝 Text / Markdown | Paragraph-level chunking |
| 🖼️ Images | Automatic captioning via BLIP (Salesforce) |

All modalities are embedded into the **same vector space** using `sentence-transformers`, indexed with **FAISS** for sub-millisecond retrieval, and answered using your choice of LLM backend.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                       │
│   PDF → text chunks │ TXT → paragraphs │ IMG → captions │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   EMBEDDING LAYER                        │
│          sentence-transformers (all-MiniLM-L6-v2)       │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   VECTOR STORE (FAISS)                   │
│      IndexFlatIP with L2-normalized cosine similarity    │
└───────────────────────────┬─────────────────────────────┘
                            │  top-k retrieval
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  GENERATION LAYER                        │
│   OpenAI GPT-3.5 │ Flan-T5 (local) │ Extractive         │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- 🔀 **Multimodal ingestion** — PDFs, text, and images in one pipeline
- 🔍 **Dense retrieval** — FAISS with cosine similarity, sub-10ms search
- 🤖 **Flexible generation** — OpenAI, local HuggingFace model, or no-dependency extractive
- 📊 **Built-in evaluation** — Faithfulness, Answer Relevance, Context Precision, MRR
- 💾 **Persistent index** — save/load vector stores to disk
- 🖥️ **Streamlit UI** — clean interactive demo app
- 🧪 **Jupyter notebook** — step-by-step walkthrough

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/NoorAlamJan/multimodal-rag.git
cd multimodal-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

### Python API

```python
from src.rag_pipeline import MultimodalRAGPipeline

# Initialize
pipeline = MultimodalRAGPipeline(
    embedding_model   = "all-MiniLM-L6-v2",
    generator_backend = "openai",   # or "huggingface" or "extractive"
    top_k             = 5,
)

# Ingest a directory of documents
pipeline.ingest("./data/")

# Query
response = pipeline.query("What is the main conclusion of the paper?")

print(response.answer)
for source in response.sources:
    print(f"  [{source.rank}] {source.document.source} (score: {source.score:.3f})")
```

---

## 📁 Project Structure

```
multimodal-rag/
│
├── src/
│   ├── rag_pipeline.py     # Core pipeline: ingestion, vector store, generation
│   └── evaluation.py       # RAG evaluation metrics
│
├── notebooks/
│   └── demo.ipynb          # End-to-end walkthrough notebook
│
├── data/                   # Drop your documents here
├── app.py                  # Streamlit demo application
├── requirements.txt
└── README.md
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **Faithfulness** | Are answer claims supported by retrieved context? |
| **Answer Relevance** | Does the answer address the question? |
| **Context Precision** | Fraction of retrieved chunks that are relevant |
| **MRR** | Mean Reciprocal Rank of first relevant result |

```python
from src.evaluation import RAGEvaluator

evaluator = RAGEvaluator()
report = evaluator.evaluate(
    query=response.query,
    answer=response.answer,
    context=response.context,
    retrieved_docs=[r.document.content for r in response.sources],
    scores=[r.score for r in response.sources],
)
evaluator.print_report(report)
```

---

## ⚙️ LLM Backends

| Backend | Setup | Quality |
|---|---|---|
| `extractive` | No API key needed | Basic — keyword overlap |
| `huggingface` | `pip install transformers torch` | Good — Flan-T5 local |
| `openai` | `OPENAI_API_KEY` env var | Best — GPT-3.5/4 |

---

## 🔧 Configuration

| Parameter | Default | Description |
|---|---|---|
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `generator_backend` | `auto` | LLM backend |
| `top_k` | `5` | Number of chunks to retrieve |
| `chunk_size` | `512` | Words per chunk |
| `chunk_overlap` | `64` | Overlap between chunks |

---

## 📚 References

- [Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Salesforce BLIP — Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)
- [FAISS — A Library for Efficient Similarity Search](https://faiss.ai/)
- [RAGAS — Evaluation Framework for RAG](https://docs.ragas.io/)

---

## 👤 Author

**Noor Alam**
- GitHub: [@NoorAlamJan](https://github.com/NoorAlamJan)
- LinkedIn: [noor-alam-0a7122209](https://linkedin.com/in/noor-alam-0a7122209)

---

## 📄 License

MIT License — free to use, modify, and distribute.
