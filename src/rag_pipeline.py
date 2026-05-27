"""
Multimodal RAG Pipeline
=======================
Supports: PDF text, images (captioning), and plain text documents.
Uses:     sentence-transformers for embeddings, FAISS for vector store,
          HuggingFace / OpenAI for generation.
"""

from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A single chunk of content (text or image caption)."""
    doc_id:    str
    content:   str
    modality:  str          # "text" | "image" | "table"
    source:    str          # file path / URL
    page:      Optional[int] = None
    metadata:  Dict         = field(default_factory=dict)

    def __repr__(self):
        return f"Document(id={self.doc_id[:8]}, modality={self.modality}, source={Path(self.source).name})"


@dataclass
class RetrievalResult:
    document:  Document
    score:     float
    rank:      int


@dataclass
class RAGResponse:
    query:     str
    answer:    str
    sources:   List[RetrievalResult]
    context:   str


# ─────────────────────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────────────────────

class DocumentIngester:
    """
    Loads and chunks documents from multiple modalities.

    Supported formats:
        .pdf   — extracts text per page
        .txt   — plain text, chunked by paragraph
        .md    — Markdown text
        .png / .jpg / .jpeg — image captioning via BLIP or stub
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Text helpers ──────────────────────────────────────────

    def _chunk_text(self, text: str, source: str, page: Optional[int] = None) -> List[Document]:
        """Split text into overlapping chunks."""
        words   = text.split()
        chunks  = []
        start   = 0

        while start < len(words):
            end   = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            doc_id = hashlib.md5(f"{source}{start}".encode()).hexdigest()
            chunks.append(Document(
                doc_id   = doc_id,
                content  = chunk,
                modality = "text",
                source   = source,
                page     = page,
            ))
            start += self.chunk_size - self.chunk_overlap

        return chunks

    # ── PDF ──────────────────────────────────────────────────

    def load_pdf(self, path: str) -> List[Document]:
        try:
            import PyPDF2
            docs = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.extend(self._chunk_text(text, path, page=i + 1))
            print(f"  [PDF]  Loaded {len(docs)} chunks from {Path(path).name}")
            return docs
        except ImportError:
            print("  [WARN] PyPDF2 not installed — skipping PDF ingestion.")
            return []

    # ── Plain text / Markdown ────────────────────────────────

    def load_text(self, path: str) -> List[Document]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs = self._chunk_text(text, path)
        print(f"  [TXT]  Loaded {len(docs)} chunks from {Path(path).name}")
        return docs

    # ── Images ───────────────────────────────────────────────

    def load_image(self, path: str) -> List[Document]:
        """
        Generates a caption for an image using BLIP (if available),
        or falls back to a filename-based stub for environments
        without GPU / transformers installed.
        """
        caption = self._caption_image(path)
        doc_id  = hashlib.md5(path.encode()).hexdigest()
        doc     = Document(
            doc_id   = doc_id,
            content  = caption,
            modality = "image",
            source   = path,
            metadata = {"original_path": path},
        )
        print(f"  [IMG]  Captioned: {Path(path).name} → \"{caption[:60]}...\"")
        return [doc]

    def _caption_image(self, path: str) -> str:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from PIL import Image
            import torch

            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            image     = Image.open(path).convert("RGB")
            inputs    = processor(image, return_tensors="pt")

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)
            return processor.decode(out[0], skip_special_tokens=True)

        except Exception:
            # Graceful stub — useful when running without GPU
            name = Path(path).stem.replace("_", " ").replace("-", " ")
            return f"Image depicting: {name}. (Install transformers + torch for real captioning.)"

    # ── Directory loader ─────────────────────────────────────

    def load_directory(self, directory: str) -> List[Document]:
        """Load all supported files from a directory recursively."""
        all_docs = []
        ext_map  = {
            ".pdf": self.load_pdf,
            ".txt": self.load_text,
            ".md":  self.load_text,
            ".png": self.load_image,
            ".jpg": self.load_image,
            ".jpeg":self.load_image,
        }
        for path in Path(directory).rglob("*"):
            if path.suffix.lower() in ext_map:
                all_docs.extend(ext_map[path.suffix.lower()](str(path)))

        print(f"\n  Total documents ingested: {len(all_docs)}")
        return all_docs


# ─────────────────────────────────────────────────────────────
# Vector Store
# ─────────────────────────────────────────────────────────────

class VectorStore:
    """
    FAISS-backed dense vector store with sentence-transformer embeddings.

    Falls back to a simple cosine-similarity NumPy store if FAISS
    is not installed (useful for CI / lightweight environments).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self._encoder  = None
        self._index    = None

    # ── Encoder ──────────────────────────────────────────────

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
                print(f"  [EMBED] Loaded encoder: {self.model_name}")
            except ImportError:
                raise ImportError("Run: pip install sentence-transformers")
        return self._encoder

    def _encode(self, texts: List[str]) -> np.ndarray:
        encoder = self._get_encoder()
        return encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # ── Index ─────────────────────────────────────────────────

    def build_index(self, documents: List[Document]) -> None:
        """Embed all documents and build FAISS index."""
        self.documents = documents
        texts          = [d.content for d in documents]

        print(f"\n  [INDEX] Embedding {len(texts)} chunks...")
        self.embeddings = self._encode(texts).astype("float32")

        try:
            import faiss
            dim          = self.embeddings.shape[1]
            self._index  = faiss.IndexFlatIP(dim)          # Inner-product (cosine after normalise)
            faiss.normalize_L2(self.embeddings)
            self._index.add(self.embeddings)
            print(f"  [INDEX] FAISS index built ({dim}d, {len(documents)} vectors)")
        except ImportError:
            print("  [WARN] FAISS not installed — using NumPy fallback (slower).")

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Return top-k most relevant documents for a query."""
        q_emb = self._encode([query]).astype("float32")

        if self._index is not None:
            import faiss
            faiss.normalize_L2(q_emb)
            scores, indices = self._index.search(q_emb, top_k)
            results = [
                RetrievalResult(document=self.documents[idx], score=float(scores[0][i]), rank=i + 1)
                for i, idx in enumerate(indices[0]) if idx < len(self.documents)
            ]
        else:
            # NumPy cosine similarity fallback
            norm_q = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            norm_e = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9)
            sims   = (norm_e @ norm_q.T).flatten()
            top_idx = np.argsort(sims)[::-1][:top_k]
            results = [
                RetrievalResult(document=self.documents[i], score=float(sims[i]), rank=rank + 1)
                for rank, i in enumerate(top_idx)
            ]

        return results

    # ── Persistence ───────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        np.save(f"{path}/embeddings.npy", self.embeddings)
        with open(f"{path}/documents.json", "w") as f:
            json.dump([vars(d) for d in self.documents], f, indent=2)
        print(f"  [SAVE] Vector store saved to {path}/")

    def load(self, path: str) -> None:
        self.embeddings = np.load(f"{path}/embeddings.npy").astype("float32")
        with open(f"{path}/documents.json") as f:
            self.documents = [Document(**d) for d in json.load(f)]
        try:
            import faiss
            dim         = self.embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(self.embeddings)
            self._index.add(self.embeddings)
        except ImportError:
            pass
        print(f"  [LOAD] Loaded {len(self.documents)} documents from {path}/")


# ─────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────

class Generator:
    """
    LLM-based answer generator.

    Supports:
        - OpenAI GPT (requires OPENAI_API_KEY env var)
        - HuggingFace local model (flan-t5-base by default — CPU-friendly)
        - Simple extractive fallback (no dependencies)
    """

    def __init__(self, backend: str = "auto"):
        """
        backend: "openai" | "huggingface" | "extractive" | "auto"
        """
        self.backend = backend
        self._pipe   = None

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            "You are a helpful AI assistant. Answer the user's question "
            "using ONLY the context provided below. If the answer is not "
            "in the context, say 'I cannot find this information in the provided documents.'\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )

    def generate(self, query: str, context: str) -> str:
        backend = self.backend

        if backend == "auto":
            if os.getenv("OPENAI_API_KEY"):
                backend = "openai"
            else:
                backend = "extractive"   # safe default

        if backend == "openai":
            return self._openai(query, context)
        elif backend == "huggingface":
            return self._huggingface(query, context)
        else:
            return self._extractive(query, context)

    def _openai(self, query: str, context: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI()
            resp   = client.chat.completions.create(
                model    = "gpt-3.5-turbo",
                messages = [
                    {"role": "system", "content": "You are a helpful document Q&A assistant."},
                    {"role": "user",   "content": self._build_prompt(query, context)},
                ],
                max_tokens  = 512,
                temperature = 0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI Error] {e}"

    def _huggingface(self, query: str, context: str) -> str:
        try:
            from transformers import pipeline
            if self._pipe is None:
                self._pipe = pipeline("text2text-generation", model="google/flan-t5-base")
            prompt = self._build_prompt(query, context)[:1024]  # flan-t5 token limit
            result = self._pipe(prompt, max_new_tokens=256, do_sample=False)
            return result[0]["generated_text"].strip()
        except Exception as e:
            return f"[HuggingFace Error] {e}"

    def _extractive(self, query: str, context: str) -> str:
        """Simple keyword-overlap extractive baseline (no dependencies)."""
        query_words = set(query.lower().split())
        sentences   = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]
        scored      = [
            (sum(w in s.lower() for w in query_words), s)
            for s in sentences
        ]
        scored.sort(reverse=True)
        top = [s for _, s in scored[:3] if _]
        if top:
            return ". ".join(top) + "."
        return "I cannot find a direct answer in the retrieved documents."


# ─────────────────────────────────────────────────────────────
# RAG Pipeline (orchestrator)
# ─────────────────────────────────────────────────────────────

class MultimodalRAGPipeline:
    """
    End-to-end Multimodal RAG pipeline.

    Usage
    -----
    >>> pipeline = MultimodalRAGPipeline()
    >>> pipeline.ingest("./data/")
    >>> response = pipeline.query("What is the main topic of the document?")
    >>> print(response.answer)
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        generator_backend: str = "auto",
        top_k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.top_k     = top_k
        self.ingester  = DocumentIngester(chunk_size, chunk_overlap)
        self.store     = VectorStore(embedding_model)
        self.generator = Generator(generator_backend)
        self._built    = False

    def ingest(self, path: str) -> None:
        """Ingest a file or directory."""
        print(f"\n{'='*50}\n  INGESTION — {path}\n{'='*50}")
        if Path(path).is_dir():
            docs = self.ingester.load_directory(path)
        elif Path(path).suffix.lower() == ".pdf":
            docs = self.ingester.load_pdf(path)
        elif Path(path).suffix.lower() in (".png", ".jpg", ".jpeg"):
            docs = self.ingester.load_image(path)
        else:
            docs = self.ingester.load_text(path)

        self.store.build_index(docs)
        self._built = True

    def query(self, question: str) -> RAGResponse:
        """Run a query against the ingested documents."""
        if not self._built:
            raise RuntimeError("Call .ingest() before .query()")

        print(f"\n  [QUERY] {question}")

        # 1. Retrieve
        results = self.store.search(question, top_k=self.top_k)

        # 2. Build context (interleave modalities)
        context_parts = []
        for r in results:
            prefix = f"[{r.document.modality.upper()} | {Path(r.document.source).name}]"
            context_parts.append(f"{prefix}\n{r.document.content}")
        context = "\n\n---\n\n".join(context_parts)

        # 3. Generate
        answer = self.generator.generate(question, context)

        return RAGResponse(
            query   = question,
            answer  = answer,
            sources = results,
            context = context,
        )

    def save(self, path: str = "./rag_store") -> None:
        self.store.save(path)

    def load(self, path: str = "./rag_store") -> None:
        self.store.load(path)
        self._built = True

    def __repr__(self):
        n = len(self.store.documents)
        return f"MultimodalRAGPipeline(docs={n}, top_k={self.top_k}, generator={self.generator.backend})"
