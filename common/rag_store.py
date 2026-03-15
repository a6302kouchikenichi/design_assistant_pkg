from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pdfplumber
from docx import Document
import pandas as pd

from .llm_client import ChatMessage, LLMClient


@dataclass
class DocChunk:
    doc_id: str
    source: str
    text: str


def simple_chunk(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = (text or '').strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks


def extract_text_from_file(file_path: Path) -> str:
    """Extract text from various file formats (md, pdf, docx, xlsx)."""
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == '.md':
            return file_path.read_text(encoding='utf-8', errors='ignore')
        
        elif suffix == '.pdf':
            text = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return '\n'.join(text)
        
        elif suffix == '.docx':
            doc = Document(file_path)
            text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    text.append(para.text)
            return '\n'.join(text)
        
        elif suffix in ['.xlsx', '.xls']:
            excel_file = pd.read_excel(file_path, sheet_name=None)
            text = []
            for sheet_name, df in excel_file.items():
                text.append(f"### Sheet: {sheet_name}")
                text.append(df.to_string())
            return '\n'.join(text)
        
        else:
            # Fallback to text read for unknown formats
            return file_path.read_text(encoding='utf-8', errors='ignore')
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


class LocalVectorStore:
    """Tiny local vector store using numpy (cosine similarity).

    Stores:
      - vectors.npy : float32 [N, D]
      - meta.jsonl  : per-chunk metadata

    This is intentionally simple so you can run it anywhere.
    For large corpora, switch to FAISS / Azure AI Search.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.root / 'vectors.npy'
        self.meta_path = self.root / 'meta.jsonl'

        self._vectors = None
        self._meta = None

    def _load(self):
        if self._vectors is None:
            if self.vec_path.exists():
                self._vectors = np.load(self.vec_path)
            else:
                self._vectors = np.zeros((0, 0), dtype=np.float32)
        if self._meta is None:
            self._meta = []
            if self.meta_path.exists():
                for line in self.meta_path.read_text(encoding='utf-8').splitlines():
                    self._meta.append(json.loads(line))

    def add(self, embeddings: np.ndarray, metadatas: List[dict]):
        self._load()
        embeddings = embeddings.astype(np.float32)
        if self._vectors.size == 0:
            self._vectors = embeddings
        else:
            self._vectors = np.vstack([self._vectors, embeddings])
        self._meta.extend(metadatas)

        np.save(self.vec_path, self._vectors)
        with self.meta_path.open('w', encoding='utf-8') as f:
            for m in self._meta:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[float, dict]]:
        self._load()
        if self._vectors.size == 0:
            return []

        q = query_vec.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        # cosine similarity
        V = self._vectors
        # normalize
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        sims = (Vn @ qn.T).reshape(-1)
        idx = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self._meta[int(i)]) for i in idx]


def build_index_from_folder(folder: str | Path, store_dir: str | Path,
                            llm: LLMClient, glob: str = '**/*.md') -> int:
    """Build index from multiple file formats (md, pdf, docx, xlsx)."""
    folder = Path(folder)
    docs = []
    
    # Support multiple file formats
    file_patterns = [
        '**/*.md',
        '**/*.pdf',
        '**/*.docx',
        '**/*.xlsx',
        '**/*.xls'
    ]
    
    for pattern in file_patterns:
        for p in folder.glob(pattern):
            if p.is_file():
                text = extract_text_from_file(p)
                if text.strip():
                    docs.append((p.as_posix(), text))

    chunks = []
    metas = []
    for path, txt in docs:
        for j, c in enumerate(simple_chunk(txt)):
            chunks.append(c)
            metas.append({"source": path, "chunk": j, "text": c})

    if not chunks:
        return 0

    vecs = llm.embed(chunks)
    store = LocalVectorStore(store_dir)
    store.add(np.array(vecs), metas)
    return len(chunks)


def rag_answer(question: str, store_dir: str | Path, llm: LLMClient,
               system_prompt: str, top_k: int = 5, max_context_chars: int = 4500) -> tuple[str, list[dict]]:
    store = LocalVectorStore(store_dir)
    qvec = np.array(llm.embed([question])[0])
    hits = store.search(qvec, top_k=top_k)
    contexts = []
    used = 0
    citations = []
    for score, meta in hits:
        text = meta.get('text','')
        if used + len(text) > max_context_chars:
            continue
        contexts.append(f"[source={meta.get('source')} chunk={meta.get('chunk')} score={score:.3f}]\n{text}")
        citations.append({"source": meta.get('source'), "chunk": meta.get('chunk'), "score": score})
        used += len(text)

    context_block = "\n\n".join(contexts) if contexts else "(no retrieved context)"

    user_prompt = f"""質問: {question}

参照コンテキスト（抜粋）:
{context_block}

指示: 上のコンテキストを優先して回答し、根拠がコンテキストにない場合は『不明』と明記してください。"""

    answer = llm.chat([
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ])
    return answer, citations