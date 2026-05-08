from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib import request


TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
EMBEDDING_DIMENSIONS = 256


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def embed_text(text: str, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokenize(text):
        vector[hash(token) % dimensions] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


class VectorStore:
    def __init__(self, documents: Iterable[Document] | None = None) -> None:
        self._items: list[tuple[Document, list[float]]] = []
        if documents:
            self.add_documents(documents)

    def add_documents(self, documents: Iterable[Document]) -> None:
        for document in documents:
            self._items.append((document, embed_text(document.text)))

    def search(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        query_vector = embed_text(query)
        scored = [
            (document, cosine_similarity(query_vector, embedding))
            for document, embedding in self._items
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: max(k, 0)]


class RAGPipeline:
    def __init__(
        self,
        documents: Iterable[Document],
        llm_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._store = VectorStore(documents)
        self._llm_fn = llm_fn or self._default_llm

    def retrieve(self, question: str, k: int = 3) -> list[tuple[Document, float]]:
        return self._store.search(question, k=k)

    def answer(self, question: str, k: int = 3) -> str:
        matches = self.retrieve(question, k=k)
        context = "\n\n".join(
            f"[{doc.id}] {doc.text}" for doc, _score in matches if doc.text.strip()
        )
        prompt = (
            "Use only the provided context to answer the question. "
            "If the answer is not present, say you do not know.\n\n"
            f"Context:\n{context or 'No relevant documents found.'}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return self._llm_fn(prompt)

    @staticmethod
    def _default_llm(prompt: str) -> str:
        return (
            "No external LLM configured. Pass a custom llm_fn or use the --openai option.\n\n"
            f"Prompt preview:\n{prompt}"
        )


def openai_chat_completion(prompt: str, model: str = "gpt-4o-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when using --openai")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def load_documents_from_directory(path: Path) -> list[Document]:
    documents: list[Document] = []
    for file in sorted(path.glob("*")):
        if file.suffix.lower() not in {".txt", ".md"}:
            continue
        documents.append(Document(id=file.name, text=file.read_text(encoding="utf-8")))
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal from-scratch RAG experiment")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--docs", default="./docs", help="Directory containing .txt/.md docs")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved documents")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI Chat Completions API")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    args = parser.parse_args()

    docs_path = Path(args.docs)
    if not docs_path.exists() or not docs_path.is_dir():
        raise SystemExit(f"Document directory not found: {docs_path}")

    documents = load_documents_from_directory(docs_path)
    if not documents:
        raise SystemExit("No .txt or .md documents found in the docs directory")

    llm_fn: Callable[[str], str] | None = None
    if args.openai:
        llm_fn = lambda prompt: openai_chat_completion(prompt, model=args.model)

    pipeline = RAGPipeline(documents=documents, llm_fn=llm_fn)
    print(pipeline.answer(args.question, k=args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
