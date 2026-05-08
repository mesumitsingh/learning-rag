# learning-rag

Learning and experimenting with RAG, from scratch with document retrieval, embeddings, and LLM integration.

## Minimal from-scratch RAG implementation

This repository now includes a lightweight Python RAG pipeline in `/home/runner/work/learning-rag/learning-rag/rag.py` with:

- Document loading from local `.txt` and `.md` files
- Simple embedding generation from token hashing
- Vector similarity retrieval (cosine similarity)
- LLM integration via pluggable `llm_fn`
- Optional OpenAI Chat Completions integration using `OPENAI_API_KEY`

## Quick start

1. Create a docs folder and add text files:

```bash
mkdir -p docs
printf "Paris is the capital of France." > docs/france.txt
printf "Tokyo is the capital of Japan." > docs/japan.txt
```

2. Run with the built-in placeholder LLM:

```bash
python rag.py "What is the capital of France?" --docs ./docs
```

3. Run with OpenAI API:

```bash
export OPENAI_API_KEY=your_key
python rag.py "What is the capital of Japan?" --docs ./docs --openai --model gpt-4o-mini
```

## Tests

```bash
python -m unittest discover -s tests -v
```
