import os

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

load_dotenv()

tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024. 

Model Y Performance 

The Model Y because the best-selling vehicle globally, with 350,000 units sold. 

Production Challenges

Supply chain issues caused a 12% increase in production costs. 

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside whatsoever making it impossible to split properly.
"""

semantic_splitter = SemanticChunker(
    embeddings=NVIDIAEmbeddings(
        model=os.getenv("EmbeddingModel"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL")
    ),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=70
)

chunks = semantic_splitter.split_text(tesla_text)

print("SEMANTIC CHUNKING RESULTS:")
print("=" * 50)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f"{chunk}")
    print()
