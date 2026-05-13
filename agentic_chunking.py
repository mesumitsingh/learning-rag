import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA


load_dotenv()

llm = ChatNVIDIA(
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0
)

tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024. 

Model Y Performance 

The Model Y because the best-selling vehicle globally, with 350,000 units sold. 

Production Challenges

Supply chain issues caused a 12% increase in production costs. 

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside whatsoever making it impossible to split properly.
"""

prompt = f"""
You are a text chunking expert. Split this text into logical chunks. 

Rules: 
- Each chunk should be around 200 characters or loss
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text: 
{tesla_text}

Return the text with <<<SPLIT>>> markers where you want to split:
"""

print("Asking AI to chunk the text...")
response = llm.invoke(prompt)
marked_text = response.content

chunks = marked_text.split("<<<SPLIT>>>")
if chunks: 
    print("chunks are ready...")

clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:
        clean_chunks.append(cleaned)

print("\n AGENTIC CHUNKING RESULTS:")
print("=" * 50)

for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f"{chunk}")
    print()
