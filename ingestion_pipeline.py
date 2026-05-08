import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma 
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"): 
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    # Check if docs directory exists
    if not os.path.exists(docs_path): 
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader( 
        path = docs_path,
        glob = "*.txt",
        loader_cls = TextLoader
    )

    documents = loader.load()

    if len(documents) == 0: 
        raise FileNotFoundError(f"No .txt fils found in {docs_path}. Please add your company documents.")
    
    for i, doc in enumerate(documents[:2]): # Show first 2 documents
        print(f"\n Document {i+1}")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")

    return documents

def main():
    print("Main Function")

    #1. Loading the files
    documents = load_documents(docs_path="docs")
    #2. Chunking the files
    #3. Embedding and Storing in Vector DB


if __name__ == "__main__":
    main()