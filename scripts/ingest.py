
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def run_ingestion():
    print("🚀 Loading PDFs from ./data...")
    loader = DirectoryLoader("./data", glob="./*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    
    if not docs:
        print("❌ No PDFs found! Add a file to the /data folder.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Free HuggingFace Embeddings (384 Dimensions)
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")

    # Auto-create the index if it doesn't exist
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        print(f"Creating index: {index_name} (384 dimensions)...")
        pc.create_index(
            name=index_name,
            dimension=384, 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    print("📤 Uploading chunks to Pinecone...")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
    print("✅ Done! Your data is ready.")

if __name__ == "__main__":
    run_ingestion()
