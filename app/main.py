import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()

def start_chat():
    # 1. Setup Embeddings (Must match the ingestion script)
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    # 2. Connect to Pinecone
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"), 
        embedding=embeddings
    )

    # 3. Setup Groq LLM (The free brain)
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # 4. Create the RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    print("\n🤖 Bot Ready! Type 'exit' to quit.")
    while True:
        query = input("\n👤 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        response = qa_chain.invoke(query)
        print(f"🤖 Bot: {response['result']}")

if __name__ == "__main__":
    start_chat()
