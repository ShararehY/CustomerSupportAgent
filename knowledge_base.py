import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings


def create_knowledge_base(docs_dir="product_docs"):
    """
    Create a FAISS vector store from product documentation
    """
    documents = []
    
    # Read all text files in the docs directory
    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_dir, filename), "r") as f:
                content = f.read()
                documents.append(Document(page_content=content))
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    # Initialize Hugging Face embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector store
    vector_store = FAISS.from_documents(chunks, embedding_model)
    
    # Save the vector store
    vector_store.save_local("faiss_index")
    
    return vector_store

if __name__ == "__main__":
    # Create docs directory if it doesn't exist
    if not os.path.exists("product_docs"):
        os.makedirs("product_docs")
        # Create a sample FAQ file
        with open("product_docs/sample_faq.txt", "w") as f:
            f.write("""
            Q: What is our return policy?
            A: Items can be returned within 30 days of purchase with receipt.
            
            Q: How do I reset my password?
            A: Go to the login page and click on "Forgot Password" to receive reset instructions.
            
            Q: What payment methods do you accept?
            A: We accept credit cards, PayPal, and bank transfers.
            """)
    
    create_knowledge_base()
    print("Knowledge base created successfully!")