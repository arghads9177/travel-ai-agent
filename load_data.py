# Import Libraries
import os
from dotenv import load_dotenv
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.chroma import ChromaDb
from agno.embedder.openai import OpenAIEmbedder
from agno.document.chunking.recursive import RecursiveChunking

# Load Environment Variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize PDF Knowledge Base
pdf_knowledge_base = PDFKnowledgeBase(
    path="./data",  # Directory containing PDF files
    vector_db=ChromaDb(
        collection="travel_data",  # Collection name for ChromaDB
        path="./data/chroma_db",  # Directory to store ChromaDB data
        embedder=OpenAIEmbedder(),  # Use OpenAI embeddings
        persistent_client=True  # Persist ChromaDB data on disk
    ),
    reader=PDFReader(chunk=True),  # Enable chunking during PDF reading
    chunking_strategy=RecursiveChunking(chunk_size=1000, overlap=50)  # Chunking strategy
)

# Load PDF data into ChromaDB
pdf_knowledge_base.load(recreate=True)  # Recreate the knowledge base if it exists

print("Travel data knowledge base stored successfully!")