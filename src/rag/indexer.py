import os
import chromadb
from chromadb.config import Settings
import logging
from dotenv import load_dotenv

load_dotenv()

from src.rag.chunker import load_and_chunk_research_files
from src.rag.embedder import GeminiEmbeddingFunction

logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "vectorstore")
RESEARCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "research")

def get_chroma_client():
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    
    # Use persistent client
    client = chromadb.PersistentClient(path=DB_DIR)
    return client

def build_index():
    """
    Reads research files, chunks them, and indexes them into ChromaDB.
    """
    try:
        # Check if API key is present
        if not os.environ.get("GEMINI_API_KEY"):
            logger.warning("GEMINI_API_KEY not found. Skipping indexing.")
            return False
            
        client = get_chroma_client()
        embedder = GeminiEmbeddingFunction()
        
        # Get or create collection
        collection = client.get_or_create_collection(
            name="research_notes",
            embedding_function=embedder
        )
        
        # Load chunks
        chunks = load_and_chunk_research_files(RESEARCH_DIR)
        
        if not chunks:
            logger.info("No research files found to index.")
            return True
            
        # Extract lists for Chroma
        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Upsert documents
        # ChromaDB automatically handles batching in recent versions, but we can batch just in case.
        batch_size = 50
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
            
        logger.info(f"Successfully indexed {len(chunks)} chunks into ChromaDB.")
        return True
        
    except Exception as e:
        logger.error(f"Error building index: {e}")
        return False

if __name__ == "__main__":
    # Configure logging for standalone run
    logging.basicConfig(level=logging.INFO)
    
    # Load env for standalone run
    from dotenv import load_dotenv
    load_dotenv()
    
    build_index()
