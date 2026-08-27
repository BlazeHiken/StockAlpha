import os
import logging
from src.rag.indexer import get_chroma_client
from src.rag.embedder import GeminiEmbeddingFunction

logger = logging.getLogger(__name__)

def retrieve_research_for_portfolio(tickers: list[str], n_results: int = 5) -> list[str]:
    """
    Retrieves the most relevant research chunks for a given set of tickers.
    Instead of semantic search, since we want specific tickers, we can query by metadata,
    or just do a simple get based on metadata filter.
    
    Args:
        tickers (list[str]): List of stock tickers in the optimal portfolio.
        n_results (int): Number of chunks to retrieve per ticker (approx).
        
    Returns:
        list[str]: A list of formatted research snippets.
    """
    try:
        # Check API key before proceeding
        if not os.environ.get("GEMINI_API_KEY"):
            return ["(GEMINI_API_KEY not found. Research notes unavailable.)"]

        client = get_chroma_client()
        embedder = GeminiEmbeddingFunction()
        
        try:
            collection = client.get_collection(name="research_notes", embedding_function=embedder)
        except Exception as e:
            logger.warning(f"Chroma collection not found. Did you run indexer? {e}")
            return []
            
        retrieved_snippets = []
        
        # We can query by simply filtering on the metadata for each ticker.
        # Since we just want the context for the selected tickers, semantic search isn't strictly necessary
        # but we can do a hybrid or just fetch all chunks for the selected tickers.
        # For simplicity and relevance, we'll fetch up to `n_results` chunks per ticker.
        
        for ticker in tickers:
            results = collection.get(
                where={"ticker": ticker},
                limit=n_results
            )
            
            if results and results['documents'] and results['metadatas']:
                for doc, meta in zip(results['documents'], results['metadatas']):
                    source = meta.get('source', 'Unknown')
                    snippet = f"[Source: {source} (Ticker: {ticker})]\n{doc}"
                    retrieved_snippets.append(snippet)
                    
        return retrieved_snippets
        
    except Exception as e:
        logger.error(f"Error retrieving research: {e}")
        return []
