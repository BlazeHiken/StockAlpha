import os
import re

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into chunks of approximately `chunk_size` words with `overlap` words.
    
    Args:
        text (str): The input text to chunk.
        chunk_size (int): The target number of words per chunk.
        overlap (int): The number of overlapping words between consecutive chunks.
        
    Returns:
        list[str]: A list of text chunks.
    """
    words = re.split(r'\s+', text.strip())
    if not words:
        return []
        
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        
        # If we reached the end, break
        if end >= len(words):
            break
            
        # Move forward by chunk_size - overlap
        start += (chunk_size - overlap)
        
    return chunks

def load_and_chunk_research_files(research_dir: str, chunk_size: int = 200, overlap: int = 40) -> list[dict]:
    """
    Loads all markdown/text files from a directory and chunks them.
    
    Args:
        research_dir (str): Path to the directory containing research notes.
        chunk_size (int): Words per chunk.
        overlap (int): Word overlap.
        
    Returns:
        list[dict]: A list of dictionaries containing 'id', 'text', 'metadata'.
    """
    if not os.path.exists(research_dir):
        return []
        
    all_chunks = []
    
    for filename in os.listdir(research_dir):
        if filename.endswith(".md") or filename.endswith(".txt"):
            filepath = os.path.join(research_dir, filename)
            ticker = filename.split(".")[0] 
            if filename.endswith(".md"):
                ticker = filename[:-3]
            elif filename.endswith(".txt"):
                ticker = filename[:-4]
                
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            text_chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
            
            for i, chunk in enumerate(text_chunks):
                all_chunks.append({
                    "id": f"{ticker}_chunk_{i}",
                    "text": chunk,
                    "metadata": {"ticker": ticker, "source": filename, "chunk_index": i}
                })
                
    return all_chunks
