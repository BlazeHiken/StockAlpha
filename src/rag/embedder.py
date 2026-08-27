import os
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom ChromaDB embedding function using the new google-genai SDK.
    """
    def __init__(self, model_name: str = "gemini-embedding-001"):
        # Relies on GEMINI_API_KEY being set in the environment
        self.client = genai.Client()
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        """
        Embeds a list of documents (strings) using Gemini.
        """
        if not input:
            return []
            
        # The new genai SDK expects a list of strings and returns a response with embeddings.
        # Check documentation of google-genai, generally client.models.embed_content
        
        try:
            import typing
            docs = typing.cast(typing.Any, input)
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=docs
            )
            
            # Extract embeddings from response.
            # Depending on the response structure, it is usually response.embeddings
            # where each embedding has a .values attribute.
            
            embeddings = []
            if hasattr(response, 'embeddings') and response.embeddings is not None:
                for emb in response.embeddings:
                    if emb.values is not None:
                        embeddings.append(emb.values)
            else:
                # Fallback if structure is different
                pass
                
            return typing.cast(typing.Any, embeddings)
            
        except Exception as e:
            print(f"Error embedding documents: {e}")
            raise
