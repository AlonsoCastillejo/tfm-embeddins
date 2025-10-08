from openai import OpenAI
from tqdm import tqdm
from .config import Config

class EmbeddingsManager:
    
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    # Generar embedding para un texto
    def generate_embedding(self, text, use_large_model=False):

        model = Config.EMBEDDING_MODEL_LARGE if use_large_model else Config.EMBEDDING_MODEL_SMALL
    
        try:
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[Error] Error generando embedding: {e}")
            
            # Si el modelo grande falla ... (fallback al peke)
            if use_large_model:
                return self.generate_embedding(text, use_large_model=False)
            raise e
        
    # Generar embeddings para múltiples textos
    def generate_embeddings_batch(self, texts, use_large_model=False):
        
        embeddings = []
        
        print(f"Generando embeddings...")
        for text in tqdm(texts, desc="Embeddings"):
            embedding = self.generate_embedding(text, use_large_model)
            embeddings.append(embedding)
        
        print(" Embeddings generados correctamente.")
        return embeddings
    
    