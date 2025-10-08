import chromadb
from .config import Config
from tqdm import tqdm
import pandas as pd

class DatabaseManager:
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.CHROMADB_PATH)
        self.collection = None
    
    # Creamos la coleccion de chromaDB
    def get_or_create_collection(self):
        try:
            self.collection = self.client.get_collection(Config.COLLECTION_NAME)
            print(f"Colección '{Config.COLLECTION_NAME}' cargada.")
        except:
            self.collection = self.client.create_collection(Config.COLLECTION_NAME)
            print(f"Colección '{Config.COLLECTION_NAME}' creada.")
        
        return self.collection
    
    # Añadimos propiedades a la bbdd con metadata estructurada
    def add_properties_to_db(self, df, descriptive_texts, embeddings, structured_metadata):
        if not self.collection:
            self.get_or_create_collection()

        print("Agregando propiedades a la base de datos...")

        for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Guardando")):
            self.collection.add(
                ids=[f"piso_{i}"],
                embeddings=[embeddings[i]],
                documents=[descriptive_texts[i]],  # Solo texto descriptivo
                metadatas=[structured_metadata[i]]  # Metadata estructurada completa
            )

        print("Base de datos actualizada correctamente.")

    # Estadisticas de la bbdd
    # TODO: Terminar analisis
    def get_collection_stats(self):
        if not self.collection:
            self.get_or_create_collection()

        collection_info = self.collection.get()
        return {
            'total_properties': len(collection_info['ids']),
            'sample_data': collection_info['metadatas'][:3] if collection_info['metadatas'] else []
        }

    def reset_database(self):
        """Resetea completamente la base de datos"""
        try:
            # Eliminar la colección si existe
            try:
                self.client.delete_collection(Config.COLLECTION_NAME)
                print(f"✅ Colección '{Config.COLLECTION_NAME}' eliminada")
            except Exception:
                print(f"ℹ️  La colección '{Config.COLLECTION_NAME}' no existía")

            # Resetear referencia local
            self.collection = None

            print("🗑️  Base de datos reseteada completamente")

        except Exception as e:
            print(f"❌ Error al resetear base de datos: {e}")
            raise e