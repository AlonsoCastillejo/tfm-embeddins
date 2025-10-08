"""
Script para borrar la base de datos ChromaDB y empezar limpio
"""

import chromadb
import shutil
import os
from config import Config

def reset_database():
    """Borra completamente la base de datos ChromaDB"""

    print("🗑️  Eliminando base de datos ChromaDB...")

    try:
        # Método 1: Intentar borrar la colección
        client = chromadb.PersistentClient(path=Config.CHROMADB_PATH)
        client.delete_collection(Config.COLLECTION_NAME)
        print(f"✓ Colección '{Config.COLLECTION_NAME}' eliminada")
    except Exception as e:
        print(f"- No se pudo eliminar colección: {e}")

    # Método 2: Borrar directorio completo (más seguro)
    if os.path.exists(Config.CHROMADB_PATH):
        shutil.rmtree(Config.CHROMADB_PATH)
        print(f"✓ Directorio '{Config.CHROMADB_PATH}' eliminado")
    else:
        print("- No había directorio ChromaDB")

    print("🆕 Base de datos lista para cargar datos nuevos")

if __name__ == "__main__":
    reset_database()