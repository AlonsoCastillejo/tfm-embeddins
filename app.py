import os
import shutil
import sys

import pandas as pd

# Añadir src al path para imports
sys.path.append("./src")

from src.config import Config
from src.data_processor import DataProcessor
from src.database_manager import DatabaseManager
from src.embeddings_manager import EmbeddingsManager
from src.query_enhancer import QueryEnhancer
from src.search_engine import PropertySearchEngine


class PropertySearchApp:
    def __init__(self):
        self.search_engine = PropertySearchEngine()

    # Cargamos y procesamos CSV con separación texto/metadata
    def load_and_process_data(self, csv_file):
        print("<<CARGANDO Y PROCESANDO DATOS>>")
        print("=" * 50)

        df = pd.read_csv(csv_file)
        print(f"Archivo cargado: {csv_file}")

        df_clean = DataProcessor.clean_dataframe(df)

        # Generar SOLO textos descriptivos para embeddings
        print("Generando textos descriptivos (solo texto)...")
        descriptive_texts = df_clean.apply(
            DataProcessor.build_descriptive_text, axis=1
        ).tolist()

        # Generar metadata estructurada separada
        print("Extrayendo metadata estructurada (numérica/categórica)...")
        structured_metadata = df_clean.apply(
            DataProcessor.build_structured_metadata, axis=1
        ).tolist()

        print(f"Ejemplo de texto descriptivo:\n{descriptive_texts[0][:200]}...")
        print(
            f"\nEjemplo de metadata estructurada:\n{list(structured_metadata[0].keys())}"
        )

        # Generar embeddings SOLO del texto descriptivo
        embeddings_manager = EmbeddingsManager()
        embeddings = embeddings_manager.generate_embeddings_batch(
            descriptive_texts, use_large_model=True
        )

        # Guardar en bbdd
        db_manager = DatabaseManager()
        db_manager.add_properties_to_db(
            df_clean, descriptive_texts, embeddings, structured_metadata
        )

        print("Datos procesados y guardados correctamente. [OK]")

    # Función para mostrar estadísticas de la bbdd
    # TODO: Terminar analisis estadístico
    def show_stats(self):
        stats = self.search_engine.db_manager.get_collection_stats()
        print(f"Total de propiedades: {stats['total_properties']}")

        if stats["sample_data"]:
            print("\nMestra de propiedades:")
            for i, meta in enumerate(stats["sample_data"]):
                print(
                    f"{i + 1}. {meta['city']} - {meta['rooms']} hab. - {meta['price']}"
                )

    # Busqueda interactiva
    def search_interactive(self):
        print("\n<<BÚSQUEDA INTERACTIVA>>")
        print("=" * 30)
        print("Ejemplos: 'piso barato barcelona', 'flat chip madrid'")
        print("Escribe 'salir' para terminar\n")

        while True:
            query = input("¿Qué propiedad buscas? ")

            if query.lower() in ["salir", "exit", "quit"]:
                print("¡Hasta luego!")
                break

            if not query.strip():
                continue

            results = self.search_engine.search(query, n_results=3)
            self.print_results(query, results)

    # Imprimir resultados de búsqueda mejorado
    def print_results(self, query, results):
        if not results:
            print(f"[KO] No se encontraron resultados para: '{query}'\n")
            return

        print(f"\nRESULTADOS PARA: '{query}'")
        print("=" * 60)

        for i, result in enumerate(results, 1):
            doc = result["document"]
            meta = result["metadata"]
            relevance = result["relevance_score"] * 100

            print(f"\nOPCIÓN {i} (Relevancia: {relevance:.1f}%)")
            print("=" * 50)

            # Extraer título del documento
            doc_lines = [line.strip() for line in doc.split("\n") if line.strip()]
            titulo = (
                doc_lines[0].replace("Propiedad: ", "") if doc_lines else "Sin título"
            )
            print(f"{titulo}")

            # Información clave de metadata
            print("\nINFORMACIÓN PRINCIPAL:")
            precio = meta.get("precio", "Consultar")
            if isinstance(precio, (int, float)):
                precio = f"{precio:,.0f}€"

            habitaciones = meta.get("Habitaciones", "N/A")
            banos = meta.get("Baños", "N/A")
            metros = meta.get("metros", "N/A")
            if isinstance(metros, (int, float)):
                metros = f"{metros}m²"

            print(f"   Precio: {precio}")
            print(f"   Habitaciones: {habitaciones}")
            print(f"   Baños: {banos}")
            print(f"   Superficie: {metros}")

            # Precio por m² si está disponible
            if meta.get("precio_por_m2"):
                print(f"   Precio/m²: {meta['precio_por_m2']:.0f}€/m²")

            # Ubicación
            print(f"\nUBICACIÓN:")
            barrio = meta.get("barrio", "N/A")
            distrito = meta.get("distrito", "N/A")
            localidad = meta.get("localidad", "N/A")
            print(f"   {localidad} - {distrito} - {barrio}")

            # Características adicionales importantes
            caracteristicas_importantes = []
            if meta.get("tipo"):
                caracteristicas_importantes.append(f"Tipo: {meta['tipo']}")
            if meta.get("Planta"):
                caracteristicas_importantes.append(f"Planta: {meta['Planta']}")
            if meta.get("Ascensor") == "1":
                caracteristicas_importantes.append("Con ascensor")
            if meta.get("Exterior") == "1":
                caracteristicas_importantes.append("Exterior")
            if meta.get("Antigüedad"):
                caracteristicas_importantes.append(f"Antigüedad: {meta['Antigüedad']}")

            if caracteristicas_importantes:
                print(f"\nCARACTERÍSTICAS:")
                for caract in caracteristicas_importantes:
                    print(f"   • {caract}")

            # Descripción (solo las primeras líneas)
            descripcion_lines = [
                line for line in doc_lines if line.startswith("Descripción:")
            ]
            if descripcion_lines:
                descripcion = descripcion_lines[0].replace("Descripción: ", "")
                if len(descripcion) > 200:
                    descripcion = descripcion[:200] + "..."
                print(f"\nDESCRIPCIÓN:")
                print(f"   {descripcion}")

            # Score de completitud
            completeness = meta.get("completeness_score", 0) * 100
            print(f"\nCompletitud de datos: {completeness:.0f}%")

            # URL
            if meta.get("url"):
                print(f"\nVer detalles: {meta['url']}")

            print("-" * 50)

        print("\n")

    # Función para probar análisis LLM sin búsqueda
    def test_llm_analysis(self):
        print("\nMODO TEST - ANÁLISIS LLM")
        print("=" * 50)
        print("Prueba cómo el LLM interpreta diferentes consultas")
        print("Ejemplos: 'casa moderna familiar', 'piso barato madrid 3 habitaciones'")
        print("Escribe 'salir' para volver al menú\n")

        query_enhancer = QueryEnhancer()

        while True:
            query = input("Consulta a analizar: ")

            if query.lower() in ["salir", "exit", "quit"]:
                break

            if not query.strip():
                continue

            try:
                # Solo mostrar análisis, no hacer búsqueda
                query_enhancer.get_enhanced_query_info(query, show_analysis=True)
                print("\n" + "-" * 60)
                print("Análisis completado")
                print("-" * 60 + "\n")

            except Exception as e:
                print(f"Error en análisis: {e}\n")

    def reset_database(self):
        """Resetea la base de datos vectorial"""
        print("\nRESETEAR BASE DE DATOS")
        print("=" * 40)
        print("ADVERTENCIA: Esta acción eliminará todos los datos")

        confirm = input("¿Estás seguro? Escribe 'BORRAR' para confirmar: ")

        if confirm == "BORRAR":
            try:
                # Resetear la base de datos usando el DatabaseManager
                db_manager = DatabaseManager()
                db_manager.reset_database()
                print("Base de datos reseteada correctamente")
            except Exception as e:
                print(f"Error al resetear base de datos: {e}")
        else:
            print("Operación cancelada")


def main():
    """Función principal"""
    app = PropertySearchApp()

    print("SISTEMA DE BÚSQUEDA DE PROPIEDADES")
    print("=" * 50)

    while True:
        print("\n- OPCIONES:")
        print("1. Cargar datos desde CSV")
        print("2. Ver estadísticas")
        print("3. Buscar propiedades")
        print("4. Probar análisis LLM")
        print("5. Borrar base de datos")
        print("6. Salir")

        choice = input("\nSelecciona una opción (1-6): ")

        if choice == "1":
            csv_file = input("Nombre del archivo CSV: ")
            try:
                app.load_and_process_data(csv_file)
            except Exception as e:
                print(f"[KO] Error: {e}")

        elif choice == "2":
            app.show_stats()

        elif choice == "3":
            app.search_interactive()

        elif choice == "4":
            app.test_llm_analysis()

        elif choice == "5":
            app.reset_database()

        elif choice == "6":
            print("¡Hasta luego!")
            break

        else:
            print("Opción no válida...")


if __name__ == "__main__":
    main()
