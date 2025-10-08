import streamlit as st
import pandas as pd
import os
import shutil
import sys
from datetime import datetime
import json

# Añadir src al path para imports
sys.path.append('./src')

from src.data_processor import DataProcessor
from src.embeddings_manager import EmbeddingsManager
from src.database_manager import DatabaseManager
from src.search_engine import PropertySearchEngine
from src.query_enhancer import QueryEnhancer
from src.config import Config

# Configuración de página
st.set_page_config(
    page_title="Sistema de Búsqueda Inmobiliaria",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .property-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background: white;
    }

    .property-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }

    .property-price {
        font-size: 1.5rem;
        font-weight: bold;
        color: #e74c3c;
        margin-bottom: 0.5rem;
    }

    .property-details {
        color: #7f8c8d;
        margin-bottom: 0.5rem;
    }

    .relevance-score {
        background: linear-gradient(90deg, #3498db, #2ecc71);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }

    .llm-analysis {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }

    .filter-tag {
        background: #17a2b8;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        margin: 0.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Estado de la aplicación
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = PropertySearchEngine()
    st.session_state.query_enhancer = QueryEnhancer()

def display_main_header():
    """Muestra el header principal"""
    st.markdown('<h1 class="main-header">Sistema de Búsqueda Inmobiliaria con IA</h1>', unsafe_allow_html=True)
    st.markdown("---")

def display_sidebar():
    """Sidebar con opciones y estadísticas"""
    st.sidebar.title("Panel de Control")

    # Estadísticas de la base de datos
    try:
        stats = st.session_state.search_engine.db_manager.get_collection_stats()
        st.sidebar.metric("Propiedades en BD", stats['total_properties'])
    except:
        st.sidebar.warning("Base de datos no inicializada")

    st.sidebar.markdown("---")

    # Opciones de gestión
    st.sidebar.subheader(" Gestión de Datos")

    # Upload de CSV
    uploaded_file = st.sidebar.file_uploader(
        "Subir archivo CSV",
        type=['csv'],
        help="Sube un archivo CSV con datos de propiedades"
    )

    if uploaded_file is not None:
        if st.sidebar.button(" Procesar y Cargar"):
            process_uploaded_file(uploaded_file)

    # Borrar base de datos
    st.sidebar.markdown("---")
    st.sidebar.subheader(" Resetear Sistema")
    if st.sidebar.button(" Borrar Base de Datos", type="secondary"):
        if st.sidebar.checkbox("Confirmar eliminación"):
            reset_database()

def process_uploaded_file(uploaded_file):
    """Procesa el archivo CSV subido"""
    try:
        with st.spinner(" Procesando archivo..."):
            # Guardar archivo temporalmente
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Cargar y procesar datos
            df = pd.read_csv(temp_path)
            st.sidebar.success(f" Archivo cargado: {len(df)} registros")

            # Limpiar datos
            df_clean = DataProcessor.clean_dataframe(df)

            # Generar textos y metadata
            with st.spinner(" Generando textos descriptivos..."):
                descriptive_texts = df_clean.apply(DataProcessor.build_descriptive_text, axis=1).tolist()
                structured_metadata = df_clean.apply(DataProcessor.build_structured_metadata, axis=1).tolist()

            # Generar embeddings
            with st.spinner(" Generando embeddings..."):
                embeddings_manager = EmbeddingsManager()
                embeddings = embeddings_manager.generate_embeddings_batch(descriptive_texts, use_large_model=True)

            # Guardar en base de datos
            with st.spinner(" Guardando en base de datos..."):
                db_manager = DatabaseManager()
                db_manager.add_properties_to_db(df_clean, descriptive_texts, embeddings, structured_metadata)

            st.sidebar.success(" Datos procesados y guardados correctamente!")
            st.rerun()

    except Exception as e:
        st.sidebar.error(f" Error procesando archivo: {str(e)}")

def reset_database():
    """Resetea la base de datos"""
    try:
        if os.path.exists(Config.CHROMADB_PATH):
            shutil.rmtree(Config.CHROMADB_PATH)
        st.sidebar.success(" Base de datos eliminada")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f" Error: {str(e)}")

def display_llm_analysis(query_info, query):
    """Muestra el análisis del LLM"""
    st.markdown('<div class="llm-analysis">', unsafe_allow_html=True)
    st.markdown("###  Análisis Inteligente de la Consulta")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"** Consulta original:** `{query}`")
        st.markdown(f"** Query optimizada:** `{query_info['semantic_query']}`")

    with col2:
        if query_info.get('filters'):
            st.markdown("** Filtros aplicados:**")
            for key, value in query_info['filters'].items():
                if key == 'precio_max' and value:
                    st.markdown(f'<span class="filter-tag"> Max: {value:,}€</span>', unsafe_allow_html=True)
                elif key == 'habitaciones' and value:
                    st.markdown(f'<span class="filter-tag"> {value} hab</span>', unsafe_allow_html=True)
                elif key == 'localidad' and value:
                    st.markdown(f'<span class="filter-tag"> {value}</span>', unsafe_allow_html=True)
                elif key == 'tipo' and value:
                    st.markdown(f'<span class="filter-tag"> {value}</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def display_property_card(result, index):
    """Muestra una tarjeta de propiedad"""
    doc = result['document']
    meta = result['metadata']
    relevance = result['relevance_score'] * 100

    with st.container():
        st.markdown('<div class="property-card">', unsafe_allow_html=True)

        # Header con título y relevancia
        col1, col2 = st.columns([3, 1])

        with col1:
            # Extraer título del documento
            doc_lines = [line.strip() for line in doc.split('\n') if line.strip()]
            titulo = doc_lines[0].replace("Propiedad: ", "") if doc_lines else "Sin título"
            st.markdown(f'<div class="property-title"> {titulo}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<span class="relevance-score"> {relevance:.1f}%</span>', unsafe_allow_html=True)

        # Información principal
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            precio = meta.get('precio', 'Consultar')
            if isinstance(precio, (int, float)):
                precio_formatted = f"{precio:,.0f}€"
            else:
                precio_formatted = str(precio)
            st.markdown(f'<div class="property-price"> {precio_formatted}</div>', unsafe_allow_html=True)

        with col2:
            habitaciones = meta.get('Habitaciones', 'N/A')
            st.metric(" Habitaciones", habitaciones)

        with col3:
            banos = meta.get('Baños', 'N/A')
            st.metric(" Baños", banos)

        with col4:
            metros = meta.get('metros', 'N/A')
            if isinstance(metros, (int, float)):
                metros_formatted = f"{metros}m²"
            else:
                metros_formatted = str(metros)
            st.metric("Superficie", metros_formatted)

        # Ubicación
        barrio = meta.get('barrio', 'N/A')
        distrito = meta.get('distrito', 'N/A')
        localidad = meta.get('localidad', 'N/A')
        st.markdown(f'<div class="property-details"> {localidad} - {distrito} - {barrio}</div>', unsafe_allow_html=True)

        # Características adicionales
        caracteristicas = []
        if meta.get('tipo'):
            caracteristicas.append(f" {meta['tipo']}")
        if meta.get('Planta'):
            caracteristicas.append(f" Planta {meta['Planta']}")
        if meta.get('Ascensor') == '1':
            caracteristicas.append(" Con ascensor")
        if meta.get('Exterior') == '1':
            caracteristicas.append(" Exterior")

        if caracteristicas:
            st.markdown(f'<div class="property-details">{" • ".join(caracteristicas)}</div>', unsafe_allow_html=True)

        # Precio por m²
        if meta.get('precio_por_m2'):
            st.markdown(f'<div class="property-details"> {meta["precio_por_m2"]:.0f}€/m²</div>', unsafe_allow_html=True)

        # Descripción (si existe)
        descripcion_lines = [line for line in doc_lines if line.startswith('Descripción:')]
        if descripcion_lines:
            descripcion = descripcion_lines[0].replace('Descripción: ', '')
            if len(descripcion) > 200:
                descripcion = descripcion[:200] + "..."
            st.markdown(f"** Descripción:** {descripcion}")

        # URL
        if meta.get('url'):
            st.markdown(f" [Ver detalles completos]({meta['url']})")

        # Score de completitud
        completeness = meta.get('completeness_score', 0) * 100
        st.progress(completeness/100, text=f" Completitud de datos: {completeness:.0f}%")

        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Función principal de la aplicación"""
    display_main_header()
    display_sidebar()

    # Área principal de búsqueda
    st.header(" Búsqueda Inteligente de Propiedades")

    # Input de búsqueda
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input(
            "¿Qué propiedad estás buscando?",
            placeholder="Ej: piso barato en madrid con 3 habitaciones",
            help="Describe tu búsqueda en lenguaje natural"
        )

    with col2:
        search_button = st.button(" Buscar", type="primary", use_container_width=True)

    # Opciones avanzadas
    with st.expander(" Opciones Avanzadas"):
        col1, col2 = st.columns(2)
        with col1:
            num_results = st.slider("Número de resultados", 1, 10, 5)
            show_analysis = st.checkbox("Mostrar análisis LLM", value=True)
        with col2:
            test_mode = st.checkbox("Modo test (solo análisis)", value=False)

    # Procesamiento de búsqueda
    if (search_button or query) and query.strip():
        try:
            # Análisis con LLM
            query_info = st.session_state.query_enhancer.get_enhanced_query_info(query, show_analysis=False)

            if show_analysis:
                display_llm_analysis(query_info, query)

            if not test_mode:
                # Realizar búsqueda
                with st.spinner(" Buscando propiedades..."):
                    results = st.session_state.search_engine.search(query, n_results=num_results)

                # Mostrar resultados
                if results:
                    st.markdown(f"###  Encontrados {len(results)} resultados para: *'{query}'*")
                    st.markdown("---")

                    for i, result in enumerate(results):
                        display_property_card(result, i)
                        if i < len(results) - 1:
                            st.markdown("---")
                else:
                    st.warning(f" No se encontraron resultados para: '{query}'")
                    st.info(" Intenta con términos más generales o diferentes")

        except Exception as e:
            st.error(f" Error en la búsqueda: {str(e)}")

    # Ejemplos de búsqueda
    if not query:
        st.markdown("###  Ejemplos de búsqueda:")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(" Piso barato Madrid", use_container_width=True):
                st.session_state.example_query = "piso barato madrid con 3 habitaciones"

        with col2:
            if st.button(" Casa familiar jardín", use_container_width=True):
                st.session_state.example_query = "casa moderna para familia con jardín"

        with col3:
            if st.button(" Ático céntrico", use_container_width=True):
                st.session_state.example_query = "ático céntrico barcelona con terraza"

        # Si se seleccionó un ejemplo, actualizar el input
        if 'example_query' in st.session_state:
            st.rerun()

    # Footer con información
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>
         Potenciado por IA •  Búsqueda Semántica •  Filtros Inteligentes
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()