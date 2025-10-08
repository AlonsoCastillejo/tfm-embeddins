import pandas as pd
import numpy as np

class DataProcessor:

    # Definir columnas de texto descriptivo vs estructuradas
    TEXT_COLUMNS = ['titulo', 'direccion', 'caract', 'caract_extra', 'descrip', 'descrip_keywords']

    NUMERICAL_COLUMNS = ['precio', 'metros', 'Habitaciones', 'Baños', 'postal_code']

    CATEGORICAL_COLUMNS = [
        'tipo', 'barrio', 'distrito', 'localidad', 'provincia', 'Antigüedad',
        'Aire acondicionado', 'Calefaccion', 'Garaje', 'Planta', 'Conservación',
        'Ascensor', 'Exterior', 'Trastero', 'Amueblado'
    ]

    # Limpieza rápida de datos
    @staticmethod
    def clean_dataframe(df):
        df_clean = df.copy()

        print(f"- Datos originales: {len(df)}")

        # Columnas críticas para mantener el registro
        critical_columns = ['titulo']
        df_clean = df_clean.dropna(subset=critical_columns)

        # Limpiar datos nulos en columnas de texto
        for col in DataProcessor.TEXT_COLUMNS:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('')

        # Limpiar datos nulos en columnas numéricas
        for col in DataProcessor.NUMERICAL_COLUMNS:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

        # Limpiar datos nulos en columnas categóricas
        for col in DataProcessor.CATEGORICAL_COLUMNS:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('No especificado')

        # Eliminar duplicados por URL
        if 'url_inmueble' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['url_inmueble'])
        else:
            df_clean = df_clean.drop_duplicates()

        print(f"- Datos después de limpieza: {len(df_clean)}")
        print(f"- Registros eliminados: {len(df) - len(df_clean)}")

        return df_clean

    # Crear texto descriptivo SOLO con columnas de texto
    @staticmethod
    def build_descriptive_text(row):
        """
        Construye texto descriptivo rico SOLO con columnas de texto.
        Las columnas numéricas/categóricas van en metadata separada.
        """
        text_parts = []

        # Título principal
        if pd.notna(row.get('titulo')) and str(row['titulo']) != '':
            text_parts.append(f"Propiedad: {row['titulo']}")

        # Dirección/ubicación
        if pd.notna(row.get('direccion')) and str(row['direccion']) != '':
            text_parts.append(f"Ubicación: {row['direccion']}")

        # Características principales
        if pd.notna(row.get('caract')) and str(row['caract']) != '':
            text_parts.append(f"Características: {row['caract']}")

        # Características adicionales
        if pd.notna(row.get('caract_extra')) and str(row['caract_extra']) != '':
            text_parts.append(f"Extras: {row['caract_extra']}")

        # Descripción principal
        if pd.notna(row.get('descrip')) and str(row['descrip']) != '':
            text_parts.append(f"Descripción: {row['descrip']}")

        # Keywords descriptivos
        if pd.notna(row.get('descrip_keywords')) and str(row['descrip_keywords']) != '':
            text_parts.append(f"Palabras clave: {row['descrip_keywords']}")

        # Si no hay texto descriptivo, crear uno básico
        if not text_parts:
            text_parts.append("Propiedad inmobiliaria")

        return "\n".join(text_parts)

    # Crear metadata estructurada para filtros y scoring
    @staticmethod
    def build_structured_metadata(row):
        """
        Extrae datos estructurados para usar en filtros y scoring.
        NO incluir en embeddings de texto.
        ChromaDB no acepta valores None, por lo que los filtramos.
        """
        metadata = {}

        # Datos numéricos (solo si tienen valor válido)
        for col in DataProcessor.NUMERICAL_COLUMNS:
            if col in row and pd.notna(row[col]) and row[col] != 0:
                metadata[col] = float(row[col])

        # Datos categóricos (solo si no están vacíos)
        for col in DataProcessor.CATEGORICAL_COLUMNS:
            if (col in row and
                pd.notna(row[col]) and
                str(row[col]) not in ['No especificado', 'nan', '', 'None']):
                metadata[col] = str(row[col])

        # URL para identificación única
        if 'url_inmueble' in row and pd.notna(row['url_inmueble']):
            metadata['url'] = str(row['url_inmueble'])

        # Calcular métricas derivadas (solo si tenemos datos base)
        if metadata.get('precio') and metadata.get('metros'):
            metadata['precio_por_m2'] = round(metadata['precio'] / metadata['metros'], 2)

        # Score de completitud de datos
        total_fields = len(DataProcessor.NUMERICAL_COLUMNS) + len(DataProcessor.CATEGORICAL_COLUMNS)
        filled_fields = len([v for v in metadata.values() if v is not None and v != ''])
        metadata['completeness_score'] = round(filled_fields / total_fields, 2)

        # IMPORTANTE: Filtrar cualquier valor None que pueda quedar
        metadata = {k: v for k, v in metadata.items() if v is not None and v != ''}

        return metadata

