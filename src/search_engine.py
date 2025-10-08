import pandas as pd
from .embeddings_manager import EmbeddingsManager
from .database_manager import DatabaseManager
from .query_enhancer import QueryEnhancer
from .config import Config

class PropertySearchEngine:
    
    def __init__(self):
        self.embeddings_manager = EmbeddingsManager()
        self.db_manager = DatabaseManager()
        self.collection = self.db_manager.get_or_create_collection()
        self.query_enhancer = QueryEnhancer()

        # Mapeo de terminos para fallback (si LLM falla)
        self.query_mapping = {
            'chip': 'cheap inexpensive low-cost affordable budget',
            'cheap': 'inexpensive affordable budget low-cost',
            'expensive': 'costly high-end luxury premium pricey',
            'big': 'large spacious roomy vast huge',
            'small': 'compact tiny little cozy',
            'flat': 'apartment unit condo',
            'house': 'home dwelling residence property',
            'near': 'close proximity nearby',
            'center': 'central downtown city-center',
            'modern': 'contemporary new renovated updated'
        }

    # Mejorar consulta con terminos similares
    def enhance_query(self, query):
        enhanced = query.lower()
        
        for term, expansion in self.query_mapping.items():
            if term in enhanced:
                enhanced = enhanced.replace(term, f"{term} {expansion}")
        
        return enhanced
    
    # Calcular puntuación mejorada con metadata estructurada
    def calculate_relevance_score(self, doc, meta, distance, query):
        # Score semántico base
        semantic_score = max(0, 1 - distance)

        # Score de completitud de datos (ya calculado en metadata)
        completeness_score = meta.get('completeness_score', 0.5)

        # Bonificaciones por datos críticos
        critical_bonus = 0
        if meta.get('precio') and meta['precio'] > 0:
            critical_bonus += 0.15
        if meta.get('Habitaciones') and meta['Habitaciones'] > 0:
            critical_bonus += 0.10
        if meta.get('metros') and meta['metros'] > 0:
            critical_bonus += 0.10

        # Bonus por ubicación específica
        location_bonus = 0
        if meta.get('barrio') and meta['barrio'] != 'No especificado':
            location_bonus += 0.05
        if meta.get('distrito') and meta['distrito'] != 'No especificado':
            location_bonus += 0.05

        # Score final ponderado
        final_score = (
            semantic_score * 0.6 +          # Relevancia semántica (texto)
            completeness_score * 0.2 +      # Completitud de datos
            critical_bonus +                # Datos críticos presentes
            location_bonus                  # Información de ubicación
        )

        return min(1.0, max(0, final_score))  # Normalizar entre 0-1
    
    # Buscar propiedades con análisis LLM y filtros estructurados
    def search(self, query, n_results=None):

        if n_results is None:
            n_results = Config.DEFAULT_RESULTS

        # 1. Analizar consulta con LLM
        query_info = self.query_enhancer.get_enhanced_query_info(query)
        semantic_query = query_info['semantic_query']
        filters = query_info['filters']

        # 2. Generar embedding de la query semántica mejorada
        query_embedding = self.embeddings_manager.generate_embedding(semantic_query, use_large_model=True)

        # 3. Buscar en ChromaDB (más resultados para luego filtrar)
        search_results = min(n_results * 3, 30)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_results,
            include=["documents", "metadatas", "distances"]
        )

        if not results['documents'][0]:
            return []

        # 4. Aplicar filtros estructurados
        filtered_results = []
        for doc, meta, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            if self._apply_filters(meta, filters):
                score = self.calculate_relevance_score(doc, meta, distance, query)
                filtered_results.append({
                    'document': doc,
                    'metadata': meta,
                    'distance': distance,
                    'relevance_score': score
                })

        # 5. Si hay pocos resultados con filtros, relajar filtros
        if len(filtered_results) < n_results and filters:
            print(f"⚠️  Solo {len(filtered_results)} resultados con filtros estrictos, relajando criterios...")
            # Buscar sin filtros estrictos
            for doc, meta, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                if not any(result['metadata'].get('url') == meta.get('url') for result in filtered_results):
                    score = self.calculate_relevance_score(doc, meta, distance, query)
                    filtered_results.append({
                        'document': doc,
                        'metadata': meta,
                        'distance': distance,
                        'relevance_score': score
                    })

        # 6. Ordenar por relevancia
        filtered_results.sort(key=lambda x: x['relevance_score'], reverse=True)

        # 7. Eliminar duplicados por URL
        unique_results = []
        seen_urls = set()

        for result in filtered_results:
            url = result['metadata'].get('url', '')
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)

        return unique_results[:n_results]

    def _apply_filters(self, metadata, filters):
        """Aplica filtros exactos a los metadatos"""
        if not filters:
            return True

        # Filtro de precio
        if 'precio_max' in filters:
            precio = metadata.get('precio')
            if precio and precio > filters['precio_max']:
                return False

        if 'precio_min' in filters:
            precio = metadata.get('precio')
            if precio and precio < filters['precio_min']:
                return False

        # Filtro de habitaciones
        if 'habitaciones' in filters:
            habitaciones = metadata.get('Habitaciones')
            if habitaciones and habitaciones != filters['habitaciones']:
                return False

        # Filtro de tipo
        if 'tipo' in filters:
            tipo = metadata.get('tipo', '').lower()
            if filters['tipo'].lower() not in tipo:
                return False

        # Filtro de localidad
        if 'localidad' in filters:
            localidad = metadata.get('localidad', '').lower()
            if filters['localidad'].lower() not in localidad:
                return False

        # Filtro de metros
        if 'metros_min' in filters:
            metros = metadata.get('metros')
            if metros and metros < filters['metros_min']:
                return False

        if 'metros_max' in filters:
            metros = metadata.get('metros')
            if metros and metros > filters['metros_max']:
                return False

        return True
    
    