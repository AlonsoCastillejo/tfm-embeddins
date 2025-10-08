import json

from openai import OpenAI

from .config import Config


class QueryEnhancer:
    """
    Utiliza GPT-3.5-turbo para extraer información estructurada de consultas naturales
    y generar tanto queries semánticas mejoradas como filtros exactos
    """

    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    def parse_query_to_json(self, user_query):
        """
        Convierte consulta natural en JSON estructurado con información
        para búsqueda semántica y filtros exactos
        """

        system_prompt = """Eres un experto en análisis de consultas inmobiliarias. Extrae información estructurada de consultas de usuarios y devuelve un JSON con la siguiente estructura:

{
  "semantic_query": "query mejorada para búsqueda semántica con sinónimos y términos expandidos",
  "filters": {
    "precio_min": null,
    "precio_max": null,
    "habitaciones": null,
    "banos": null,
    "metros_min": null,
    "metros_max": null,
    "tipo": null,
    "localidad": null,
    "barrio": null,
    "distrito": null
  },
  "preferences": {
    "estilo_vida": [],
    "caracteristicas_deseadas": [],
    "ubicacion_tipo": null
  }
}

INSTRUCCIONES:
1. semantic_query: Expande la consulta con sinónimos, términos relacionados y características implícitas
2. filters: Extrae valores exactos para filtrar la base de datos (solo números/texto exacto)
3. preferences: Información cualitativa para mejorar el ranking

EJEMPLOS:

Input: "piso barato madrid con 3 habitaciones"
Output: {
  "semantic_query": "piso económico asequible bajo precio Madrid capital vivienda apartamento 3 habitaciones dormitorios",
  "filters": {
    "precio_min": null,
    "precio_max": 400000,
    "habitaciones": 3,
    "banos": null,
    "metros_min": null,
    "metros_max": null,
    "tipo": "Piso",
    "localidad": "Madrid",
    "barrio": null,
    "distrito": null
  },
  "preferences": {
    "estilo_vida": ["economico", "primera_vivienda"],
    "caracteristicas_deseadas": ["precio_accesible"],
    "ubicacion_tipo": "urbano"
  }
}

Input: "casa moderna para familia con jardín en las afueras bajo 600k"
Output: {
  "semantic_query": "casa moderna contemporánea familia jardín patio exterior zona residencial tranquila niños espacioso",
  "filters": {
    "precio_min": null,
    "precio_max": 600000,
    "habitaciones": null,
    "banos": null,
    "metros_min": 100,
    "metros_max": null,
    "tipo": "Casa",
    "localidad": null,
    "barrio": null,
    "distrito": null
  },
  "preferences": {
    "estilo_vida": ["familiar", "residencial"],
    "caracteristicas_deseadas": ["jardin", "moderna", "espacioso"],
    "ubicacion_tipo": "afueras"
  }
}

IMPORTANTE: Devuelve ÚNICAMENTE el objeto JSON válido, sin texto explicativo, sin markdown, sin comentarios. El JSON debe empezar con { y terminar con }."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            # Extraer y parsear JSON
            content = response.choices[0].message.content.strip()

            # Limpiar posibles caracteres extra y formato markdown
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            # Limpiar caracteres problemáticos
            content = content.strip()

            # Debug: mostrar contenido antes de parsear
            print(f"Contenido JSON a parsear:\n{content}\n")

            # Intentar parsear JSON con múltiples métodos de limpieza
            parsed_query = self._safe_json_parse(content)

            # Validar estructura
            self._validate_parsed_query(parsed_query)

            return parsed_query

        except Exception as e:
            print(f"Error al procesar query con LLM: {e}")
            # Fallback a query simple
            return {
                "semantic_query": user_query,
                "filters": {},
                "preferences": {
                    "estilo_vida": [],
                    "caracteristicas_deseadas": [],
                    "ubicacion_tipo": None,
                },
            }

    def _validate_parsed_query(self, parsed_query):
        """Valida que el JSON tenga la estructura correcta"""
        required_keys = ["semantic_query", "filters", "preferences"]

        for key in required_keys:
            if key not in parsed_query:
                raise ValueError(f"Falta clave requerida: {key}")

        # Limpiar filtros con valores None o vacíos
        if "filters" in parsed_query:
            parsed_query["filters"] = {
                k: v
                for k, v in parsed_query["filters"].items()
                if v is not None and v != ""
            }

    def _safe_json_parse(self, content):
        """Intenta parsear JSON con múltiples métodos de limpieza"""
        import re

        # Método 1: Intentar parsear directo
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error JSON (intento 1): {e}")

        # Método 2: Limpiar comillas problemáticas en strings
        try:
            # Escapar comillas dentro de valores de string
            fixed_content = re.sub(r'("semantic_query"\s*:\s*"[^"]*)"([^"]*")', r'\1\\\2', content)
            fixed_content = re.sub(r'(".*?"\s*:\s*"[^"]*)"([^"]*")', r'\1\\\2', fixed_content)
            return json.loads(fixed_content)
        except json.JSONDecodeError as e:
            print(f"Error JSON (intento 2): {e}")

        # Método 3: Buscar solo la estructura JSON básica
        try:
            # Extraer solo desde el primer { hasta el último }
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                json_part = content[start:end]
                return json.loads(json_part)
        except json.JSONDecodeError as e:
            print(f"Error JSON (intento 3): {e}")

        # Si todos fallan, usar regex para extraer partes importantes
        print("Fallback: extrayendo información con regex...")
        return self._extract_with_regex(content)

    def _extract_with_regex(self, content):
        """Extrae información usando regex como último recurso"""
        import re

        result = {
            "semantic_query": "",
            "filters": {},
            "preferences": {
                "estilo_vida": [],
                "caracteristicas_deseadas": [],
                "ubicacion_tipo": None
            }
        }

        # Extraer semantic_query
        semantic_match = re.search(r'"semantic_query"\s*:\s*"([^"]*)"', content)
        if semantic_match:
            result["semantic_query"] = semantic_match.group(1)

        # Extraer habitaciones
        hab_match = re.search(r'"habitaciones"\s*:\s*(\d+)', content)
        if hab_match:
            result["filters"]["habitaciones"] = int(hab_match.group(1))

        # Extraer tipo
        tipo_match = re.search(r'"tipo"\s*:\s*"([^"]*)"', content)
        if tipo_match:
            result["filters"]["tipo"] = tipo_match.group(1)

        print(f"Fallback exitoso: {result}")
        return result

    def get_enhanced_query_info(self, user_query, show_analysis=True):
        """
        Función principal que devuelve información completa para búsqueda
        """
        parsed = self.parse_query_to_json(user_query)

        if show_analysis:
            self._display_analysis(user_query, parsed)

        return parsed

    def _display_analysis(self, user_query, parsed):
        """Muestra el análisis detallado del LLM por terminal"""

        print(f"\nANÁLISIS LLM DE LA CONSULTA")
        print("=" * 60)
        print(f"Consulta original: '{user_query}'")
        print("-" * 60)

        # Query semántica mejorada
        print(f"Query semántica optimizada:")
        print(f"   '{parsed['semantic_query']}'")

        # Filtros estructurados
        if parsed["filters"]:
            print(f"\nFiltros exactos extraídos:")
            filters_applied = []

            for key, value in parsed["filters"].items():
                if key == "precio_max" and value:
                    filters_applied.append(f"Precio máximo: {value:,}€")
                elif key == "precio_min" and value:
                    filters_applied.append(f"Precio mínimo: {value:,}€")
                elif key == "habitaciones" and value:
                    filters_applied.append(f"Habitaciones: {value}")
                elif key == "banos" and value:
                    filters_applied.append(f"Baños: {value}")
                elif key == "metros_min" and value:
                    filters_applied.append(f"Superficie mínima: {value}m²")
                elif key == "metros_max" and value:
                    filters_applied.append(f"Superficie máxima: {value}m²")
                elif key == "tipo" and value:
                    filters_applied.append(f"Tipo: {value}")
                elif key == "localidad" and value:
                    filters_applied.append(f"Ciudad: {value}")
                elif key == "barrio" and value:
                    filters_applied.append(f"Barrio: {value}")
                elif key == "distrito" and value:
                    filters_applied.append(f"Distrito: {value}")

            if filters_applied:
                for filter_desc in filters_applied:
                    print(f"   • {filter_desc}")
            else:
                print("   • Sin filtros específicos detectados")
        else:
            print(f"\nFiltros exactos: Ninguno detectado")

        # Preferencias cualitativas
        prefs = parsed.get("preferences", {})
        if prefs.get("estilo_vida") or prefs.get("caracteristicas_deseadas"):
            print(f"\nPreferencias detectadas:")

            if prefs.get("estilo_vida"):
                print(f"   Estilo de vida: {', '.join(prefs['estilo_vida'])}")

            if prefs.get("caracteristicas_deseadas"):
                print(
                    f"   Características: {', '.join(prefs['caracteristicas_deseadas'])}"
                )

            if prefs.get("ubicacion_tipo"):
                print(f"   Tipo ubicación: {prefs['ubicacion_tipo']}")

        print("=" * 60)
        print("Iniciando búsqueda con parámetros optimizados...")
        print()

    def test_query_parsing(self, test_queries):
        """Función para probar diferentes consultas y ver el análisis"""
        print("\nMODO TEST - ANÁLISIS DE CONSULTAS")
        print("=" * 70)

        for i, query in enumerate(test_queries, 1):
            print(f"\n--- TEST {i} ---")
            parsed = self.get_enhanced_query_info(query, show_analysis=True)
            input("Presiona Enter para continuar...")  # Pausa entre tests
