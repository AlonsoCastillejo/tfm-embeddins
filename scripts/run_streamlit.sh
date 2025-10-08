#!/bin/bash
# Script para ejecutar Streamlit con el entorno virtual correcto

echo "🚀 Iniciando aplicación Streamlit..."
echo "📍 Usando entorno virtual: .venv"
echo "📁 Directorio organizado: src/, data/, docs/, scripts/"

# Navegar al directorio raíz del proyecto
cd "$(dirname "$0")/.."

# Activar entorno virtual
source .venv/bin/activate

# Ejecutar con Python del entorno virtual desde la raíz del proyecto
.venv/bin/python -m streamlit run streamlit_app.py

echo "✅ Aplicación finalizada"