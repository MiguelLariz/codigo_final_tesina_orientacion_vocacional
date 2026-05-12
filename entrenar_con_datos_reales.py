"""
entrenar_con_datos_reales.py
============================
Script de entrenamiento que usa los datos reales del CSV
db_entrenamiento_fase1.csv para re-entrenar el modelo.

Ejecutar desde la carpeta del proyecto:
    python entrenar_con_datos_reales.py

Requiere TensorFlow instalado:
    pip install tensorflow scikit-learn numpy pandas joblib
"""

import sys
import os

# Asegurarse de que codigo_final está en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from codigo_final import pipeline_reentrenamiento

if __name__ == "__main__":
    csv_path = "db_entrenamiento_fase1.csv"

    if not os.path.exists(csv_path):
        print(f"ERROR: No se encontró el archivo {csv_path}")
        print("Colócalo en la misma carpeta que este script.")
        sys.exit(1)

    pipeline_reentrenamiento(csv_path=csv_path)
