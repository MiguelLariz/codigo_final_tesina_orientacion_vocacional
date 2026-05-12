"""
=============================================================
  SISTEMA DE RECOMENDACIÓN VOCACIONAL — UAA / CBTis 168
  Red Neuronal con TensorFlow + Scikit-Learn + NumPy

  Autor  : Miguel Ángel Durón Láriz
  Carrera: Ingeniería en Computación Inteligente — UAA

  MEJORAS v2:
  - Entrada: puntuaciones reales 0-12 del Test de Holland
    (ya NO se usa solo el orden; se usan los valores exactos)
  - Vector de entrada: 6 dimensiones Holland + 1 dimensión CBTis
    (total = 7 features) → el modelo distingue Construcción
    de Programación aunque ambas tengan el mismo orden Holland
  - Entrenamiento mixto: sintéticos + datos reales del CSV
=============================================================

INSTALACIÓN:
    pip install tensorflow scikit-learn numpy pandas joblib supabase python-dotenv

USO:
    # 1. Entrenar por primera vez (datos sintéticos)
    python codigo_final.py

    # 2. Re-entrenar con datos reales del CSV
    python codigo_final.py --reentrenar --csv db_entrenamiento_fase1.csv

    # 3. Desde encuesta_vocacional.py (Streamlit)
    from codigo_final import obtener_recomendacion_para_streamlit
    carreras = obtener_recomendacion_para_streamlit(
        scores_holland={"R": 4, "I": 11, "A": 3, "S": 2, "E": 8, "C": 6},
        carrera_cbtis="Programacion",
    )
=============================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ─────────────────────────────────────────────────────────────
# 1. PARÁMETROS GLOBALES
# ─────────────────────────────────────────────────────────────

BOOST_INGENIERIA = 1.35
BOOST_TECNOLOGIA = 1.20
BOOST_OTRAS      = 1.00

# ─────────────────────────────────────────────────────────────
# BOOST CONTEXTUAL POR ESPECIALIDAD CBTis
# Cada especialidad define:
#   "grupos"  → conjunto(s) de carreras UAA que reciben el bonus
#   "factor"  → multiplicador adicional (encima del boost normal)
#   "etiqueta"→ texto explicativo para el estudiante
# El factor se aplica en post-processing, igual que _boost_carrera.
# ─────────────────────────────────────────────────────────────

BOOST_POR_CBTIS = {
    "Programacion": {
        "factor": 1.40,
        "etiqueta": "🖥️ Área de Tecnología e Informática",
        "grupos": {
            # Tecnología / cómputo
            "Ing. en Computación Inteligente",
            "Ing. en Sistemas Computacionales",
            "Lic. en Desarrollo de Videojuegos y Entornos Virtuales",
            "Lic. en Informática y Tecnologías Computacionales",
            "Lic. en Matemáticas Aplicadas",
            "Lic. en Comercio Electrónico",
            # Afines con alta componente I/C
            "Ing. en Electrónica",
            "Ing. Robótica",
            "Ing. en Manufactura y Automatización Industrial",
        },
    },
    "Construccion": {
        "factor": 1.40,
        "etiqueta": "🏗️ Área de Ingeniería y Construcción",
        "grupos": {
            "Ing. Civil",
            "Ing. en Diseño Mecánico",
            "Ing. en Energías Renovables",
            "Ing. Automotriz",
            "Ing. en Manufactura y Automatización Industrial",
            "Lic. en Arquitectura",
            "Lic. en Urbanismo",
            "Ing. Agrónomo",
        },
    },
    "Mecatronica": {
        "factor": 1.40,
        "etiqueta": "🤖 Área de Mecatrónica e Ingeniería",
        "grupos": {
            "Ing. Robótica",
            "Ing. en Manufactura y Automatización Industrial",
            "Ing. en Electrónica",
            "Ing. Automotriz",
            "Ing. en Computación Inteligente",
            "Ing. en Sistemas Computacionales",
            "Ing. en Diseño Mecánico",
            "Ing. en Energías Renovables",
        },
    },
    "Mecanica Industrial": {
        "factor": 1.40,
        "etiqueta": "⚙️ Área de Manufactura e Ingeniería Industrial",
        "grupos": {
            "Ing. en Manufactura y Automatización Industrial",
            "Ing. Industrial Estadístico",
            "Ing. Automotriz",
            "Ing. en Diseño Mecánico",
            "Ing. Robótica",
            "Ing. en Energías Renovables",
            "Ing. Civil",
        },
    },
    "Laboratorista Clinico": {
        "factor": 1.40,
        "etiqueta": "🔬 Área de Ciencias de la Salud y Biociencias",
        "grupos": {
            "Médico Cirujano",
            "Médico Estomatólogo",
            "Médico Veterinario Zootecnista",
            "Lic. en Enfermería",
            "Lic. en Nutrición",
            "Lic. en Optometría",
            "Lic. en Terapia Física",
            "Lic. en Biología",
            "Lic. en Biotecnología",
            "Ing. Bioquímico",
            "Químico Farmacéutico Biólogo",
        },
    },
    "Ofimatica": {
        "factor": 1.40,
        "etiqueta": "📊 Área Administrativa, Contable y de Negocios",
        "grupos": {
            "Contador Público",
            "Lic. en Administración de Empresas",
            "Lic. en Administración Financiera",
            "Lic. en Administración de la Producción y Servicios",
            "Lic. en Administración y Gestión Fiscal de PYMES",
            "Lic. en Logística Empresarial",
            "Lic. en Economía",
            "Lic. en Comercio Internacional",
            "Lic. en Comercio Electrónico",
            "Lic. en Informática y Tecnologías Computacionales",
        },
    },
}


def _boost_cbtis(nombre_carrera: str, carrera_cbtis: str) -> float:
    """
    Devuelve el factor de boost contextual según la especialidad CBTis.
    Si la carrera UAA está en el grupo de afinidad de la especialidad,
    retorna el factor configurado; de lo contrario retorna 1.0.
    """
    if not carrera_cbtis or carrera_cbtis not in BOOST_POR_CBTIS:
        return 1.0
    config = BOOST_POR_CBTIS[carrera_cbtis]
    return config["factor"] if nombre_carrera in config["grupos"] else 1.0


CARRERAS_INGENIERIA = {
    "Ing. Agrónomo",
    "Ing. en Alimentos",
    "Médico Veterinario Zootecnista",
    "Ing. Bioquímico",
    "Ing. en Electrónica",
    "Ing. Industrial Estadístico",
    "Ing. Automotriz",
    "Ing. Biomédica",
    "Ing. en Diseño Mecánico",
    "Ing. en Energías Renovables",
    "Ing. en Manufactura y Automatización Industrial",
    "Ing. Robótica",
    "Ing. Civil",
}

CARRERAS_TECNOLOGIA = {
    "Ing. en Computación Inteligente",
    "Ing. en Sistemas Computacionales",
    "Lic. en Desarrollo de Videojuegos y Entornos Virtuales",
    "Lic. en Informática y Tecnologías Computacionales",
    "Lic. en Matemáticas Aplicadas",
    "Lic. en Biotecnología",
    "Químico Farmacéutico Biólogo",
    "Lic. en Biología",
    "Lic. en Comercio Electrónico",
}

# ─────────────────────────────────────────────────────────────
# 2. CARRERAS UAA — Perfiles Holland de referencia
#    [R, I, A, S, E, C]  → puntuaciones relativas (0-12)
#    Fuente: Clasificación oficial UAA
# ─────────────────────────────────────────────────────────────

CARRERAS_UAA = {
    # ── Centro de Ciencias ───────────────────────────────────
    "Ing. Agrónomo":                                         {"R":11,"I":9,"A":2,"S":5,"E":8,"C":6},
    "Ing. en Alimentos":                                     {"R":8, "I":11,"A":2,"S":5,"E":7,"C":9},
    "Médico Veterinario Zootecnista":                        {"R":10,"I":9,"A":2,"S":7,"E":6,"C":5},
    # ── Centro de Ciencias Básicas ───────────────────────────
    "Ing. Bioquímico":                                       {"R":8, "I":12,"A":2,"S":4,"E":6,"C":9},
    "Ing. en Computación Inteligente":                       {"R":9, "I":12,"A":4,"S":2,"E":7,"C":8},
    "Ing. en Electrónica":                                   {"R":11,"I":8, "A":2,"S":2,"E":7,"C":7},
    "Ing. en Sistemas Computacionales":                      {"R":8, "I":12,"A":3,"S":2,"E":7,"C":9},
    "Ing. Industrial Estadístico":                           {"R":10,"I":7, "A":1,"S":4,"E":9,"C":10},
    "Lic. en Biología":                                      {"R":7, "I":12,"A":4,"S":8,"E":4,"C":6},
    "Lic. en Biotecnología":                                 {"R":7, "I":12,"A":2,"S":5,"E":6,"C":9},
    "Lic. en Desarrollo de Videojuegos y Entornos Virtuales":{"R":6, "I":10,"A":11,"S":2,"E":7,"C":5},
    "Lic. en Informática y Tecnologías Computacionales":     {"R":7, "I":10,"A":3,"S":3,"E":7,"C":11},
    "Lic. en Matemáticas Aplicadas":                         {"R":5, "I":12,"A":2,"S":3,"E":5,"C":11},
    "Químico Farmacéutico Biólogo":                          {"R":7, "I":12,"A":2,"S":7,"E":4,"C":10},
    # ── Centro de Ciencias de la Ingeniería ──────────────────
    "Ing. Automotriz":                                       {"R":12,"I":7, "A":2,"S":2,"E":8,"C":7},
    "Ing. Biomédica":                                        {"R":7, "I":12,"A":3,"S":7,"E":7,"C":6},
    "Ing. en Diseño Mecánico":                               {"R":11,"I":7, "A":8,"S":2,"E":7,"C":5},
    "Ing. en Energías Renovables":                           {"R":10,"I":10,"A":3,"S":5,"E":8,"C":7},
    "Ing. en Manufactura y Automatización Industrial":        {"R":11,"I":7, "A":2,"S":3,"E":9,"C":10},
    "Ing. Robótica":                                         {"R":10,"I":12,"A":4,"S":2,"E":7,"C":8},
    # ── Centro de Ciencias de la Salud ───────────────────────
    "Lic. en Cultura Física y Deporte":                      {"R":8, "I":3, "A":6,"S":11,"E":9,"C":4},
    "Lic. en Enfermería":                                    {"R":6, "I":8, "A":2,"S":12,"E":5,"C":7},
    "Lic. en Nutrición":                                     {"R":4, "I":9, "A":3,"S":11,"E":6,"C":9},
    "Lic. en Optometría":                                    {"R":7, "I":11,"A":2,"S":9,"E":5,"C":8},
    "Lic. en Terapia Física":                                {"R":7, "I":7, "A":3,"S":12,"E":4,"C":7},
    "Médico Cirujano":                                       {"R":6, "I":12,"A":2,"S":10,"E":5,"C":7},
    "Médico Estomatólogo":                                   {"R":9, "I":9, "A":4,"S":8,"E":7,"C":6},
    # ── Centro de Ciencias del Diseño y de la Construcción ───
    "Ing. Civil":                                            {"R":11,"I":7, "A":4,"S":3,"E":8,"C":10},
    "Lic. en Arquitectura":                                  {"R":9, "I":6, "A":11,"S":3,"E":8,"C":5},
    "Lic. en Diseño de Interiores":                          {"R":5, "I":3, "A":12,"S":6,"E":9,"C":5},
    "Lic. en Diseño de Moda en Indumentaria y Textiles":     {"R":4, "I":2, "A":12,"S":6,"E":9,"C":5},
    "Lic. en Diseño Gráfico":                                {"R":5, "I":4, "A":12,"S":6,"E":9,"C":5},
    "Lic. en Diseño Industrial":                             {"R":8, "I":5, "A":11,"S":3,"E":8,"C":4},
    "Lic. en Urbanismo":                                     {"R":3, "I":9, "A":9, "S":8,"E":6,"C":7},
    # ── Centro de Ciencias Económicas y Administrativas ──────
    "Contador Público":                                      {"R":2, "I":7, "A":2,"S":5,"E":9,"C":12},
    "Lic. en Administración de Empresas":                    {"R":3, "I":5, "A":3,"S":8,"E":12,"C":10},
    "Lic. en Administración de la Producción y Servicios":   {"R":6, "I":6, "A":2,"S":5,"E":10,"C":12},
    "Lic. en Administración Financiera":                     {"R":2, "I":8, "A":2,"S":5,"E":10,"C":12},
    "Lic. en Comercio Internacional":                        {"R":2, "I":6, "A":3,"S":7,"E":12,"C":10},
    "Lic. en Economía":                                      {"R":2, "I":11,"A":3,"S":6,"E":8,"C":11},
    "Lic. en Gestión Turística":                             {"R":4, "I":3, "A":7,"S":10,"E":11,"C":8},
    "Lic. en Mercadotecnia":                                 {"R":2, "I":6, "A":9,"S":8,"E":12,"C":5},
    "Lic. en Relaciones Industriales":                       {"R":2, "I":6, "A":4,"S":11,"E":10,"C":8},
    # ── Centro de Ciencias Empresariales ─────────────────────
    "Lic. en Administración y Gestión Fiscal de PYMES":      {"R":2, "I":5, "A":2,"S":7,"E":10,"C":12},
    "Lic. en Agronegocios":                                  {"R":8, "I":5, "A":2,"S":5,"E":12,"C":8},
    "Lic. en Comercio Electrónico":                          {"R":3, "I":8, "A":7,"S":4,"E":12,"C":9},
    "Lic. en Logística Empresarial":                         {"R":5, "I":6, "A":2,"S":4,"E":10,"C":12},
    # ── Centro de Ciencias Sociales y Humanidades ────────────
    "Lic. en Asesoría Psicopedagógica":                      {"R":2, "I":9, "A":6,"S":12,"E":5,"C":7},
    "Lic. en Ciencias Políticas y Administración Pública":   {"R":2, "I":8, "A":4,"S":9,"E":12,"C":7},
    "Lic. en Comunicación Corporativa Estratégica":          {"R":2, "I":5, "A":8,"S":9,"E":12,"C":6},
    "Lic. en Comunicación e Información":                    {"R":2, "I":7, "A":11,"S":9,"E":7,"C":4},
    "Lic. en Derecho":                                       {"R":2, "I":7, "A":5,"S":9,"E":12,"C":8},
    "Lic. en Docencia de Francés y Español":                 {"R":2, "I":7, "A":9,"S":12,"E":5,"C":6},
    "Lic. en Docencia del Idioma Inglés":                    {"R":2, "I":7, "A":9,"S":12,"E":5,"C":6},
    "Lic. en Filosofía":                                     {"R":2, "I":11,"A":9,"S":8,"E":4,"C":5},
    "Lic. en Historia":                                      {"R":2, "I":11,"A":7,"S":7,"E":3,"C":9},
    "Lic. en Psicología":                                    {"R":2, "I":9, "A":7,"S":12,"E":5,"C":5},
    "Lic. en Sociología":                                    {"R":2, "I":10,"A":7,"S":10,"E":4,"C":6},
    "Lic. en Trabajo Social":                                {"R":2, "I":5, "A":3,"S":12,"E":8,"C":7},
    # ── Centro de las Artes y la Cultura ─────────────────────
    "Lic. en Actuación":                                     {"R":3, "I":4, "A":12,"S":10,"E":8,"C":2},
    "Lic. en Artes Cinematográficas y Audiovisuales":        {"R":6, "I":5, "A":12,"S":7,"E":8,"C":2},
    "Lic. en Estudios del Arte y Gestión Cultural":          {"R":2, "I":7, "A":12,"S":8,"E":8,"C":4},
    "Lic. en Letras Hispánicas":                             {"R":2, "I":9, "A":12,"S":7,"E":3,"C":5},
    "Lic. en Música":                                        {"R":5, "I":5, "A":12,"S":7,"E":4,"C":7},
}

# Orden fijo de personalidades Holland
PERSONALIDADES = ["R", "I", "A", "S", "E", "C"]

# Carreras del CBTis 168 y su codificación numérica.
# El valor refleja el perfil técnico de cada especialidad:
#   R  Realista/Mecánico   → construcción, mecatronica, mecánica
#   I  Investigador/Cómputo → programación
#   C  Convencional/Datos  → ofimática
CBTIS_CARRERAS = {
    "Ofimatica":              {"R": 4, "I": 7, "A": 3, "S": 5, "E": 6, "C": 10},
    "Mecanica Industrial":    {"R": 12,"I": 6, "A": 3, "S": 3, "E": 7, "C": 5},
    "Mecatronica":            {"R": 11,"I": 9, "A": 4, "S": 3, "E": 7, "C": 6},
    "Laboratorista Clinico":  {"R": 6, "I": 11,"A": 3, "S": 9, "E": 4, "C": 7},
    "Construccion":           {"R": 12,"I": 5, "A": 5, "S": 3, "E": 8, "C": 7},
    "Programacion":           {"R": 5, "I": 12,"A": 4, "S": 2, "E": 7, "C": 9},
}

# Rutas de los artefactos del modelo
RUTA_MODELO  = "modelo_uaa.keras"
RUTA_SCALER  = "scaler_uaa.pkl"
RUTA_ENCODER = "label_encoder_uaa.pkl"

# Dimensión del vector de entrada:
# 6 scores Holland (R,I,A,S,E,C) + 6 scores del CBTis = 12
INPUT_DIM = 12


# ─────────────────────────────────────────────────────────────
# 3. CONVERSIÓN DE DATOS A VECTOR
# ─────────────────────────────────────────────────────────────

def scores_a_vector(scores_holland: dict, carrera_cbtis: str = None) -> np.ndarray:
    """
    Convierte los scores 0-12 del Test de Holland en un vector
    numérico de INPUT_DIM dimensiones.

    Los primeros 6 valores son los scores Holland del estudiante.
    Los siguientes 6 valores son el perfil Holland de la carrera
    CBTis que cursa (contexto técnico del plantel).

    Si carrera_cbtis no se provee o no es reconocida, los últimos
    6 valores se rellenan con ceros.

    Parámetros
    ----------
    scores_holland : dict  ej. {"R": 4, "I": 11, "A": 3, "S": 2, "E": 8, "C": 6}
    carrera_cbtis  : str   ej. "Programacion"

    Retorna
    -------
    np.ndarray de shape (INPUT_DIM,)
    """
    # Scores Holland del estudiante (normalizar a 0-12)
    holland_vec = np.array([
        float(scores_holland.get(p, 0)) for p in PERSONALIDADES
    ], dtype=float)
    holland_vec = np.clip(holland_vec, 0, 12)

    # Perfil técnico de la carrera CBTis
    if carrera_cbtis and carrera_cbtis in CBTIS_CARRERAS:
        cbtis_vec = np.array([
            float(CBTIS_CARRERAS[carrera_cbtis].get(p, 0))
            for p in PERSONALIDADES
        ], dtype=float)
    else:
        cbtis_vec = np.zeros(6, dtype=float)

    return np.concatenate([holland_vec, cbtis_vec])


def ranking_a_scores(ranking_lista: list) -> dict:
    """
    Convierte una lista ordenada de personalidades Holland
    (orden 1°→6°) a un diccionario de scores aproximados 0-12.

    Útil como fallback cuando solo se tiene el orden (datos antiguos).

    Ejemplo:
        ranking_a_scores(["I","R","E","C","A","S"])
        → {"I":12, "R":10, "E":8, "C":6, "A":4, "S":2}
    """
    pesos = {1: 12, 2: 10, 3: 8, 4: 6, 5: 4, 6: 2}
    return {p: pesos[i + 1] for i, p in enumerate(ranking_lista)}


def carrera_a_vector_referencia(nombre: str) -> np.ndarray:
    """
    Convierte el perfil de referencia de una carrera UAA
    en un vector numpy de 6 dimensiones.
    """
    scores = CARRERAS_UAA[nombre]
    return np.array([float(scores.get(p, 0)) for p in PERSONALIDADES])


def _boost_carrera(nombre: str) -> float:
    if nombre in CARRERAS_INGENIERIA:
        return BOOST_INGENIERIA
    if nombre in CARRERAS_TECNOLOGIA:
        return BOOST_TECNOLOGIA
    return BOOST_OTRAS


# ─────────────────────────────────────────────────────────────
# 4. RECOMENDADOR POR SIMILITUD COSENO (fallback sin NN)
# ─────────────────────────────────────────────────────────────

def _similitud_coseno(v1: np.ndarray, v2: np.ndarray) -> float:
    norma = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norma) if norma > 0 else 0.0


def recomendar_coseno(
    scores_holland: dict,
    carrera_cbtis: str = None,
    top_n: int = 5,
) -> list:
    """
    Recomendador de respaldo basado en similitud coseno.
    Se usa cuando la red neuronal aún no está entrenada.

    Combina la similitud Holland pura con el boost de ingeniería
    y un factor de compatibilidad con la carrera CBTis.
    """
    vec_estudiante = np.array([
        float(scores_holland.get(p, 0)) for p in PERSONALIDADES
    ], dtype=float)

    scores = {}
    for nombre in CARRERAS_UAA:
        vec_carrera = carrera_a_vector_referencia(nombre)
        sim = _similitud_coseno(vec_estudiante, vec_carrera)

        # Bonus CBTis: si la carrera UAA es afín al perfil CBTis
        bonus_cbtis = 1.0
        if carrera_cbtis and carrera_cbtis in CBTIS_CARRERAS:
            vec_cbtis = np.array([
                float(CBTIS_CARRERAS[carrera_cbtis].get(p, 0))
                for p in PERSONALIDADES
            ])
            sim_cbtis = _similitud_coseno(vec_carrera, vec_cbtis)
            bonus_cbtis = 1.0 + 0.25 * sim_cbtis  # hasta +25 %

        scores[nombre] = sim * _boost_carrera(nombre) * bonus_cbtis * _boost_cbtis(nombre, carrera_cbtis)

    max_score = max(scores.values()) if scores else 1.0
    resultados = {n: round((s / max_score) * 100, 1) for n, s in scores.items()}
    top = sorted(resultados.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top


# ─────────────────────────────────────────────────────────────
# 5. GENERACIÓN DEL DATASET SINTÉTICO
# ─────────────────────────────────────────────────────────────

def generar_dataset_sintetico(
    muestras_por_carrera: int = 200,
    ruido: float = 1.2,
    semilla: int = 42,
) -> tuple:
    """
    Genera un dataset sintético para el entrenamiento inicial.

    Cada muestra es el vector de 12 dimensiones:
      - Scores Holland del estudiante (con ruido gaussiano)
      - Perfil Holland de la carrera CBTis más compatible
        (rotando entre las 6 especialidades disponibles)

    Retorna
    -------
    X             : np.ndarray (N, INPUT_DIM)
    y             : np.ndarray (N,)  etiquetas numéricas
    label_encoder : LabelEncoder ajustado
    """
    np.random.seed(semilla)
    X_list, y_list = [], []

    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(CARRERAS_UAA.keys()))

    cbtis_keys = list(CBTIS_CARRERAS.keys())

    for nombre, scores_ref in CARRERAS_UAA.items():
        # Vector de referencia de la carrera
        vec_holland = np.array([
            float(scores_ref.get(p, 0)) for p in PERSONALIDADES
        ], dtype=float)

        etiqueta = label_encoder.transform([nombre])[0]

        if nombre in CARRERAS_INGENIERIA:
            n = int(muestras_por_carrera * 1.5)
        elif nombre in CARRERAS_TECNOLOGIA:
            n = int(muestras_por_carrera * 1.2)
        else:
            n = muestras_por_carrera

        for i in range(n):
            # Ruido en scores Holland del estudiante
            muestra_holland = vec_holland + np.random.normal(0, ruido, size=6)
            muestra_holland = np.clip(muestra_holland, 0, 12)

            # Rotar entre carreras CBTis para que el modelo las aprenda
            cbtis = cbtis_keys[i % len(cbtis_keys)]
            vec_cbtis = np.array([
                float(CBTIS_CARRERAS[cbtis].get(p, 0))
                for p in PERSONALIDADES
            ], dtype=float)

            muestra = np.concatenate([muestra_holland, vec_cbtis])
            X_list.append(muestra)
            y_list.append(etiqueta)

    return np.array(X_list), np.array(y_list), label_encoder


# ─────────────────────────────────────────────────────────────
# 6. CARGA DE DATOS REALES
# ─────────────────────────────────────────────────────────────

def cargar_datos_csv(ruta_csv: str) -> "pd.DataFrame | None":
    if not os.path.exists(ruta_csv):
        print(f"  ⚠️  Archivo no encontrado: {ruta_csv}")
        return None
    df = pd.read_csv(ruta_csv)
    if df.empty:
        print("  ⚠️  El CSV está vacío.")
        return None
    print(f"  ✅ {len(df)} respuestas cargadas desde {ruta_csv}")
    return df


def construir_dataset_real(df: pd.DataFrame) -> "tuple | None":
    """
    Convierte el CSV de respuestas reales en (X, y).

    El CSV puede tener dos formatos:
      - Formato v1 (antiguo): 'ranking_holland' = "I,R,E,C,A,S"
        → se convierten a scores aproximados con ranking_a_scores()
      - Formato v2 (nuevo):   columnas 'score_R', 'score_I', ...
        → se usan directamente

    Solo se agregan muestras cuando el estudiante respondió
    "Si", "Sí" o "Tal vez" a alguna carrera recomendada.
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(CARRERAS_UAA.keys()))
    nombres_validos = set(CARRERAS_UAA.keys())
    POSITIVAS = {"Si", "Sí", "Tal vez", "Tal Vez"}

    X_list, y_list = [], []
    rechazadas = 0

    # Detectar si hay columnas de scores individuales (formato v2)
    tiene_scores = all(f"score_{p}" in df.columns for p in PERSONALIDADES)

    for _, fila in df.iterrows():
        # ─ Obtener scores Holland ─
        if tiene_scores:
            scores = {p: float(fila.get(f"score_{p}", 0)) for p in PERSONALIDADES}
        else:
            # Formato v1: ranking como cadena "I,R,E,C,A,S"
            ranking_str = str(fila.get("ranking_holland", "")).strip()
            partes = [p.strip() for p in ranking_str.split(",")]
            if len(partes) != 6 or not all(p in PERSONALIDADES for p in partes):
                rechazadas += 1
                continue
            scores = ranking_a_scores(partes)

        # ─ Carrera CBTis ─
        carrera_cbtis = str(fila.get("carrera_cbtis", "")).strip()

        # ─ Vector de entrada ─
        vector = scores_a_vector(scores, carrera_cbtis)

        # ─ Evaluar las 3 carreras con feedback del estudiante ─
        for k in range(1, 4):
            carrera  = fila.get(f"carrera_rec_{k}")
            escogida = str(fila.get(f"escogida_{k}", "")).strip()

            if pd.isna(carrera) or carrera not in nombres_validos:
                continue
            if escogida not in POSITIVAS:
                continue

            etiqueta = label_encoder.transform([carrera])[0]
            muestra  = vector + np.random.normal(0, 0.3, size=INPUT_DIM)
            muestra[:6]  = np.clip(muestra[:6], 0, 12)   # Holland: 0-12
            muestra[6:]  = np.clip(muestra[6:], 0, 12)   # CBTis: 0-12
            X_list.append(muestra)
            y_list.append(etiqueta)

    print(f"  ℹ️  Filas: {len(df)}  |  Muestras positivas: {len(X_list)}  |  "
          f"Rechazadas: {rechazadas}")

    if len(X_list) < 10:
        print(f"  ⚠️  Muy pocas muestras reales ({len(X_list)}). "
              "Se reforzará con sintéticos.")
        if len(X_list) == 0:
            return None

    print(f"  ✅ {len(X_list)} muestras reales construidas.")
    return np.array(X_list), np.array(y_list), label_encoder


# ─────────────────────────────────────────────────────────────
# 7. ARQUITECTURA DE LA RED NEURONAL
# ─────────────────────────────────────────────────────────────

def construir_modelo(num_clases: int, input_dim: int = INPUT_DIM):
    """
    Arquitectura de la red neuronal para INPUT_DIM=12.

    Entrada (12)  →  Dense 128 + BN + Dropout 0.30
                  →  Dense 256 + BN + Dropout 0.30
                  →  Dense 128 + BN + Dropout 0.20
                  →  Dense  64
                  →  Softmax (num_clases)

    Se importa keras aquí para no fallar en entornos sin TF.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    modelo = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),

            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.30),

            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.30),

            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.20),

            layers.Dense(64, activation="relu"),

            layers.Dense(num_clases, activation="softmax"),
        ],
        name="recomendador_vocacional_uaa_v2",
    )

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


# ─────────────────────────────────────────────────────────────
# 8. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────

def entrenar(modelo, X_train, y_train, X_val, y_val,
             epocas=120, batch_size=64):
    from tensorflow import keras
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-5, verbose=1,
        ),
    ]
    return modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epocas, batch_size=batch_size,
        callbacks=callbacks, verbose=1,
    )


# ─────────────────────────────────────────────────────────────
# 9. GUARDAR / CARGAR ARTEFACTOS
# ─────────────────────────────────────────────────────────────

def guardar_artefactos(modelo, scaler, label_encoder) -> None:
    modelo.save(RUTA_MODELO)
    joblib.dump(scaler,        RUTA_SCALER)
    joblib.dump(label_encoder, RUTA_ENCODER)
    print(f"\n  💾 Modelo guardado  → {RUTA_MODELO}")
    print(f"  💾 Scaler guardado  → {RUTA_SCALER}")
    print(f"  💾 Encoder guardado → {RUTA_ENCODER}")


def cargar_artefactos() -> tuple:
    from tensorflow import keras
    faltantes = [r for r in [RUTA_MODELO, RUTA_SCALER, RUTA_ENCODER]
                 if not os.path.exists(r)]
    if faltantes:
        raise FileNotFoundError(
            f"Archivos del modelo no encontrados: {faltantes}\n"
            "Ejecuta primero:  python codigo_final.py"
        )
    modelo        = keras.models.load_model(RUTA_MODELO)
    scaler        = joblib.load(RUTA_SCALER)
    label_encoder = joblib.load(RUTA_ENCODER)
    return modelo, scaler, label_encoder


# ─────────────────────────────────────────────────────────────
# 10. MOTOR DE RECOMENDACIÓN NEURONAL
# ─────────────────────────────────────────────────────────────

def recomendar_carreras(
    modelo,
    scaler,
    label_encoder,
    scores_holland: dict,
    carrera_cbtis: str = None,
    top_n: int = 5,
) -> list:
    """
    Genera las top_n recomendaciones usando la red neuronal.

    Pipeline:
      1. Convierte scores Holland + carrera CBTis a vector (12 dims).
      2. Normaliza con el scaler del entrenamiento.
      3. Obtiene probabilidades con la red neuronal.
      4. Aplica boost de ingeniería (post-processing suave).
      5. Re-normaliza y devuelve top_n con porcentaje.

    Parámetros
    ----------
    scores_holland : dict  {"R": 4, "I": 11, "A": 3, "S": 2, "E": 8, "C": 6}
    carrera_cbtis  : str   "Programacion" | "Construccion" | ...
    """
    vector      = scores_a_vector(scores_holland, carrera_cbtis).reshape(1, -1)
    vector_norm = scaler.transform(vector)
    probs       = modelo.predict(vector_norm, verbose=0)[0]

    nombres = label_encoder.classes_
    probs_ajustadas = np.array([
        p * _boost_carrera(n) * _boost_cbtis(n, carrera_cbtis)
        for p, n in zip(probs, nombres)
    ])
    probs_ajustadas /= probs_ajustadas.sum()

    indices_top = np.argsort(probs_ajustadas)[::-1][:top_n]
    return [
        (nombres[i], round(float(probs_ajustadas[i]) * 100, 2))
        for i in indices_top
    ]


# ─────────────────────────────────────────────────────────────
# 11. FUNCIÓN PÚBLICA PARA STREAMLIT
# ─────────────────────────────────────────────────────────────

def obtener_recomendacion_para_streamlit(
    scores_holland: dict,
    carrera_cbtis: str = None,
    top_n: int = 5,
) -> list:
    """
    Función de alto nivel lista para llamar desde encuesta_vocacional.py.

    Parámetros
    ----------
    scores_holland : dict
        Puntuaciones 0-12 del Test de Holland.
        Ej: {"R": 4, "I": 11, "A": 3, "S": 2, "E": 8, "C": 6}
    carrera_cbtis : str
        Especialidad del CBTis que cursa el estudiante.
        Ej: "Programacion", "Construccion", etc.
    top_n : int
        Cuántas carreras devolver (default 5).

    Retorna
    -------
    list[tuple[str, float]]
        [(nombre_carrera, porcentaje), ...]

    Lanza
    -----
    FileNotFoundError si el modelo no está entrenado.
    """
    modelo, scaler, label_encoder = cargar_artefactos()
    return recomendar_carreras(
        modelo, scaler, label_encoder,
        scores_holland, carrera_cbtis, top_n,
    )


# ─────────────────────────────────────────────────────────────
# 12. PIPELINES DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────

def _split_y_normalizar(X, y, usar_stratify=True):
    from collections import Counter
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    conteos = Counter(y.tolist())
    clases_pequeñas = [c for c, n in conteos.items() if n < 2]
    stratify = y if (usar_stratify and len(clases_pequeñas) == 0) else None

    try:
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=stratify
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.176, random_state=42,
            stratify=y_tmp if stratify is not None else None,
        )
    except ValueError:
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.176, random_state=42
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def pipeline_entrenamiento_inicial() -> None:
    """Entrena la red desde cero con datos sintéticos."""
    print("=" * 65)
    print("  SISTEMA DE RECOMENDACIÓN VOCACIONAL UAA v2")
    print("  Entrenamiento inicial con datos sintéticos")
    print(f"  Vector de entrada: {INPUT_DIM} dimensiones "
          "(6 Holland + 6 CBTis)")
    print("=" * 65)

    print(f"\n[1/5] Generando dataset sintético...")
    X, y, label_encoder = generar_dataset_sintetico(
        muestras_por_carrera=200, ruido=1.2, semilla=42
    )
    print(f"      {X.shape[0]:,} muestras  |  {len(CARRERAS_UAA)} carreras")

    print("\n[2/5] Dividiendo y normalizando datos...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
        _split_y_normalizar(X, y)
    print(f"      Train: {len(X_train):,}  Val: {len(X_val):,}  "
          f"Test: {len(X_test):,}")

    print("\n[3/5] Construyendo red neuronal...")
    modelo = construir_modelo(len(CARRERAS_UAA))
    modelo.summary()

    print("\n[4/5] Entrenando...")
    entrenar(modelo, X_train, y_train, X_val, y_val)

    print("\n[5/5] Evaluando...")
    loss, acc = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"      Accuracy: {acc*100:.2f} %   Loss: {loss:.4f}")

    guardar_artefactos(modelo, scaler, label_encoder)
    print("\n  ✅ Entrenamiento inicial completado.")
    _mostrar_demo(modelo, scaler, label_encoder)


def pipeline_reentrenamiento(csv_path: str = None) -> None:
    """
    Re-entrena combinando datos sintéticos + datos reales del CSV.
    Los datos reales se triplican para que tengan mayor peso.
    """
    print("=" * 65)
    print("  SISTEMA DE RECOMENDACIÓN VOCACIONAL UAA v2")
    fuente = f"CSV: {csv_path}" if csv_path else "Supabase"
    print(f"  Re-entrenamiento con datos reales ({fuente})")
    print(f"  Vector de entrada: {INPUT_DIM} dimensiones")
    print("=" * 65)

    # ── Datos reales ──────────────────────────────────────────
    print(f"\n[1/5] Cargando respuestas reales...")
    if csv_path:
        df_real = cargar_datos_csv(csv_path)
    else:
        df_real = _cargar_supabase()

    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(CARRERAS_UAA.keys()))

    X_real, y_real = np.empty((0, INPUT_DIM)), np.empty(0, dtype=int)

    if df_real is not None:
        resultado = construir_dataset_real(df_real)
        if resultado is not None:
            X_real, y_real, label_encoder = resultado

    # ── Datos sintéticos ──────────────────────────────────────
    print("\n[2/5] Generando dataset sintético de soporte...")
    X_sint, y_sint, _ = generar_dataset_sintetico(
        muestras_por_carrera=150, ruido=1.1, semilla=99
    )

    # Combinar: sintéticos + reales ×3
    if len(X_real) > 0:
        X_real_w = np.vstack([X_real] * 3)
        y_real_w = np.concatenate([y_real] * 3)
        X = np.vstack([X_sint, X_real_w])
        y = np.concatenate([y_sint, y_real_w])
        print(f"      Sintéticos: {len(X_sint):,}  |  "
              f"Reales ×3: {len(X_real_w):,}  |  Total: {len(X):,}")
    else:
        X, y = X_sint, y_sint
        print(f"      Solo sintéticos: {len(X):,}")

    print("\n[3/5] Dividiendo y normalizando...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
        _split_y_normalizar(X, y)
    print(f"      Train: {len(X_train):,}  Val: {len(X_val):,}  "
          f"Test: {len(X_test):,}")

    print("\n[4/5] Fine-tuning del modelo...")
    try:
        modelo, _, _ = cargar_artefactos()
        from tensorflow import keras
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print("  → Modelo previo cargado. Aplicando fine-tuning...")
    except (FileNotFoundError, Exception) as e:
        print(f"  → Construyendo desde cero ({e})")
        modelo = construir_modelo(len(CARRERAS_UAA))

    entrenar(modelo, X_train, y_train, X_val, y_val, epocas=100, batch_size=32)

    print("\n[5/5] Evaluando modelo re-entrenado...")
    loss, acc = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"      Accuracy: {acc*100:.2f} %   Loss: {loss:.4f}")

    guardar_artefactos(modelo, scaler, label_encoder)
    print("\n  ✅ Re-entrenamiento completado.")
    _mostrar_demo(modelo, scaler, label_encoder)


def _cargar_supabase():
    try:
        from dotenv import load_dotenv
        from supabase import create_client
        load_dotenv()
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            return None
        supabase = create_client(url, key)
        resp = (
            supabase.table("respuestas_encuesta")
            .select("*")
            .execute()
        )
        if not resp.data:
            return None
        df = pd.DataFrame(resp.data)
        print(f"  ✅ {len(df)} respuestas de Supabase")
        return df
    except Exception as e:
        print(f"  ⚠️  Supabase error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# 13. DEMO
# ─────────────────────────────────────────────────────────────

def _mostrar_demo(modelo, scaler, label_encoder) -> None:
    print("\n" + "=" * 65)
    print("  DEMO — Perfiles típicos CBTis 168 (con puntuaciones reales)")
    print("=" * 65)

    demos = [
        (
            {"R": 5, "I": 11, "A": 4, "S": 2, "E": 7, "C": 9},
            "Programacion",
            "Programación — perfil I alto",
        ),
        (
            {"R": 11, "I": 5, "A": 4, "S": 3, "E": 8, "C": 7},
            "Construccion",
            "Construcción — perfil R alto",
        ),
        (
            {"R": 11, "I": 5, "A": 4, "S": 3, "E": 8, "C": 7},
            "Programacion",
            "MISMO perfil R alto pero CBTis=Programación",
        ),
        (
            {"R": 2, "I": 4, "A": 3, "S": 9, "E": 7, "C": 10},
            "Ofimatica",
            "Ofimática — perfil C/S alto",
        ),
        (
            {"R": 3, "I": 10, "A": 3, "S": 9, "E": 3, "C": 5},
            "Laboratorista Clinico",
            "Laboratorista — perfil I/S",
        ),
    ]

    for scores, cbtis, desc in demos:
        print(f"\n  {desc}")
        print(f"  CBTis: {cbtis}  |  Scores: {scores}")
        recs = recomendar_carreras(
            modelo, scaler, label_encoder,
            scores, cbtis, top_n=3,
        )
        for i, (carrera, pct) in enumerate(recs, 1):
            icono = ("🔧" if carrera in CARRERAS_INGENIERIA
                     else "💻" if carrera in CARRERAS_TECNOLOGIA
                     else "📚")
            barra = "█" * max(1, int(pct / 5))
            print(f"    {i}. {icono} {carrera:<50} {barra} {pct:.1f}%")

    print("\n  🔧 Ingeniería  |  💻 Tecnología  |  📚 Otras")
    print("  El modelo está listo para usar en encuesta_vocacional.py\n")


# ─────────────────────────────────────────────────────────────
# 14. PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sistema de Recomendación Vocacional UAA v2"
    )
    parser.add_argument("--reentrenar", action="store_true",
                        help="Re-entrena con datos reales.")
    parser.add_argument("--csv", type=str, default=None,
                        metavar="ARCHIVO.csv",
                        help="Ruta al CSV con respuestas reales.")
    args = parser.parse_args()

    if args.reentrenar:
        pipeline_reentrenamiento(csv_path=args.csv)
    else:
        pipeline_entrenamiento_inicial()
