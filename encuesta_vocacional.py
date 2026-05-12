"""
encuesta_vocacional.py  —  Sistema de Recomendación Vocacional UAA v2
Autor: Miguel Ángel Durón Láriz

CAMBIOS v2:
  - El estudiante ingresa la puntuación real (0-12) de cada
    personalidad Holland, NO solo el orden de las barras.
  - El vector de entrada al modelo incluye la carrera CBTis
    (Construction ≠ Programacion aunque tengan el mismo orden).
  - Se muestran las 6 puntuaciones capturadas para transparencia.
"""

import os
import streamlit as st
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Test Vocacional CBTis 168",
    layout="centered",
)

from codigo_final import (
    CARRERAS_UAA,
    CARRERAS_INGENIERIA,
    CARRERAS_TECNOLOGIA,
    PERSONALIDADES,
    CBTIS_CARRERAS,
    BOOST_POR_CBTIS,
    scores_a_vector,
    recomendar_coseno,
    obtener_recomendacion_para_streamlit,
)

# ─────────────────────────────────────────────────────────────
# Funciones auxiliares
# ─────────────────────────────────────────────────────────────

def obtener_recomendaciones(
    scores_holland: dict,
    carrera_cbtis: str,
    top_n: int = 5,
) -> tuple:
    """Intenta la red neuronal; si no existe usa similitud coseno."""
    try:
        recs = obtener_recomendacion_para_streamlit(
            scores_holland=scores_holland,
            carrera_cbtis=carrera_cbtis,
            top_n=top_n,
        )
        return recs, "red_neuronal"
    except (FileNotFoundError, Exception):
        recs = recomendar_coseno(
            scores_holland=scores_holland,
            carrera_cbtis=carrera_cbtis,
            top_n=top_n,
        )
        return recs, "coseno"


@st.cache_resource
def get_supabase_client():
    try:
        from supabase import create_client
        try:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
        except Exception:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            url = os.environ.get("SUPABASE_URL")
            key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            return None
        return create_client(url, key)
    except ImportError:
        return None


def guardar_en_supabase(datos: dict) -> bool:
    supabase = get_supabase_client()
    if supabase is None:
        return False
    try:
        supabase.table("respuestas_encuesta").insert(datos).execute()
        return True
    except Exception as e:
        st.warning(f"No se pudo guardar en la base de datos: {e}")
        return False


# Descripciones de cada personalidad Holland
DESCRIPCIONES_HOLLAND = {
    "R": "**Realista (R)** — Actividades prácticas, mecánicas, con herramientas, naturaleza o maquinaria.",
    "I": "**Investigador (I)** — Análisis, ciencia, resolución de problemas complejos, investigación.",
    "A": "**Artístico (A)** — Creatividad, expresión, diseño, música, escritura, artes.",
    "S": "**Social (S)** — Ayuda a otros, enseñanza, trabajo en equipo, servicios de salud.",
    "E": "**Emprendedor (E)** — Liderazgo, negocios, persuasión, ventas, proyectos.",
    "C": "**Convencional (C)** — Organización, datos, procedimientos, finanzas, administración.",
}

# ─────────────────────────────────────────────────────────────
# Estado de sesión
# ─────────────────────────────────────────────────────────────

if "mostrar_resultados" not in st.session_state:
    st.session_state.mostrar_resultados = False

if "evaluacion_guardada" not in st.session_state:
    st.session_state.evaluacion_guardada = False

# ─────────────────────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────────────────────

st.title("Sistema de Recomendación Vocacional")
st.markdown("""
Bienvenido a esta herramienta de orientación vocacional.

Este sistema forma parte de un proyecto de tesina enfocado en desarrollar una
plataforma inteligente que auxilie a los estudiantes en su elección de carrera.
Mediante la recopilación de datos y el entrenamiento de un modelo de red neuronal,
buscamos identificar patrones para generar recomendaciones alineadas a tus intereses
y al perfil del plantel.

Tus datos son completamente anónimos. No te pediremos tu nombre ni datos
que te identifiquen, y tus respuestas se utilizarán exclusivamente con fines
académicos para alimentar y evaluar este algoritmo. Gracias por tu participación.
""")

st.divider()

# ─────────────────────────────────────────────────────────────
# PASO 1 — Especialidad del CBTis
# ─────────────────────────────────────────────────────────────

st.header("1. Cuéntanos sobre ti")

carrera_actual = st.selectbox(
    "¿Qué especialidad estás estudiando actualmente en el CBTis?",
    options=[
        "Selecciona una opción...",
        "Ofimática",
        "Mecánica Industrial",
        "Mecatrónica",
        "Laboratorista Clínico",
        "Construcción",
        "Programación",
    ],
)

# Mapa de nombre display → clave interna
MAPA_CBTIS = {
    "Ofimática":             "Ofimatica",
    "Mecánica Industrial":   "Mecanica Industrial",
    "Mecatrónica":           "Mecatronica",
    "Laboratorista Clínico": "Laboratorista Clinico",
    "Construcción":          "Construccion",
    "Programación":          "Programacion",
}

st.divider()

# ─────────────────────────────────────────────────────────────
# PASO 2 — Test de Holland con puntuaciones reales
# ─────────────────────────────────────────────────────────────

st.header("2. Descubre tu Perfil (Test de Holland)")
st.markdown("""
Por favor, ingresa al siguiente enlace y responde el cuestionario.

**[Ir al Test de los Intereses Profesionales de Holland](https://www.psicoactiva.com/test/autoconocimiento/test-de-los-intereses-profesionales-de-holland/)**

Al finalizar, la página te mostrará un **gráfico de barras** con un puntaje del **0 al 12**
para cada una de las 6 personalidades. Ingresa aquí el valor exacto de cada barra.
""")

st.info(
    "💡 Ingresa el número que aparece en el gráfico de resultados, "
    "no el orden de las barras. El rango va de **0 a 12**."
)

st.subheader("Ingresa tus puntuaciones:")

col1, col2 = st.columns(2)
scores = {}

with col1:
    for p in ["R", "I", "A"]:
        scores[p] = st.number_input(
            DESCRIPCIONES_HOLLAND[p],
            min_value=0, max_value=12, value=0, step=1,
            key=f"score_{p}",
        )

with col2:
    for p in ["S", "E", "C"]:
        scores[p] = st.number_input(
            DESCRIPCIONES_HOLLAND[p],
            min_value=0, max_value=12, value=0, step=1,
            key=f"score_{p}",
        )

# Resumen visual de las puntuaciones ingresadas
total_scores = sum(scores.values())
if total_scores > 0:
    st.markdown("**Vista previa de tu perfil:**")
    for p in PERSONALIDADES:
        barra = "█" * scores[p] + "░" * (12 - scores[p])
        nombre_corto = {
            "R": "Realista     ",
            "I": "Investigador ",
            "A": "Artístico    ",
            "S": "Social       ",
            "E": "Emprendedor  ",
            "C": "Convencional ",
        }[p]
        st.text(f"  {nombre_corto} [{barra}] {scores[p]:2d}/12")

st.divider()

# ─────────────────────────────────────────────────────────────
# PASO 3 — Resultados
# ─────────────────────────────────────────────────────────────

st.header("3. Tus Resultados")

if st.button("Obtener Recomendación Profesional", type="primary"):
    if carrera_actual == "Selecciona una opción...":
        st.warning("Selecciona tu especialidad del CBTis en el Paso 1.")
        st.session_state.mostrar_resultados = False
        st.session_state.evaluacion_guardada = False
    elif total_scores == 0:
        st.warning(
            "Todas tus puntuaciones son 0. "
            "Por favor ingresa los valores del gráfico de Holland en el Paso 2."
        )
        st.session_state.mostrar_resultados = False
        st.session_state.evaluacion_guardada = False
    elif total_scores < 12:
        st.warning(
            f"La suma de tus puntuaciones es {total_scores}. "
            "Asegúrate de ingresar el valor de cada barra del gráfico correctamente."
        )
        st.session_state.mostrar_resultados = False
        st.session_state.evaluacion_guardada = False
    else:
        st.session_state.mostrar_resultados = True
        st.session_state.evaluacion_guardada = False

if st.session_state.mostrar_resultados:

    st.info(
        "Aviso Importante: Los resultados mostrados a continuación son datos "
        "de prueba generados como parte de la investigación de esta tesina. "
        "El modelo de recomendación aún se encuentra en fase de entrenamiento, "
        "por lo que estas sugerencias no deben considerarse como resultados "
        "profesionales o definitivos para tu orientación vocacional."
    )

    # Obtener la clave interna de la carrera CBTis
    cbtis_key = MAPA_CBTIS.get(carrera_actual, None)

    # Ranking de personalidades (para mostrar el código Holland)
    ranking_ordenado = sorted(PERSONALIDADES, key=lambda p: scores[p], reverse=True)
    perfil_top3 = "".join(ranking_ordenado[:3])

    st.success(
        f"Datos procesados. Tu código Holland principal es: **{perfil_top3}**  \n"
        f"Especialidad CBTis: **{carrera_actual}**"
    )

    # ── Banner contextual por especialidad ───────────────────
    if cbtis_key and cbtis_key in BOOST_POR_CBTIS:
        config_boost = BOOST_POR_CBTIS[cbtis_key]
        etiqueta_area = config_boost["etiqueta"]
        carreras_priorizadas = sorted(config_boost["grupos"])
        st.info(
            f"**{etiqueta_area}**  \n"
            f"Como estudiante de **{carrera_actual}**, el sistema prioriza "
            f"carreras afines a tu especialidad técnica. Las siguientes opciones "
            f"reciben un impulso adicional en tu recomendación:\n\n"
            + "\n".join(f"- {c}" for c in carreras_priorizadas)
        )

    # Obtener recomendaciones
    recomendaciones, metodo = obtener_recomendaciones(
        scores_holland=scores,
        carrera_cbtis=cbtis_key,
        top_n=5,
    )
    carreras_sugeridas = [nombre for nombre, _ in recomendaciones]

    if metodo == "coseno":
        st.warning(
            "El modelo de inteligencia artificial aún no ha sido entrenado. "
            "Se están usando recomendaciones basadas en compatibilidad de perfil básica."
        )

    st.markdown("### Carreras Sugeridas para ti")
    contexto_caption = (
        f"Las recomendaciones consideran tu puntuación exacta de Holland, "
        f"el perfil técnico de tu especialidad en el CBTis "
        f"y una priorización de carreras afines a **{carrera_actual}**.  "
        "🔧 Ingeniería | 💻 Tecnología/Cómputo | 📚 Otras áreas"
    )
    st.caption(contexto_caption)

    for nombre, compat in recomendaciones:
        if nombre in CARRERAS_INGENIERIA:
            etiqueta = f"🔧 {nombre}"
        elif nombre in CARRERAS_TECNOLOGIA:
            etiqueta = f"💻 {nombre}"
        else:
            etiqueta = f"📚 {nombre}"

        st.progress(
            min(int(compat), 100),
            text=f"**{etiqueta}** — {compat:.1f}% compatibilidad",
        )

    st.write(
        "Con base en tu perfil, el sistema sugiere las opciones anteriores. "
        "Ayúdanos a evaluar estas recomendaciones preliminares para mejorar el algoritmo:"
    )

    # ─────────────────────────────────────────────────────────
    # Formulario de evaluación
    # ─────────────────────────────────────────────────────────

    with st.form("form_evaluacion"):

        for i, carrera in enumerate(carreras_sugeridas):
            st.markdown(f"#### Opción {i + 1}: {carrera}")
            st.radio(
                f"¿Habías considerado estudiar **{carrera}** antes de esta prueba?",
                options=["Si", "No"],
                key=f"considerada_{i}",
                horizontal=True,
            )
            st.radio(
                f"Conociendo tu perfil, ¿escogerías **{carrera}** como tu carrera?",
                options=["Si", "No", "Tal vez"],
                key=f"escogida_{i}",
                horizontal=True,
            )
            st.divider()

        st.markdown("#### Evaluación General del Sistema")
        st.radio(
            "En general, ¿las recomendaciones te fueron de utilidad?",
            options=[
                "Si, me ayudaron a aclarar mis opciones",
                "Fueron algo útiles, pero aún tengo dudas",
                "No, las opciones no encajan con mis intereses",
            ],
            key="utilidad_general",
        )

        st.write("")
        submit_evaluacion = st.form_submit_button("Guardar Evaluación de Resultados")

        if submit_evaluacion and not st.session_state.evaluacion_guardada:

            datos = {
                "created_at":       datetime.utcnow().isoformat(),
                "carrera_cbtis":    carrera_actual,
                # Puntuaciones reales 0-12 (formato v2)
                "score_R":          int(scores["R"]),
                "score_I":          int(scores["I"]),
                "score_A":          int(scores["A"]),
                "score_S":          int(scores["S"]),
                "score_E":          int(scores["E"]),
                "score_C":          int(scores["C"]),
                # Compatibilidad v1 (ranking como cadena)
                "ranking_holland":  ",".join(ranking_ordenado),
                "perfil_top3":      perfil_top3,
                "metodo_recomend":  metodo,
                "carrera_rec_1":    carreras_sugeridas[0] if len(carreras_sugeridas) > 0 else None,
                "carrera_rec_2":    carreras_sugeridas[1] if len(carreras_sugeridas) > 1 else None,
                "carrera_rec_3":    carreras_sugeridas[2] if len(carreras_sugeridas) > 2 else None,
                "carrera_rec_4":    carreras_sugeridas[3] if len(carreras_sugeridas) > 3 else None,
                "carrera_rec_5":    carreras_sugeridas[4] if len(carreras_sugeridas) > 4 else None,
                "considerada_1":    st.session_state.get("considerada_0"),
                "considerada_2":    st.session_state.get("considerada_1"),
                "considerada_3":    st.session_state.get("considerada_2"),
                "escogida_1":       st.session_state.get("escogida_0"),
                "escogida_2":       st.session_state.get("escogida_1"),
                "escogida_3":       st.session_state.get("escogida_2"),
                "utilidad_general": st.session_state.get("utilidad_general"),
            }

            exito = guardar_en_supabase(datos)

            if exito:
                st.session_state.evaluacion_guardada = True
                st.success(
                    "Gracias por tu retroalimentación. Tus respuestas ayudarán a "
                    "entrenar y mejorar la efectividad de nuestro algoritmo."
                )
            else:
                st.session_state.evaluacion_guardada = True
                st.success(
                    "Gracias por tu retroalimentación. "
                    "Tus respuestas han sido registradas."
                )

        elif st.session_state.evaluacion_guardada:
            st.success("Tu evaluación ya fue guardada. Gracias.")
