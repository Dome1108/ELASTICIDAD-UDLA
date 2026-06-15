# ============================================================
# DASHBOARD DE ELASTICIDAD DE COSTOS
# ============================================================

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
import html
import os
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Elasticidad de costos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# COLORES INSTITUCIONALES
# ============================================================

COLOR_PRINCIPAL = "#A5133D"
COLOR_OSCURO = "#650D28"
COLOR_MEDIO = "#C55B77"
COLOR_CLARO = "#F3CED7"
COLOR_GRIS = "#6B7280"
COLOR_VERDE = "#198754"
COLOR_AMARILLO = "#D89B00"
COLOR_ROJO = "#B42318"


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    f"""
    <style>

        .block-container {{
            padding-top: 4rem;
            padding-bottom: 2rem;
            padding-left: 2.3rem;
            padding-right: 2.3rem;
            max-width: 100%;
        }}

        .titulo-principal {{
            color: {COLOR_PRINCIPAL};
            font-size: 38px;
            font-weight: 800;
            border-left: 6px solid {COLOR_PRINCIPAL};
            padding-left: 14px;
            margin-top: 0;
            margin-bottom: 6px;
            line-height: 1.15;
        }}

        .subtitulo-principal {{
            color: #667085;
            font-size: 14px;
            margin-left: 20px;
            margin-bottom: 18px;
        }}

        .tarjeta {{
            border: 2px solid {COLOR_PRINCIPAL};
            border-radius: 6px;
            padding: 16px 12px;
            min-height: 160px;
            text-align: center;
            background-color: {COLOR_CLARO};
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
        }}

        .tarjeta-titulo {{
            color: #7A3247;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 13px;
            min-height: 20px;
        }}

        .tarjeta-valor {{
            color: #581629;
            font-size: 26px;
            font-weight: 800;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }}

        .tarjeta-subtitulo {{
            color: #85465A;
            font-size: 13px;
            margin-top: 11px;
        }}

        .bloque-informacion {{
            background-color: #FFFFFF;
            border-left: 5px solid {COLOR_PRINCIPAL};
            border-radius: 6px;
            padding: 15px 17px;
            min-height: 110px;
            margin-top: 8px;
            margin-bottom: 10px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
            box-sizing: border-box;
        }}

        .bloque-informacion-titulo {{
            color: {COLOR_PRINCIPAL};
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 8px;
        }}

        .bloque-informacion-texto {{
            color: #344054;
            font-size: 15px;
            line-height: 1.4;
            overflow-wrap: anywhere;
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid #E1E4E8;
            border-radius: 5px;
        }}

        div[data-testid="stMetric"] {{
            background-color: #FFFFFF;
            border: 1px solid #E4E7EC;
            border-radius: 6px;
            padding: 12px;
        }}

        div[data-testid="stSelectbox"] label,
        div[data-testid="stTextInput"] label {{
            color: #7A1E3A;
            font-weight: 650;
        }}

        h1, h2, h3 {{
            color: {COLOR_OSCURO};
        }}

        hr {{
            border-color: #E4E7EC;
        }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FUNCIONES DE RENDERIZADO
# ============================================================

def renderizar_html(contenido: str) -> None:
    """
    Renderiza HTML evitando que Streamlit lo muestre como código.
    """

    if hasattr(st, "html"):
        st.html(contenido)
    else:
        st.markdown(
            contenido,
            unsafe_allow_html=True,
        )


# ============================================================
# FUNCIONES DE RUTAS Y CARGA
# ============================================================

def obtener_carpeta_descargas() -> Path:
    """
    Obtiene la carpeta Descargas del usuario.
    """

    user_profile = os.environ.get("USERPROFILE")

    carpeta_usuario = (
        Path(user_profile)
        if user_profile
        else Path.home()
    )

    carpeta_descargas = carpeta_usuario / "Downloads"

    if not carpeta_descargas.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta Descargas: "
            f"{carpeta_descargas}"
        )

    return carpeta_descargas


def obtener_csv_mas_reciente() -> Path:
    """
    Busca el archivo CSV de elasticidad más reciente
    en la carpeta Descargas.
    """

    carpeta_descargas = obtener_carpeta_descargas()

    archivos = list(
        carpeta_descargas.glob(
            "AdFactElasticidadCostos_*.csv"
        )
    )

    if not archivos:
        raise FileNotFoundError(
            "No se encontraron archivos con el patrón "
            "'AdFactElasticidadCostos_*.csv' en Descargas."
        )

    return max(
        archivos,
        key=lambda archivo: archivo.stat().st_mtime,
    )


def preparar_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normaliza los tipos de datos usados en el dashboard.
    """

    df = df.copy()

    columnas_texto = [
        "AdPeriodo",
        "AdIdentificacion",
        "AdLeadContacto",
        "AdCodCarrera",
        "AdNombreCompleto",
        "AdEmail",
        "AdAsesorNombre",
        "AdAsesorCorreo",
        "AdCarrera",
        "AdCarreraHomologada",
        "AdMotivoPerdida",
        "AdSubMotivoPerdida",
        "AdRangoDeNegociacion",
        "AdCategoriaQuintil",
        "AdNivelSocioec",
        "AdNivelSocioecMercado",
        "AdFuenteCostoMercado",
        "AdConfianzaReferenciaMercado",
        "AdUsoReferenciaMercado",
        "AdUniversidadCompetidoraRef",
        "AdUniversidadCompetidorMinimo",
        "AdResultadoCompetitividadFinal",
        "AdMotivoBecaFinal",
        "AdEstadoIngreso",
    ]

    for columna in columnas_texto:

        if columna in df.columns:

            df[columna] = (
                df[columna]
                .astype("string")
                .str.strip()
                .replace(
                    {
                        "": pd.NA,
                        "nan": pd.NA,
                        "None": pd.NA,
                        "<NA>": pd.NA,
                    }
                )
            )

            if columna in {
                "AdPeriodo",
                "AdIdentificacion",
                "AdLeadContacto",
                "AdCodCarrera",
            }:
                df[columna] = df[columna].str.replace(
                    r"\.0$",
                    "",
                    regex=True,
                )

    columnas_fecha = [
        "AdFechaActualizacion",
        "AudFechaCarga",
    ]

    for columna in columnas_fecha:

        if columna in df.columns:

            df[columna] = pd.to_datetime(
                df[columna],
                errors="coerce",
            )

    columnas_numericas = [
        "AdIndCerrado",
        "AdDiasSinGestion",
        "AdIndContactado",
        "AdIndDocumentado",
        "AdIngresoFamiliaAprox",
        "AdIngresoImputado",
        "AdQuintilIngresoAprox",
        "AdSegmentoCompetenciaImputado",
        "AdRequiereValidarIngreso",
        "AdRequiereRevisionComercial",
        "AdBecaFueAjustada",
        "AdBecaLower",
        "AdBecaRecomendada",
        "AdBecaUpper",
        "AdBecaNecesariaMercado",
        "AdBecaNecesariaCompetidorMinimo",
        "AdBecaFinalSugerida",
        "AdCostoRefMercado",
        "AdCostoCompetidorMinimo",
        "AdMatriculaUDLA",
        "AdArancelUDLA",
        "AdCostoInstitucion",
        "AdCostoUDLAConBecaRecomendada",
        "AdCostoUDLAConBecaUpper",
        "AdCostoUDLAConBecaFinal",
        "AdGapFinalVsMercado",
        "AdGapPostBecaMercado",
        "AdGapPostBecaCompetidorMinimo",
    ]

    for columna in columnas_numericas:

        if columna in df.columns:

            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce",
            )

    return df


@st.cache_data(
    show_spinner="Cargando información local..."
)
def cargar_datos_desde_ruta(
    ruta_archivo: str,
    fecha_modificacion: float,
) -> pd.DataFrame:
    """
    Carga el CSV desde una ruta local.
    """

    del fecha_modificacion

    df = pd.read_csv(
        ruta_archivo,
        low_memory=False,
        encoding="utf-8-sig",
        dtype={
            "AdPeriodo": "string",
            "AdIdentificacion": "string",
            "AdLeadContacto": "string",
            "AdCodCarrera": "string",
        },
    )

    return preparar_dataframe(df)


@st.cache_data(
    show_spinner="Procesando archivo cargado..."
)
def cargar_datos_desde_bytes(
    contenido: bytes,
) -> pd.DataFrame:
    """
    Carga un archivo CSV enviado desde el navegador.
    """

    df = pd.read_csv(
        BytesIO(contenido),
        low_memory=False,
        encoding="utf-8-sig",
        dtype={
            "AdPeriodo": "string",
            "AdIdentificacion": "string",
            "AdLeadContacto": "string",
            "AdCodCarrera": "string",
        },
    )

    return preparar_dataframe(df)


# ============================================================
# FUNCIONES DE FORMATO
# ============================================================

def es_nulo(
    valor: Any,
) -> bool:
    """
    Comprueba si un valor es nulo.
    """

    try:

        resultado = pd.isna(valor)

        if isinstance(
            resultado,
            (bool, np.bool_),
        ):
            return bool(resultado)

        return False

    except Exception:
        return False


def valor_texto(
    valor: Any,
    defecto: str = "Sin información",
) -> str:
    """
    Convierte un valor en texto.
    """

    if es_nulo(valor):
        return defecto

    texto = str(valor).strip()

    if not texto:
        return defecto

    if texto.lower() in {
        "nan",
        "none",
        "<na>",
    }:
        return defecto

    return texto


def formato_moneda(
    valor: Any,
) -> str:
    """
    Formatea un valor monetario.
    """

    if es_nulo(valor):
        return "Sin información"

    try:
        numero = float(valor)

    except (TypeError, ValueError):
        return "Sin información"

    resultado = f"${numero:,.2f}"

    return (
        resultado
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def formato_porcentaje(
    valor: Any,
) -> str:
    """
    Convierte una proporción decimal en porcentaje.
    """

    if es_nulo(valor):
        return "Sin información"

    try:
        numero = float(valor) * 100

    except (TypeError, ValueError):
        return "Sin información"

    resultado = f"{numero:,.2f}%"

    return (
        resultado
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def crear_tarjeta(
    titulo: str,
    valor: str,
    subtitulo: str = "",
) -> None:
    """
    Crea una tarjeta institucional.
    """

    titulo_seguro = html.escape(
        valor_texto(titulo, "")
    )

    valor_seguro = html.escape(
        valor_texto(valor)
    )

    subtitulo_seguro = html.escape(
        valor_texto(subtitulo, "")
    )

    contenido = (
        f'<div class="tarjeta">'
        f'<div class="tarjeta-titulo">'
        f'{titulo_seguro}'
        f'</div>'
        f'<div class="tarjeta-valor">'
        f'{valor_seguro}'
        f'</div>'
        f'<div class="tarjeta-subtitulo">'
        f'{subtitulo_seguro}'
        f'</div>'
        f'</div>'
    )

    renderizar_html(contenido)


def crear_bloque_informacion(
    titulo: str,
    texto: str,
) -> None:
    """
    Crea un bloque informativo.
    """

    titulo_seguro = html.escape(
        valor_texto(titulo, "")
    )

    texto_seguro = html.escape(
        valor_texto(texto)
    )

    contenido = (
        f'<div class="bloque-informacion">'
        f'<div class="bloque-informacion-titulo">'
        f'{titulo_seguro}'
        f'</div>'
        f'<div class="bloque-informacion-texto">'
        f'{texto_seguro}'
        f'</div>'
        f'</div>'
    )

    renderizar_html(contenido)


def opciones_columna(
    dataframe: pd.DataFrame,
    columna: str,
) -> list[str]:
    """
    Devuelve las opciones únicas y ordenadas de una columna.
    """

    if columna not in dataframe.columns:
        return []

    valores = (
        dataframe[columna]
        .dropna()
        .astype(str)
        .str.strip()
    )

    valores = valores[
        valores.ne("")
        & valores.str.lower().ne("nan")
    ]

    return sorted(
        valores.unique().tolist()
    )


def filtrar_por_valor(
    dataframe: pd.DataFrame,
    columna: str,
    valor: str,
    valor_todos: str,
) -> pd.DataFrame:
    """
    Aplica un filtro por igualdad.
    """

    if columna not in dataframe.columns:
        return dataframe

    if valor == valor_todos:
        return dataframe

    mascara = (
        dataframe[columna]
        .astype("string")
        .eq(str(valor))
        .fillna(False)
    )

    return dataframe[mascara]


def obtener_numero(
    registro: pd.Series,
    columna: str,
) -> float | None:
    """
    Obtiene un valor numérico desde una fila.
    """

    if columna not in registro.index:
        return None

    valor = pd.to_numeric(
        pd.Series(
            [registro[columna]]
        ),
        errors="coerce",
    ).iloc[0]

    if pd.isna(valor):
        return None

    return float(valor)


def obtener_primer_valor(
    registro: pd.Series,
    columnas: list[str],
    defecto: str = "Sin información",
) -> str:
    """
    Devuelve el primer valor disponible entre varias columnas.
    """

    for columna in columnas:

        if columna in registro.index:

            valor = valor_texto(
                registro.get(columna),
                "",
            )

            if valor:
                return valor

    return defecto


# ============================================================
# FUNCIONES DE BÚSQUEDA
# ============================================================

def normalizar_texto_busqueda(
    valor: Any,
) -> str:
    """
    Convierte un valor a texto, elimina tildes
    y transforma a minúsculas.
    """

    if es_nulo(valor):
        return ""

    texto = str(valor).strip()

    texto = unicodedata.normalize(
        "NFKD",
        texto,
    )

    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )

    return texto.casefold()


def aplicar_busqueda_general(
    dataframe: pd.DataFrame,
    texto_busqueda: str,
) -> pd.DataFrame:
    """
    Busca simultáneamente en los principales campos
    de cada postulante.
    """

    if not texto_busqueda:
        return dataframe

    if not texto_busqueda.strip():
        return dataframe

    columnas_busqueda = [
        "AdIdentificacion",
        "AdNombreCompleto",
        "AdEmail",
        "AdCarrera",
        "AdCarreraHomologada",
        "AdAsesorNombre",
        "AdAsesorCorreo",
        "AdLeadContacto",
        "AdCodCarrera",
        "AdNivelSocioec",
        "AdRangoDeNegociacion",
    ]

    columnas_disponibles = [
        columna
        for columna in columnas_busqueda
        if columna in dataframe.columns
    ]

    if not columnas_disponibles:
        return dataframe

    consulta_normalizada = (
        normalizar_texto_busqueda(
            texto_busqueda
        )
    )

    texto_por_fila = (
        dataframe[columnas_disponibles]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
        .map(normalizar_texto_busqueda)
    )

    mascara = texto_por_fila.str.contains(
        consulta_normalizada,
        regex=False,
        na=False,
    )

    return dataframe[mascara]


# ============================================================
# CARGA DE DATOS
# ============================================================

st.sidebar.header(
    "Base de elasticidad"
)

archivo_subido = st.sidebar.file_uploader(
    "Cargar archivo CSV",
    type=["csv"],
    help=(
        "Carga el archivo AdFactElasticidadCostos "
        "generado desde Jupyter."
    ),
)

st.sidebar.caption(
    "En ejecución local, si no cargas un archivo, "
    "el sistema buscará automáticamente el CSV "
    "más reciente en la carpeta Descargas."
)

if archivo_subido is not None:

    try:

        contenido_archivo = (
            archivo_subido.getvalue()
        )

        df = cargar_datos_desde_bytes(
            contenido_archivo
        )

        nombre_fuente = archivo_subido.name

        tipo_fuente = (
            "Archivo cargado manualmente"
        )

    except Exception as error:

        st.error(
            "No fue posible leer el archivo cargado."
        )

        st.exception(error)

        st.stop()

else:

    try:

        ruta_csv = obtener_csv_mas_reciente()

        df = cargar_datos_desde_ruta(
            str(ruta_csv),
            ruta_csv.stat().st_mtime,
        )

        nombre_fuente = ruta_csv.name

        tipo_fuente = (
            "Archivo local en Descargas"
        )

    except Exception:

        st.info(
            "No se encontró una base local. "
            "Carga el CSV desde el panel lateral "
            "para visualizar el dashboard."
        )

        st.stop()


if df.empty:

    st.warning(
        "La base cargada no contiene registros."
    )

    st.stop()


# ============================================================
# ENCABEZADO
# ============================================================

renderizar_html(
    '<div class="titulo-principal">'
    'ELASTICIDAD DE COSTOS'
    '</div>'
)

renderizar_html(
    '<div class="subtitulo-principal">'
    f'{html.escape(tipo_fuente)}: '
    f'<strong>{html.escape(nombre_fuente)}</strong>'
    f' · {len(df):,} registros'
    f' · {len(df.columns)} variables'
    '</div>'
)


# ============================================================
# FILTROS
# ============================================================

with st.container(border=True):

    st.markdown(
        "### Filtros de consulta"
    )

    busqueda_general = st.text_input(
        "Buscar postulante",
        placeholder=(
            "Escribe nombre, identificación, correo, carrera, "
            "consultor, código de carrera o lead..."
        ),
        help=(
            "La búsqueda ignora mayúsculas, minúsculas y tildes."
        ),
        key="busqueda_general",
    )

    (
        col_periodo,
        col_asesor,
        col_identificacion,
        col_email,
        col_cerrado,
        col_carrera,
    ) = st.columns(
        [
            1,
            1.8,
            1.4,
            1.8,
            1.15,
            1.6,
        ]
    )

    with col_periodo:

        opciones_periodo = opciones_columna(
            df,
            "AdPeriodo",
        )

        filtro_periodo = st.selectbox(
            "Periodo",
            options=[
                "Todos"
            ] + opciones_periodo,
        )

    with col_asesor:

        opciones_asesor = opciones_columna(
            df,
            "AdAsesorNombre",
        )

        filtro_asesor = st.selectbox(
            "Consultor cierre",
            options=[
                "Todos"
            ] + opciones_asesor,
            disabled=not opciones_asesor,
        )

    with col_identificacion:

        opciones_identificacion = opciones_columna(
            df,
            "AdIdentificacion",
        )

        filtro_identificacion = st.selectbox(
            "Identificación",
            options=[
                "Todas"
            ] + opciones_identificacion,
            disabled=not opciones_identificacion,
        )

    with col_email:

        opciones_email = opciones_columna(
            df,
            "AdEmail",
        )

        filtro_email = st.selectbox(
            "E-mail",
            options=[
                "Todos"
            ] + opciones_email,
            disabled=not opciones_email,
        )

    with col_cerrado:

        filtro_cerrado = st.selectbox(
            "Cartera cerrada",
            options=[
                "Todas",
                "Sí",
                "No",
            ],
        )

    with col_carrera:

        opciones_carrera = opciones_columna(
            df,
            "AdCarrera",
        )

        filtro_carrera = st.selectbox(
            "Carrera",
            options=[
                "Todas"
            ] + opciones_carrera,
            disabled=not opciones_carrera,
        )


# ============================================================
# APLICAR FILTROS
# ============================================================

df_filtrado = df.copy()

df_filtrado = aplicar_busqueda_general(
    df_filtrado,
    busqueda_general,
)

df_filtrado = filtrar_por_valor(
    df_filtrado,
    "AdPeriodo",
    filtro_periodo,
    "Todos",
)

df_filtrado = filtrar_por_valor(
    df_filtrado,
    "AdAsesorNombre",
    filtro_asesor,
    "Todos",
)

df_filtrado = filtrar_por_valor(
    df_filtrado,
    "AdIdentificacion",
    filtro_identificacion,
    "Todas",
)

df_filtrado = filtrar_por_valor(
    df_filtrado,
    "AdEmail",
    filtro_email,
    "Todos",
)

df_filtrado = filtrar_por_valor(
    df_filtrado,
    "AdCarrera",
    filtro_carrera,
    "Todas",
)

if (
    filtro_cerrado != "Todas"
    and "AdIndCerrado" in df_filtrado.columns
):

    valor_cerrado = (
        1
        if filtro_cerrado == "Sí"
        else 0
    )

    valores_cerrados = pd.to_numeric(
        df_filtrado["AdIndCerrado"],
        errors="coerce",
    )

    df_filtrado = df_filtrado[
        valores_cerrados.eq(
            valor_cerrado
        )
    ]


if df_filtrado.empty:

    st.warning(
        "No existen registros para los filtros "
        "o la búsqueda seleccionada."
    )

    st.stop()


# ============================================================
# SELECCIÓN DEL REGISTRO
# ============================================================

df_detalle = (
    df_filtrado
    .reset_index(drop=False)
    .rename(
        columns={
            "index": "_indice_original"
        }
    )
)


def crear_etiqueta_registro(
    fila: pd.Series,
) -> str:
    """
    Construye la etiqueta mostrada en el selector.
    """

    identificacion = valor_texto(
        fila.get(
            "AdIdentificacion"
        ),
        "Sin identificación",
    )

    nombre = valor_texto(
        fila.get(
            "AdNombreCompleto"
        ),
        "Sin nombre",
    )

    carrera = valor_texto(
        fila.get(
            "AdCarrera"
        ),
        "Sin carrera",
    )

    return (
        f"{identificacion} | "
        f"{nombre} | "
        f"{carrera}"
    )


etiquetas_registro = [
    crear_etiqueta_registro(fila)
    for _, fila in df_detalle.iterrows()
]

indice_detalle = st.selectbox(
    "Registro para visualizar el detalle",
    options=list(
        range(
            len(df_detalle)
        )
    ),
    index=0,
    format_func=lambda indice: (
        etiquetas_registro[indice]
    ),
)

registro = df_detalle.iloc[
    indice_detalle
]


# ============================================================
# MÉTRICAS GENERALES
# ============================================================

if "AdIdentificacion" in df_filtrado.columns:

    cantidad_postulantes = (
        df_filtrado[
            "AdIdentificacion"
        ]
        .nunique(
            dropna=True
        )
    )

else:

    cantidad_postulantes = len(
        df_filtrado
    )


if "AdDiasSinGestion" in df_filtrado.columns:

    promedio_dias = pd.to_numeric(
        df_filtrado[
            "AdDiasSinGestion"
        ],
        errors="coerce",
    ).mean()

else:

    promedio_dias = np.nan


nivel_socioeconomico = valor_texto(
    registro.get(
        "AdNivelSocioec"
    )
)

rango_negociacion = valor_texto(
    registro.get(
        "AdRangoDeNegociacion"
    )
)

beca_recomendada = formato_porcentaje(
    registro.get(
        "AdBecaRecomendada"
    )
)

beca_final = formato_porcentaje(
    registro.get(
        "AdBecaFinalSugerida"
    )
)


# ============================================================
# TARJETAS
# ============================================================

(
    tarjeta_1,
    tarjeta_2,
    tarjeta_3,
    tarjeta_4,
    tarjeta_5,
    tarjeta_6,
) = st.columns(6)

with tarjeta_1:

    crear_tarjeta(
        titulo="Nivel socioeconómico",
        valor=nivel_socioeconomico,
        subtitulo="Clasificación del postulante",
    )

with tarjeta_2:

    crear_tarjeta(
        titulo="Rango de negociación",
        valor=rango_negociacion,
        subtitulo="Resultado del modelo",
    )

with tarjeta_3:

    crear_tarjeta(
        titulo="Cantidad de postulantes",
        valor=f"{cantidad_postulantes:,}",
        subtitulo="Según filtros aplicados",
    )

with tarjeta_4:

    crear_tarjeta(
        titulo="Promedio días sin gestión",
        valor=(
            f"{promedio_dias:.0f}"
            if pd.notna(
                promedio_dias
            )
            else "Sin información"
        ),
        subtitulo="Seguimiento comercial",
    )

with tarjeta_5:

    crear_tarjeta(
        titulo="Beca recomendada",
        valor=beca_recomendada,
        subtitulo="Priorización inicial",
    )

with tarjeta_6:

    crear_tarjeta(
        titulo="Beca final sugerida",
        valor=beca_final,
        subtitulo="Ajustada por mercado",
    )


# ============================================================
# PESTAÑAS
# ============================================================

tab_detalle, tab_resumen = st.tabs(
    [
        "Detalle del postulante",
        "Resumen general",
    ]
)


# ============================================================
# PESTAÑA: DETALLE
# ============================================================

with tab_detalle:

    st.subheader(
        "Información comercial"
    )

    (
        columna_estado,
        columna_confianza,
        columna_motivo,
    ) = st.columns(3)

    with columna_estado:

        crear_bloque_informacion(
            "Resultado de competitividad",
            valor_texto(
                registro.get(
                    "AdResultadoCompetitividadFinal"
                )
            ),
        )

    with columna_confianza:

        crear_bloque_informacion(
            "Confianza de la referencia",
            valor_texto(
                registro.get(
                    "AdConfianzaReferenciaMercado"
                )
            ),
        )

    with columna_motivo:

        crear_bloque_informacion(
            "Decisión sobre la beca",
            valor_texto(
                registro.get(
                    "AdMotivoBecaFinal"
                )
            ),
        )

    resultado_competitivo = valor_texto(
        registro.get(
            "AdResultadoCompetitividadFinal"
        )
    )

    resultado_normalizado = (
        resultado_competitivo
        .strip()
        .lower()
    )

    if (
        "iguala" in resultado_normalizado
        or "mejora" in resultado_normalizado
    ):

        st.success(
            "La beca final permite igualar o mejorar "
            "la referencia de mercado."
        )

    elif "por encima" in resultado_normalizado:

        st.warning(
            "La propuesta permanece por encima "
            "de la referencia de mercado."
        )

    else:

        st.info(
            "No existe información suficiente para "
            "determinar la competitividad final."
        )

    st.divider()

    # --------------------------------------------------------
    # COMPARACIÓN DE COSTOS
    # --------------------------------------------------------

    st.subheader(
        "Comparación de costos"
    )

    valores_costos = [
        (
            "Competidor más económico",
            obtener_numero(
                registro,
                "AdCostoCompetidorMinimo",
            ),
        ),
        (
            "Referencia central de mercado",
            obtener_numero(
                registro,
                "AdCostoRefMercado",
            ),
        ),
        (
            "UDLA sin beca",
            obtener_numero(
                registro,
                "AdCostoInstitucion",
            ),
        ),
        (
            "UDLA con beca recomendada",
            obtener_numero(
                registro,
                "AdCostoUDLAConBecaRecomendada",
            ),
        ),
        (
            "UDLA con beca final",
            obtener_numero(
                registro,
                "AdCostoUDLAConBecaFinal",
            ),
        ),
    ]

    datos_costos = pd.DataFrame(
        [
            {
                "Escenario": escenario,
                "Costo": costo,
            }
            for escenario, costo
            in valores_costos
            if costo is not None
        ]
    )

    columna_grafico, columna_detalle_costos = (
        st.columns(
            [
                1.6,
                1,
            ]
        )
    )

    with columna_grafico:

        if datos_costos.empty:

            st.info(
                "No existen valores monetarios suficientes "
                "para construir la comparación."
            )

        else:

            colores_costos = [
                "#7A7F87",
                "#B38B98",
                COLOR_OSCURO,
                COLOR_MEDIO,
                COLOR_PRINCIPAL,
            ]

            figura_costos = go.Figure()

            figura_costos.add_trace(
                go.Bar(
                    x=datos_costos[
                        "Escenario"
                    ],
                    y=datos_costos[
                        "Costo"
                    ],
                    text=[
                        formato_moneda(valor)
                        for valor
                        in datos_costos[
                            "Costo"
                        ]
                    ],
                    textposition="outside",
                    marker_color=(
                        colores_costos[
                            :len(datos_costos)
                        ]
                    ),
                )
            )

            figura_costos.update_layout(
                height=440,
                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=120,
                ),
                yaxis_title="Costo total",
                xaxis_title="",
                template="plotly_white",
                showlegend=False,
            )

            figura_costos.update_yaxes(
                tickprefix="$",
                separatethousands=True,
            )

            st.plotly_chart(
                figura_costos,
                use_container_width=True,
            )

    with columna_detalle_costos:

        st.metric(
            "Matrícula UDLA",
            formato_moneda(
                registro.get(
                    "AdMatriculaUDLA"
                )
            ),
        )

        st.metric(
            "Arancel UDLA",
            formato_moneda(
                registro.get(
                    "AdArancelUDLA"
                )
            ),
        )

        st.metric(
            "Total UDLA sin beca",
            formato_moneda(
                registro.get(
                    "AdCostoInstitucion"
                )
            ),
        )

        st.metric(
            "Total UDLA con beca final",
            formato_moneda(
                registro.get(
                    "AdCostoUDLAConBecaFinal"
                )
            ),
        )

        st.metric(
            "Brecha final frente al mercado",
            formato_moneda(
                registro.get(
                    "AdGapFinalVsMercado"
                )
            ),
        )

    # --------------------------------------------------------
    # REFERENCIA COMPETITIVA
    # --------------------------------------------------------

    st.subheader(
        "Referencia competitiva utilizada"
    )

    universidad_referencia = obtener_primer_valor(
        registro,
        [
            "AdUniversidadCompetidoraRef",
            "AdUniversidadReferenciaMercado",
        ],
    )

    universidad_minima = obtener_primer_valor(
        registro,
        [
            "AdUniversidadCompetidorMinimo",
            "AdUniversidadCompetidoraMin",
            "AdUniversidadCompetidoraRef",
        ],
    )

    referencia_competitiva = pd.DataFrame(
        {
            "Tipo de referencia": [
                "Referencia central",
                "Competidor más económico",
            ],
            "Universidad": [
                universidad_referencia,
                universidad_minima,
            ],
            "Fuente": [
                valor_texto(
                    registro.get(
                        "AdFuenteCostoMercado"
                    )
                ),
                "Precio mínimo disponible",
            ],
            "Costo": [
                obtener_numero(
                    registro,
                    "AdCostoRefMercado",
                ),
                obtener_numero(
                    registro,
                    "AdCostoCompetidorMinimo",
                ),
            ],
        }
    )

    st.dataframe(
        referencia_competitiva,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Costo": (
                st.column_config.NumberColumn(
                    "Costo",
                    format="$ %.2f",
                )
            ),
        },
    )

    # --------------------------------------------------------
    # TABLA DE POSTULANTES
    # --------------------------------------------------------

    st.subheader(
        "Detalle de postulantes"
    )

    mapa_columnas_tabla = {
        "AdAsesorNombre": "Consultor cierre",
        "AdIdentificacion": "Identificación",
        "AdNombreCompleto": "Nombre completo",
        "AdEmail": "E-mail",
        "AdCarrera": "Carrera",
        "AdDiasSinGestion": "Días sin gestión",
        "AdMotivoPerdida": "Motivo pérdida",
        "AdSubMotivoPerdida": "Submotivo pérdida",
        "AdRangoDeNegociacion": "Rango negociación",
        "AdBecaRecomendada": "Beca recomendada",
        "AdBecaLower": "Beca lower",
        "AdBecaUpper": "Beca upper",
        "AdBecaFinalSugerida": "Beca final",
        "AdNivelSocioec": "Nivel socioeconómico",
        "AdCostoRefMercado": "Costo ref. mercado",
        "AdCostoInstitucion": "Costo institución",
        "AdCostoUDLAConBecaFinal": "Costo con beca final",
        "AdResultadoCompetitividadFinal": "Resultado competitivo",
    }

    columnas_tabla_disponibles = [
        columna
        for columna
        in mapa_columnas_tabla
        if columna
        in df_filtrado.columns
    ]

    tabla_postulantes = (
        df_filtrado[
            columnas_tabla_disponibles
        ]
        .copy()
        .rename(
            columns=mapa_columnas_tabla
        )
    )

    columnas_porcentaje_tabla = [
        "Beca recomendada",
        "Beca lower",
        "Beca upper",
        "Beca final",
    ]

    for columna in columnas_porcentaje_tabla:

        if columna in tabla_postulantes.columns:

            tabla_postulantes[
                columna
            ] = (
                pd.to_numeric(
                    tabla_postulantes[
                        columna
                    ],
                    errors="coerce",
                )
                * 100
            )

    configuracion_columnas = {}

    for columna in columnas_porcentaje_tabla:

        if columna in tabla_postulantes.columns:

            configuracion_columnas[
                columna
            ] = (
                st.column_config.NumberColumn(
                    columna,
                    format="%.2f %%",
                )
            )

    for columna in [
        "Costo ref. mercado",
        "Costo institución",
        "Costo con beca final",
    ]:

        if columna in tabla_postulantes.columns:

            configuracion_columnas[
                columna
            ] = (
                st.column_config.NumberColumn(
                    columna,
                    format="$ %.2f",
                )
            )

    if (
        "Días sin gestión"
        in tabla_postulantes.columns
    ):

        configuracion_columnas[
            "Días sin gestión"
        ] = (
            st.column_config.NumberColumn(
                "Días sin gestión",
                format="%.0f",
            )
        )

    st.dataframe(
        tabla_postulantes,
        use_container_width=True,
        hide_index=True,
        height=430,
        column_config=configuracion_columnas,
    )

    st.caption(
        f"Registros filtrados: "
        f"{len(df_filtrado):,} · "
        f"Postulantes distintos: "
        f"{cantidad_postulantes:,}"
    )


# ============================================================
# PESTAÑA: RESUMEN GENERAL
# ============================================================

with tab_resumen:

    st.subheader(
        "Indicadores generales"
    )

    total_registros = len(
        df_filtrado
    )

    serie_beca_ajustada = df_filtrado.get(
        "AdBecaFueAjustada",
        pd.Series(
            0,
            index=df_filtrado.index,
        ),
    )

    total_becas_ajustadas = (
        pd.to_numeric(
            serie_beca_ajustada,
            errors="coerce",
        )
        .fillna(0)
        .eq(1)
        .sum()
    )

    serie_validar_ingreso = df_filtrado.get(
        "AdRequiereValidarIngreso",
        pd.Series(
            0,
            index=df_filtrado.index,
        ),
    )

    total_validar_ingreso = (
        pd.to_numeric(
            serie_validar_ingreso,
            errors="coerce",
        )
        .fillna(0)
        .eq(1)
        .sum()
    )

    serie_revision_comercial = df_filtrado.get(
        "AdRequiereRevisionComercial",
        pd.Series(
            0,
            index=df_filtrado.index,
        ),
    )

    total_revision_comercial = (
        pd.to_numeric(
            serie_revision_comercial,
            errors="coerce",
        )
        .fillna(0)
        .eq(1)
        .sum()
    )

    if (
        "AdCostoRefMercado"
        in df_filtrado.columns
    ):

        cobertura_mercado = (
            df_filtrado[
                "AdCostoRefMercado"
            ]
            .notna()
            .mean()
        )

    else:

        cobertura_mercado = np.nan

    if (
        "AdArancelUDLA"
        in df_filtrado.columns
    ):

        cobertura_tarifa = (
            df_filtrado[
                "AdArancelUDLA"
            ]
            .notna()
            .mean()
        )

    else:

        cobertura_tarifa = np.nan

    (
        metrica_1,
        metrica_2,
        metrica_3,
        metrica_4,
        metrica_5,
        metrica_6,
    ) = st.columns(6)

    metrica_1.metric(
        "Registros",
        f"{total_registros:,}",
    )

    metrica_2.metric(
        "Postulantes",
        f"{cantidad_postulantes:,}",
    )

    metrica_3.metric(
        "Becas ajustadas",
        f"{total_becas_ajustadas:,}",
    )

    metrica_4.metric(
        "Validar ingreso",
        f"{total_validar_ingreso:,}",
    )

    metrica_5.metric(
        "Revisión comercial",
        f"{total_revision_comercial:,}",
    )

    metrica_6.metric(
        "Cobertura de mercado",
        (
            f"{cobertura_mercado:.1%}"
            if pd.notna(
                cobertura_mercado
            )
            else "Sin información"
        ),
    )

    st.caption(
        "Cobertura de tarifas UDLA: "
        + (
            f"{cobertura_tarifa:.1%}"
            if pd.notna(
                cobertura_tarifa
            )
            else "Sin información"
        )
    )

    st.divider()

    columna_grafico_1, columna_grafico_2 = (
        st.columns(2)
    )

    # --------------------------------------------------------
    # RESULTADO COMPETITIVO
    # --------------------------------------------------------

    with columna_grafico_1:

        st.subheader(
            "Resultado competitivo final"
        )

        if (
            "AdResultadoCompetitividadFinal"
            in df_filtrado.columns
        ):

            resumen_resultado = (
                df_filtrado[
                    "AdResultadoCompetitividadFinal"
                ]
                .fillna(
                    "Sin información"
                )
                .value_counts()
                .rename_axis(
                    "Resultado"
                )
                .reset_index(
                    name="Cantidad"
                )
                .sort_values(
                    "Cantidad",
                    ascending=True,
                )
            )

            figura_resultado = px.bar(
                resumen_resultado,
                x="Cantidad",
                y="Resultado",
                orientation="h",
                text="Cantidad",
                color="Resultado",
                color_discrete_sequence=[
                    COLOR_PRINCIPAL,
                    COLOR_MEDIO,
                    "#C9A5AF",
                    "#7A7F87",
                ],
            )

            figura_resultado.update_layout(
                height=430,
                showlegend=False,
                template="plotly_white",
                xaxis_title="Cantidad",
                yaxis_title="",
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20,
                ),
            )

            figura_resultado.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                figura_resultado,
                use_container_width=True,
            )

        else:

            st.info(
                "No está disponible la variable "
                "de resultado competitivo."
            )

    # --------------------------------------------------------
    # NIVEL SOCIOECONÓMICO
    # --------------------------------------------------------

    with columna_grafico_2:

        st.subheader(
            "Distribución socioeconómica"
        )

        if (
            "AdNivelSocioec"
            in df_filtrado.columns
        ):

            resumen_nivel = (
                df_filtrado[
                    "AdNivelSocioec"
                ]
                .fillna(
                    "Sin información"
                )
                .value_counts()
                .rename_axis(
                    "Nivel"
                )
                .reset_index(
                    name="Cantidad"
                )
            )

            figura_nivel = px.pie(
                resumen_nivel,
                names="Nivel",
                values="Cantidad",
                hole=0.45,
                color_discrete_sequence=[
                    COLOR_OSCURO,
                    COLOR_PRINCIPAL,
                    COLOR_MEDIO,
                    "#E6A5B5",
                    "#7A7F87",
                ],
            )

            figura_nivel.update_layout(
                height=430,
                template="plotly_white",
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20,
                ),
                legend_title="",
            )

            st.plotly_chart(
                figura_nivel,
                use_container_width=True,
            )

        else:

            st.info(
                "No está disponible la variable "
                "de nivel socioeconómico."
            )

    columna_grafico_3, columna_grafico_4 = (
        st.columns(2)
    )

    # --------------------------------------------------------
    # TOP DE CARRERAS
    # --------------------------------------------------------

    with columna_grafico_3:

        st.subheader(
            "Carreras con más postulantes"
        )

        if (
            "AdCarrera"
            in df_filtrado.columns
        ):

            resumen_carreras = (
                df_filtrado[
                    "AdCarrera"
                ]
                .fillna(
                    "Sin carrera"
                )
                .value_counts()
                .head(10)
                .rename_axis(
                    "Carrera"
                )
                .reset_index(
                    name="Cantidad"
                )
                .sort_values(
                    "Cantidad",
                    ascending=True,
                )
            )

            figura_carreras = px.bar(
                resumen_carreras,
                x="Cantidad",
                y="Carrera",
                orientation="h",
                text="Cantidad",
                color_discrete_sequence=[
                    COLOR_PRINCIPAL
                ],
            )

            figura_carreras.update_layout(
                height=480,
                template="plotly_white",
                xaxis_title="Cantidad",
                yaxis_title="",
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20,
                ),
            )

            figura_carreras.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                figura_carreras,
                use_container_width=True,
            )

        else:

            st.info(
                "No está disponible la variable carrera."
            )

    # --------------------------------------------------------
    # MOTIVOS DE BECA FINAL
    # --------------------------------------------------------

    with columna_grafico_4:

        st.subheader(
            "Decisión final sobre la beca"
        )

        if (
            "AdMotivoBecaFinal"
            in df_filtrado.columns
        ):

            resumen_motivos = (
                df_filtrado[
                    "AdMotivoBecaFinal"
                ]
                .fillna(
                    "Sin información"
                )
                .value_counts()
                .head(10)
                .rename_axis(
                    "Motivo"
                )
                .reset_index(
                    name="Cantidad"
                )
                .sort_values(
                    "Cantidad",
                    ascending=True,
                )
            )

            figura_motivos = px.bar(
                resumen_motivos,
                x="Cantidad",
                y="Motivo",
                orientation="h",
                text="Cantidad",
                color_discrete_sequence=[
                    COLOR_MEDIO
                ],
            )

            figura_motivos.update_layout(
                height=480,
                template="plotly_white",
                xaxis_title="Cantidad",
                yaxis_title="",
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20,
                ),
            )

            figura_motivos.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                figura_motivos,
                use_container_width=True,
            )

        else:

            st.info(
                "No está disponible el motivo "
                "de la beca final."
            )


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.divider()

st.caption(
    "Dashboard de Elasticidad de Costos · "
    "Dirección de Inteligencia de la Información"
)

