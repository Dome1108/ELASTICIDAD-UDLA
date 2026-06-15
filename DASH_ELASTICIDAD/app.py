# ============================================================
# DASHBOARD DE ELASTICIDAD DE COSTOS
# ============================================================

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
import html
import os

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
COLOR_FONDO = "#F4F5F7"
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
            padding-top: 1.2rem;
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
            margin-bottom: 5px;
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
            border-radius: 5px;
            padding: 16px 12px;
            min-height: 150px;
            text-align: center;
            background-color: {COLOR_CLARO};
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
        }}

        .tarjeta-titulo {{
            color: #7A3247;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 13px;
            min-height: 21px;
        }}

        .tarjeta-valor {{
            color: #581629;
            font-size: 27px;
            font-weight: 800;
            line-height: 1.15;
            word-break: break-word;
        }}

        .tarjeta-subtitulo {{
            color: #85465A;
            font-size: 13px;
            margin-top: 11px;
        }}

        .bloque-informacion {{
            background-color: #FFFFFF;
            border-left: 5px solid {COLOR_PRINCIPAL};
            border-radius: 5px;
            padding: 14px 16px;
            margin-top: 10px;
            margin-bottom: 10px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
        }}

        .bloque-informacion-titulo {{
            color: {COLOR_PRINCIPAL};
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 5px;
        }}

        .bloque-informacion-texto {{
            color: #344054;
            font-size: 15px;
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
        div[data-testid="stMultiSelect"] label {{
            color: #5D172A;
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
# FUNCIONES DE RUTAS Y CARGA
# ============================================================

def obtener_carpeta_descargas() -> Path:
    """
    Obtiene la carpeta Descargas del usuario en Windows.
    """

    user_profile = os.environ.get("USERPROFILE")

    if user_profile:
        carpeta_usuario = Path(user_profile)
    else:
        carpeta_usuario = Path.home()

    carpeta_descargas = carpeta_usuario / "Downloads"

    if not carpeta_descargas.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta Descargas: "
            f"{carpeta_descargas}"
        )

    return carpeta_descargas


def obtener_csv_mas_reciente() -> Path:
    """
    Busca el CSV más reciente de elasticidad en Descargas.
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


def preparar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los principales tipos de datos usados
    por el dashboard.
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
        "AdResultadoCompetitividadFinal",
        "AdMotivoBecaFinal",
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
    ]

    for columna in columnas_numericas:
        if columna in df.columns:
            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce",
            )

    return df


@st.cache_data(show_spinner="Cargando información local...")
def cargar_datos_desde_ruta(
    ruta_archivo: str,
    fecha_modificacion: float,
) -> pd.DataFrame:
    """
    Carga un CSV desde una ruta local.

    fecha_modificacion se utiliza para actualizar
    automáticamente la memoria caché.
    """

    del fecha_modificacion

    df = pd.read_csv(
        ruta_archivo,
        low_memory=False,
        encoding="utf-8-sig",
    )

    return preparar_dataframe(df)


@st.cache_data(show_spinner="Procesando archivo cargado...")
def cargar_datos_desde_bytes(
    contenido: bytes,
) -> pd.DataFrame:
    """
    Carga un CSV enviado mediante file_uploader.
    """

    df = pd.read_csv(
        BytesIO(contenido),
        low_memory=False,
        encoding="utf-8-sig",
    )

    return preparar_dataframe(df)


# ============================================================
# FUNCIONES DE FORMATO
# ============================================================

def es_nulo(valor: Any) -> bool:
    """
    Comprueba si un valor es nulo sin generar errores
    con tipos especiales de pandas.
    """

    try:
        resultado = pd.isna(valor)

        if isinstance(resultado, (bool, np.bool_)):
            return bool(resultado)

        return False

    except Exception:
        return False


def valor_texto(
    valor: Any,
    defecto: str = "Sin información",
) -> str:
    """
    Convierte un valor en texto para mostrarlo.
    """

    if es_nulo(valor):
        return defecto

    texto = str(valor).strip()

    if not texto or texto.lower() in {
        "nan",
        "none",
        "<na>",
    }:
        return defecto

    return texto


def formato_moneda(valor: Any) -> str:
    """
    Formatea valores monetarios con formato latino.
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


def formato_porcentaje(valor: Any) -> str:
    """
    Convierte una proporción decimal a porcentaje.
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
    Muestra una tarjeta con formato institucional.
    """

    titulo_seguro = html.escape(valor_texto(titulo, ""))
    valor_seguro = html.escape(valor_texto(valor))
    subtitulo_seguro = html.escape(valor_texto(subtitulo, ""))

    st.markdown(
        f"""
        <div class="tarjeta">
            <div class="tarjeta-titulo">
                {titulo_seguro}
            </div>

            <div class="tarjeta-valor">
                {valor_seguro}
            </div>

            <div class="tarjeta-subtitulo">
                {subtitulo_seguro}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def crear_bloque_informacion(
    titulo: str,
    texto: str,
) -> None:
    """
    Muestra una sección breve de información.
    """

    st.markdown(
        f"""
        <div class="bloque-informacion">
            <div class="bloque-informacion-titulo">
                {html.escape(valor_texto(titulo, ""))}
            </div>

            <div class="bloque-informacion-texto">
                {html.escape(valor_texto(texto))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def opciones_columna(
    dataframe: pd.DataFrame,
    columna: str,
) -> list[str]:
    """
    Devuelve opciones únicas ordenadas para un filtro.
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
    Aplica un filtro de igualdad a una columna.
    """

    if (
        columna not in dataframe.columns
        or valor == valor_todos
    ):
        return dataframe

    return dataframe[
        dataframe[columna]
        .astype("string")
        .eq(str(valor))
        .fillna(False)
    ]


def obtener_numero(
    registro: pd.Series,
    columna: str,
) -> float | None:
    """
    Obtiene un número desde una fila.
    """

    if columna not in registro.index:
        return None

    valor = pd.to_numeric(
        pd.Series([registro[columna]]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(valor):
        return None

    return float(valor)


# ============================================================
# CARGA DE DATOS
# ============================================================

st.sidebar.header("Base de elasticidad")

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
    "se buscará automáticamente el CSV más reciente "
    "en la carpeta Descargas."
)

if archivo_subido is not None:

    try:
        contenido_archivo = archivo_subido.getvalue()

        df = cargar_datos_desde_bytes(
            contenido_archivo
        )

        nombre_fuente = archivo_subido.name
        tipo_fuente = "Archivo cargado manualmente"

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
        tipo_fuente = "Archivo local en Descargas"

    except Exception:
        st.info(
            "No se encontró una base local. "
            "Carga el CSV desde el panel lateral "
            "para visualizar el dashboard."
        )
        st.stop()


if df.empty:
    st.warning("La base cargada no contiene registros.")
    st.stop()


# ============================================================
# ENCABEZADO
# ============================================================

st.markdown(
    '<div class="titulo-principal">'
    'ELASTICIDAD DE COSTOS'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="subtitulo-principal">
        {html.escape(tipo_fuente)}:
        <strong>{html.escape(nombre_fuente)}</strong>
        · {len(df):,} registros
        · {len(df.columns)} variables
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILTROS
# ============================================================

with st.container(border=True):

    st.markdown(
        "### Filtros de consulta"
    )

    (
        col_periodo,
        col_asesor,
        col_identificacion,
        col_email,
        col_cerrado,
        col_carrera,
    ) = st.columns(
        [1, 1.8, 1.4, 1.8, 1.15, 1.6]
    )

    with col_periodo:

        opciones_periodo = opciones_columna(
            df,
            "AdPeriodo",
        )

        filtro_periodo = st.selectbox(
            "Periodo",
            options=["Todos"] + opciones_periodo,
        )

    with col_asesor:

        opciones_asesor = opciones_columna(
            df,
            "AdAsesorNombre",
        )

        filtro_asesor = st.selectbox(
            "Consultor cierre",
            options=["Todos"] + opciones_asesor,
            disabled=not opciones_asesor,
        )

    with col_identificacion:

        opciones_identificacion = opciones_columna(
            df,
            "AdIdentificacion",
        )

        filtro_identificacion = st.selectbox(
            "Identificación",
            options=["Todas"] + opciones_identificacion,
            disabled=not opciones_identificacion,
        )

    with col_email:

        opciones_email = opciones_columna(
            df,
            "AdEmail",
        )

        filtro_email = st.selectbox(
            "E-mail",
            options=["Todos"] + opciones_email,
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
            options=["Todas"] + opciones_carrera,
            disabled=not opciones_carrera,
        )


# ============================================================
# APLICAR FILTROS
# ============================================================

df_filtrado = df.copy()

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

    df_filtrado = df_filtrado[
        pd.to_numeric(
            df_filtrado["AdIndCerrado"],
            errors="coerce",
        ).eq(valor_cerrado)
    ]


if df_filtrado.empty:
    st.warning(
        "No existen registros para los filtros seleccionados."
    )
    st.stop()


# ============================================================
# SELECCIÓN DEL REGISTRO PARA EL DETALLE
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

    identificacion = valor_texto(
        fila.get("AdIdentificacion"),
        "Sin identificación",
    )

    nombre = valor_texto(
        fila.get("AdNombreCompleto"),
        "Sin nombre",
    )

    carrera = valor_texto(
        fila.get("AdCarrera"),
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
    options=list(range(len(df_detalle))),
    format_func=lambda indice: etiquetas_registro[indice],
)

registro = df_detalle.iloc[indice_detalle]


# ============================================================
# MÉTRICAS GENERALES
# ============================================================

if "AdIdentificacion" in df_filtrado.columns:

    cantidad_postulantes = (
        df_filtrado["AdIdentificacion"]
        .nunique(dropna=True)
    )

else:
    cantidad_postulantes = len(df_filtrado)


if "AdDiasSinGestion" in df_filtrado.columns:

    promedio_dias = pd.to_numeric(
        df_filtrado["AdDiasSinGestion"],
        errors="coerce",
    ).mean()

else:
    promedio_dias = np.nan


nivel_socioeconomico = valor_texto(
    registro.get("AdNivelSocioec")
)

rango_negociacion = valor_texto(
    registro.get("AdRangoDeNegociacion")
)

beca_recomendada = formato_porcentaje(
    registro.get("AdBecaRecomendada")
)

beca_final = formato_porcentaje(
    registro.get("AdBecaFinalSugerida")
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
            if pd.notna(promedio_dias)
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
# PESTAÑAS PRINCIPALES
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

    st.subheader("Información comercial")

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

    if (
        resultado_competitivo
        == "Iguala o mejora la referencia de mercado"
    ):
        st.success(
            "La beca final permite igualar o mejorar "
            "la referencia de mercado."
        )

    elif (
        resultado_competitivo
        == "Permanece por encima de la referencia de mercado"
    ):
        st.warning(
            "La propuesta permanece por encima de la "
            "referencia de mercado."
        )

    else:
        st.info(
            "No existe información suficiente para determinar "
            "la competitividad final de este registro."
        )

    st.divider()

    # --------------------------------------------------------
    # COMPARACIÓN DE COSTOS
    # --------------------------------------------------------

    st.subheader("Comparación de costos")

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
            for escenario, costo in valores_costos
            if costo is not None
        ]
    )

    columna_grafico, columna_detalle_costos = (
        st.columns([1.6, 1])
    )

    with columna_grafico:

        if datos_costos.empty:

            st.info(
                "No existen valores monetarios suficientes "
                "para construir la comparación."
            )

        else:

            figura_costos = go.Figure()

            figura_costos.add_trace(
                go.Bar(
                    x=datos_costos["Escenario"],
                    y=datos_costos["Costo"],
                    text=[
                        formato_moneda(valor)
                        for valor in datos_costos["Costo"]
                    ],
                    textposition="outside",
                    marker_color=[
                        "#7A7F87",
                        "#B38B98",
                        COLOR_OSCURO,
                        COLOR_MEDIO,
                        COLOR_PRINCIPAL,
                    ][: len(datos_costos)],
                )
            )

            figura_costos.update_layout(
                height=430,
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=100,
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

        matricula_udla = registro.get(
            "AdMatriculaUDLA"
        )

        arancel_udla = registro.get(
            "AdArancelUDLA"
        )

        costo_udla = registro.get(
            "AdCostoInstitucion"
        )

        costo_post_beca = registro.get(
            "AdCostoUDLAConBecaFinal"
        )

        gap_final = registro.get(
            "AdGapFinalVsMercado"
        )

        st.metric(
            "Matrícula UDLA",
            formato_moneda(matricula_udla),
        )

        st.metric(
            "Arancel UDLA",
            formato_moneda(arancel_udla),
        )

        st.metric(
            "Total UDLA sin beca",
            formato_moneda(costo_udla),
        )

        st.metric(
            "Total UDLA con beca final",
            formato_moneda(costo_post_beca),
        )

        st.metric(
            "Brecha final frente al mercado",
            formato_moneda(gap_final),
        )

    # --------------------------------------------------------
    # REFERENCIA COMPETITIVA
    # --------------------------------------------------------

    st.subheader("Referencia competitiva utilizada")

    universidad_competidora = valor_texto(
        registro.get(
            "AdUniversidadCompetidoraRef"
        )
    )

    referencia_competitiva = pd.DataFrame(
        {
            "Tipo de referencia": [
                "Referencia central",
                "Competidor más económico",
            ],
            "Universidad": [
                universidad_competidora,
                universidad_competidora,
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
            "Costo": st.column_config.NumberColumn(
                "Costo",
                format="$ %.2f",
            ),
        },
    )

    # --------------------------------------------------------
    # TABLA DE POSTULANTES
    # --------------------------------------------------------

    st.subheader("Detalle de postulantes")

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
        for columna in mapa_columnas_tabla
        if columna in df_filtrado.columns
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

            tabla_postulantes[columna] = (
                pd.to_numeric(
                    tabla_postulantes[columna],
                    errors="coerce",
                )
                * 100
            )

    configuracion_columnas = {}

    for columna in columnas_porcentaje_tabla:

        if columna in tabla_postulantes.columns:

            configuracion_columnas[columna] = (
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

            configuracion_columnas[columna] = (
                st.column_config.NumberColumn(
                    columna,
                    format="$ %.2f",
                )
            )

    if "Días sin gestión" in tabla_postulantes.columns:

        configuracion_columnas[
            "Días sin gestión"
        ] = st.column_config.NumberColumn(
            "Días sin gestión",
            format="%.0f",
        )

    st.dataframe(
        tabla_postulantes,
        use_container_width=True,
        hide_index=True,
        height=430,
        column_config=configuracion_columnas,
    )

    st.caption(
        f"Registros filtrados: {len(df_filtrado):,} · "
        f"Postulantes distintos: {cantidad_postulantes:,}"
    )


# ============================================================
# PESTAÑA: RESUMEN GENERAL
# ============================================================

with tab_resumen:

    st.subheader("Indicadores generales")

    total_registros = len(df_filtrado)

    total_becas_ajustadas = (
        pd.to_numeric(
            df_filtrado.get(
                "AdBecaFueAjustada",
                pd.Series(
                    0,
                    index=df_filtrado.index,
                ),
            ),
            errors="coerce",
        )
        .fillna(0)
        .eq(1)
        .sum()
    )

    total_validar_ingreso = (
        pd.to_numeric(
            df_filtrado.get(
                "AdRequiereValidarIngreso",
                pd.Series(
                    0,
                    index=df_filtrado.index,
                ),
            ),
            errors="coerce",
        )
        .fillna(0)
        .eq(1)
        .sum()
    )

    total_revision_comercial = (
        pd.to_numeric(
            df_filtrado.get(
                "AdRequiereRevisionComercial",
                pd.Series(
                    0,
                    index=df_filtrado.index,
                ),
            ),
            errors="coerce",
        )
        .fillna(0)
        .eq(1)
        .sum()
    )

    cobertura_mercado = (
        df_filtrado["AdCostoRefMercado"]
        .notna()
        .mean()
        if "AdCostoRefMercado" in df_filtrado.columns
        else np.nan
    )

    cobertura_tarifa = (
        df_filtrado["AdArancelUDLA"]
        .notna()
        .mean()
        if "AdArancelUDLA" in df_filtrado.columns
        else np.nan
    )

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
            if pd.notna(cobertura_mercado)
            else "Sin información"
        ),
    )

    st.caption(
        "Cobertura de tarifas UDLA: "
        + (
            f"{cobertura_tarifa:.1%}"
            if pd.notna(cobertura_tarifa)
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
                .fillna("Sin información")
                .value_counts()
                .rename_axis("Resultado")
                .reset_index(name="Cantidad")
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

        if "AdNivelSocioec" in df_filtrado.columns:

            resumen_nivel = (
                df_filtrado["AdNivelSocioec"]
                .fillna("Sin información")
                .value_counts()
                .rename_axis("Nivel")
                .reset_index(name="Cantidad")
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
    # TOP CARRERAS
    # --------------------------------------------------------

    with columna_grafico_3:

        st.subheader(
            "Carreras con más postulantes"
        )

        if "AdCarrera" in df_filtrado.columns:

            resumen_carreras = (
                df_filtrado["AdCarrera"]
                .fillna("Sin carrera")
                .value_counts()
                .head(10)
                .rename_axis("Carrera")
                .reset_index(name="Cantidad")
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
                "No está disponible la carrera."
            )

    # --------------------------------------------------------
    # MOTIVOS DE LA BECA FINAL
    # --------------------------------------------------------

    with columna_grafico_4:

        st.subheader(
            "Decisión final sobre la beca"
        )

        if "AdMotivoBecaFinal" in df_filtrado.columns:

            resumen_motivos = (
                df_filtrado["AdMotivoBecaFinal"]
                .fillna("Sin información")
                .value_counts()
                .head(10)
                .rename_axis("Motivo")
                .reset_index(name="Cantidad")
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
