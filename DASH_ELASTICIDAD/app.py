from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
import html, os, unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================== CONFIG ==============================
st.set_page_config(
    page_title="Elasticidad de costos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

P = "#A5133D"
D = "#650D28"
M = "#C55B77"
L = "#F3CED7"
G = "#6B7280"
GREEN = "#198754"
YELLOW = "#D89B00"
RED = "#B42318"

st.markdown(
    f"""
<style>
.block-container{{
    padding:4.5rem 2.2rem 2rem;
    max-width:100%
}}
.titulo{{
    color:{P};
    font-size:38px;
    font-weight:800;
    border-left:6px solid {P};
    padding-left:14px;
    margin:0 0 6px;
    line-height:1.15
}}
.subtitulo{{
    color:#667085;
    font-size:14px;
    margin:0 0 18px 20px
}}
.tarjeta{{
    border:2px solid {P};
    border-radius:7px;
    padding:15px 10px;
    min-height:155px;
    text-align:center;
    background:{L};
    box-shadow:0 2px 4px rgba(0,0,0,.08);
    display:flex;
    flex-direction:column;
    justify-content:center;
    box-sizing:border-box
}}
.tarjeta-titulo{{
    color:#7A3247;
    font-size:14px;
    font-weight:700;
    margin-bottom:12px;
    min-height:20px
}}
.tarjeta-valor{{
    color:#581629;
    font-size:25px;
    font-weight:800;
    line-height:1.18;
    overflow-wrap:anywhere
}}
.tarjeta-subtitulo{{
    color:#85465A;
    font-size:13px;
    margin-top:10px
}}
.bloque{{
    background:#fff;
    border-left:5px solid {P};
    border-radius:6px;
    padding:15px 17px;
    min-height:110px;
    margin:8px 0 10px;
    box-shadow:0 1px 4px rgba(0,0,0,.08);
    box-sizing:border-box
}}
.bloque-titulo{{
    color:{P};
    font-weight:700;
    font-size:14px;
    margin-bottom:8px
}}
.bloque-texto{{
    color:#344054;
    font-size:15px;
    line-height:1.4;
    overflow-wrap:anywhere
}}
div[data-testid="stDataFrame"]{{
    border:1px solid #E1E4E8;
    border-radius:5px
}}
div[data-testid="stMetric"]{{
    background:#fff;
    border:1px solid #E4E7EC;
    border-radius:6px;
    padding:12px
}}
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label{{
    color:#7A1E3A;
    font-weight:650
}}
h1,h2,h3{{
    color:{D}
}}
hr{{
    border-color:#E4E7EC
}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================== HELPERS ==============================
def render(s: str) -> None:
    if hasattr(st, "html"):
        st.html(s)
    else:
        st.markdown(s, unsafe_allow_html=True)


def isna(v: Any) -> bool:
    try:
        r = pd.isna(v)
        return bool(r) if isinstance(r, (bool, np.bool_)) else False
    except Exception:
        return False


def txt(v: Any, default="Sin información") -> str:
    if isna(v):
        return default

    s = str(v).strip()

    if not s or s.lower() in {"nan", "none", "<na>"}:
        return default

    return s


def money(v: Any) -> str:
    try:
        s = f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "Sin información"

    return (
        s.replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def pct(v: Any) -> str:
    try:
        s = f"{float(v) * 100:,.2f}%"
    except (TypeError, ValueError):
        return "Sin información"

    return (
        s.replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def card(title, value, subtitle=""):
    render(
        f'<div class="tarjeta">'
        f'<div class="tarjeta-titulo">'
        f'{html.escape(txt(title, ""))}'
        f'</div>'
        f'<div class="tarjeta-valor">'
        f'{html.escape(txt(value))}'
        f'</div>'
        f'<div class="tarjeta-subtitulo">'
        f'{html.escape(txt(subtitle, ""))}'
        f'</div>'
        f'</div>'
    )


def info_block(title, value):
    render(
        f'<div class="bloque">'
        f'<div class="bloque-titulo">'
        f'{html.escape(txt(title, ""))}'
        f'</div>'
        f'<div class="bloque-texto">'
        f'{html.escape(txt(value))}'
        f'</div>'
        f'</div>'
    )


def num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")

    return pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )


def get_num(row: pd.Series, col: str):
    if col not in row.index:
        return None

    v = pd.to_numeric(
        pd.Series([row[col]]),
        errors="coerce",
    ).iloc[0]

    return None if pd.isna(v) else float(v)


def first_col(df: pd.DataFrame, cols: list[str]):
    return next(
        (c for c in cols if c in df.columns),
        None,
    )


def first_value(
    row: pd.Series,
    cols: list[str],
    default="Sin información",
):
    for c in cols:
        if c in row.index and txt(row.get(c), ""):
            return txt(row.get(c), "")

    return default


def norm(v: Any) -> str:
    if isna(v):
        return ""

    s = unicodedata.normalize(
        "NFKD",
        str(v).strip(),
    )

    return "".join(
        c for c in s
        if not unicodedata.combining(c)
    ).casefold()


def contains(
    series: pd.Series,
    text: str,
) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .map(norm)
        .str.contains(
            norm(text),
            regex=False,
            na=False,
        )
    )


def options(df, col):
    if col not in df.columns:
        return []

    s = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    s = s[
        s.ne("")
        & s.str.lower().ne("nan")
    ]

    return sorted(
        s.unique().tolist()
    )


def eq_filter(
    df,
    col,
    value,
    all_value,
):
    if (
        col not in df.columns
        or value == all_value
    ):
        return df

    return df[
        df[col]
        .astype("string")
        .eq(str(value))
        .fillna(False)
    ]


def general_search(df, q):
    if not q or not q.strip():
        return df

    cols = [
        c
        for c in [
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
        if c in df.columns
    ]

    if not cols:
        return df

    text = (
        df[cols]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
        .map(norm)
    )

    return df[
        text.str.contains(
            norm(q),
            regex=False,
            na=False,
        )
    ]


def closed_filter(df, option):
    if (
        option == "Todas"
        or "AdIndCerrado" not in df.columns
    ):
        return df

    raw = df["AdIndCerrado"]

    n = pd.to_numeric(
        raw,
        errors="coerce",
    )

    if n.notna().any():
        objetivo = 1 if option == "Sí" else 0
        return df[n.eq(objetivo)]

    s = (
        raw.fillna("")
        .astype(str)
        .map(norm)
    )

    yes = s.isin(
        {
            norm(v)
            for v in [
                "si",
                "sí",
                "1",
                "true",
                "cerrado",
                "cerrada",
            ]
        }
    )

    return df[yes] if option == "Sí" else df[~yes]


# ============================== DATA ==============================
def downloads() -> Path:
    p = (
        Path(
            os.environ.get(
                "USERPROFILE",
                Path.home(),
            )
        )
        / "Downloads"
    )

    if not p.exists():
        raise FileNotFoundError(
            f"No se encontró Descargas: {p}"
        )

    return p


def newest_csv() -> Path:
    files = list(
        downloads().glob(
            "AdFactElasticidadCostos_*.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No existen archivos "
            "AdFactElasticidadCostos_*.csv "
            "en Descargas"
        )

    return max(
        files,
        key=lambda x: x.stat().st_mtime,
    )


def prepare(df):
    df = df.copy()

    text_cols = [
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

    for c in text_cols:
        if c in df.columns:
            df[c] = (
                df[c]
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

            if c in {
                "AdPeriodo",
                "AdIdentificacion",
                "AdLeadContacto",
                "AdCodCarrera",
            }:
                df[c] = df[c].str.replace(
                    r"\.0$",
                    "",
                    regex=True,
                )

    for c in [
        "AdFechaActualizacion",
        "AudFechaCarga",
    ]:
        if c in df.columns:
            df[c] = pd.to_datetime(
                df[c],
                errors="coerce",
            )

    numeric_cols = [
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

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce",
            )

    return df


@st.cache_data(
    show_spinner="Cargando información local..."
)
def load_path(path, mtime):
    del mtime

    return prepare(
        pd.read_csv(
            path,
            low_memory=False,
            encoding="utf-8-sig",
            dtype={
                "AdPeriodo": "string",
                "AdIdentificacion": "string",
                "AdLeadContacto": "string",
                "AdCodCarrera": "string",
            },
        )
    )


@st.cache_data(
    show_spinner="Procesando archivo cargado..."
)
def load_bytes(content):
    return prepare(
        pd.read_csv(
            BytesIO(content),
            low_memory=False,
            encoding="utf-8-sig",
            dtype={
                "AdPeriodo": "string",
                "AdIdentificacion": "string",
                "AdLeadContacto": "string",
                "AdCodCarrera": "string",
            },
        )
    )


st.sidebar.header(
    "Base de elasticidad"
)

uploaded = st.sidebar.file_uploader(
    "Cargar archivo CSV",
    type=["csv"],
    help=(
        "Carga el archivo "
        "AdFactElasticidadCostos "
        "generado desde Jupyter."
    ),
)

st.sidebar.caption(
    "En local, si no cargas un archivo, "
    "se buscará automáticamente el CSV "
    "más reciente en Descargas."
)

try:
    if uploaded:
        df = load_bytes(
            uploaded.getvalue()
        )

        source_name = uploaded.name
        source_type = (
            "Archivo cargado manualmente"
        )

    else:
        path = newest_csv()

        df = load_path(
            str(path),
            path.stat().st_mtime,
        )

        source_name = path.name
        source_type = (
            "Archivo local en Descargas"
        )

except Exception:
    st.info(
        "No se encontró una base local. "
        "Carga el CSV desde el panel lateral."
    )
    st.stop()

if df.empty:
    st.warning(
        "La base cargada no contiene registros."
    )
    st.stop()


# ============================== HEADER & FILTERS ==============================
render(
    '<div class="titulo">'
    'ELASTICIDAD DE COSTOS'
    '</div>'
)

render(
    f'<div class="subtitulo">'
    f'{html.escape(source_type)}: '
    f'<strong>{html.escape(source_name)}</strong>'
    f' · {len(df):,} registros'
    f' · {len(df.columns)} variables'
    f'</div>'
)

with st.container(border=True):
    st.markdown(
        "### Filtros de consulta"
    )

    search = st.text_input(
        "Buscar postulante",
        placeholder=(
            "Escribe nombre, identificación, "
            "correo, carrera, consultor, "
            "código de carrera o lead..."
        ),
        help=(
            "La búsqueda ignora mayúsculas, "
            "minúsculas y tildes."
        ),
    )

    (
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
    ) = st.columns(
        [1, 1.8, 1.4, 1.8, 1.15, 1.6]
    )

    with c1:
        f_period = st.selectbox(
            "Periodo",
            ["Todos"]
            + options(
                df,
                "AdPeriodo",
            ),
        )

    with c2:
        o = options(
            df,
            "AdAsesorNombre",
        )

        f_advisor = st.selectbox(
            "Consultor cierre",
            ["Todos"] + o,
            disabled=not o,
        )

    with c3:
        o = options(
            df,
            "AdIdentificacion",
        )

        f_id = st.selectbox(
            "Identificación",
            ["Todas"] + o,
            disabled=not o,
        )

    with c4:
        o = options(
            df,
            "AdEmail",
        )

        f_email = st.selectbox(
            "E-mail",
            ["Todos"] + o,
            disabled=not o,
        )

    with c5:
        f_closed = st.selectbox(
            "Cartera cerrada",
            [
                "Todas",
                "Sí",
                "No",
            ],
        )

    with c6:
        o = options(
            df,
            "AdCarrera",
        )

        f_career = st.selectbox(
            "Carrera",
            ["Todas"] + o,
            disabled=not o,
        )


f = general_search(
    df.copy(),
    search,
)

f = eq_filter(
    f,
    "AdPeriodo",
    f_period,
    "Todos",
)

f = eq_filter(
    f,
    "AdAsesorNombre",
    f_advisor,
    "Todos",
)

f = eq_filter(
    f,
    "AdIdentificacion",
    f_id,
    "Todas",
)

f = eq_filter(
    f,
    "AdEmail",
    f_email,
    "Todos",
)

f = eq_filter(
    f,
    "AdCarrera",
    f_career,
    "Todas",
)

f = closed_filter(
    f,
    f_closed,
)

if f.empty:
    st.warning(
        "No existen registros para los filtros "
        "o la búsqueda seleccionada."
    )
    st.stop()


# ============================== SELECTED RECORD & CARDS ==============================
detail = (
    f.reset_index(drop=False)
    .rename(
        columns={
            "index": "_indice_original"
        }
    )
)


def label(row):
    return (
        f'{txt(row.get("AdIdentificacion"), "Sin identificación")}'
        f' | '
        f'{txt(row.get("AdNombreCompleto"), "Sin nombre")}'
        f' | '
        f'{txt(row.get("AdCarrera"), "Sin carrera")}'
    )


labels = [
    label(r)
    for _, r in detail.iterrows()
]

idx = st.selectbox(
    "Registro para visualizar el detalle",
    range(len(detail)),
    index=0,
    format_func=lambda i: labels[i],
)

row = detail.iloc[idx]

if "AdIdentificacion" in f.columns:
    applicants = (
        f["AdIdentificacion"]
        .nunique(dropna=True)
    )
else:
    applicants = len(f)

avg_days = num(
    f,
    "AdDiasSinGestion",
).mean()

cols = st.columns(6)

with cols[0]:
    card(
        "Nivel socioeconómico",
        txt(
            row.get("AdNivelSocioec")
        ),
        "Clasificación del postulante",
    )

with cols[1]:
    card(
        "Rango de negociación",
        txt(
            row.get("AdRangoDeNegociacion")
        ),
        "Resultado del modelo",
    )

with cols[2]:
    card(
        "Cantidad de postulantes",
        f"{applicants:,}",
        "Según filtros aplicados",
    )

with cols[3]:
    card(
        "Promedio días sin gestión",
        (
            f"{avg_days:.0f}"
            if pd.notna(avg_days)
            else "Sin información"
        ),
        "Seguimiento comercial",
    )

with cols[4]:
    card(
        "Beca recomendada",
        pct(
            row.get("AdBecaRecomendada")
        ),
        "Priorización inicial",
    )

with cols[5]:
    card(
        "Beca final sugerida",
        pct(
            row.get("AdBecaFinalSugerida")
        ),
        "Ajustada por mercado",
    )


# ============================== TABS ==============================
(
    t_detail,
    t_summary,
    t_advanced,
) = st.tabs(
    [
        "Detalle del postulante",
        "Resumen general",
        "Análisis avanzado",
    ]
)


# ============================== DETAIL TAB ==============================
with t_detail:
    st.subheader(
        "Información comercial"
    )

    a, b, c = st.columns(3)

    with a:
        info_block(
            "Resultado de competitividad",
            txt(
                row.get(
                    "AdResultadoCompetitividadFinal"
                )
            ),
        )

    with b:
        info_block(
            "Confianza de la referencia",
            txt(
                row.get(
                    "AdConfianzaReferenciaMercado"
                )
            ),
        )

    with c:
        info_block(
            "Decisión sobre la beca",
            txt(
                row.get(
                    "AdMotivoBecaFinal"
                )
            ),
        )

    r = norm(
        txt(
            row.get(
                "AdResultadoCompetitividadFinal"
            )
        )
    )

    if "iguala" in r or "mejora" in r:
        st.success(
            "La beca final permite igualar "
            "o mejorar la referencia de mercado."
        )

    elif "por encima" in r:
        st.warning(
            "La propuesta permanece por encima "
            "de la referencia de mercado."
        )

    else:
        st.info(
            "No existe información suficiente "
            "para determinar la competitividad final."
        )

    st.divider()

    st.subheader(
        "Comparación de costos"
    )

    cost_rows = [
        (
            "Competidor más económico",
            get_num(
                row,
                "AdCostoCompetidorMinimo",
            ),
        ),
        (
            "Referencia central de mercado",
            get_num(
                row,
                "AdCostoRefMercado",
            ),
        ),
        (
            "UDLA sin beca",
            get_num(
                row,
                "AdCostoInstitucion",
            ),
        ),
        (
            "UDLA con beca recomendada",
            get_num(
                row,
                "AdCostoUDLAConBecaRecomendada",
            ),
        ),
        (
            "UDLA con beca final",
            get_num(
                row,
                "AdCostoUDLAConBecaFinal",
            ),
        ),
    ]

    cost_df = pd.DataFrame(
        [
            {
                "Escenario": x,
                "Costo": y,
            }
            for x, y in cost_rows
            if y is not None
        ]
    )

    a, b = st.columns(
        [1.6, 1]
    )

    with a:
        if cost_df.empty:
            st.info(
                "No existen valores monetarios "
                "suficientes para construir "
                "la comparación."
            )

        else:
            fig = go.Figure(
                go.Bar(
                    x=cost_df["Escenario"],
                    y=cost_df["Costo"],
                    text=[
                        money(v)
                        for v in cost_df["Costo"]
                    ],
                    textposition="outside",
                    marker_color=[
                        "#7A7F87",
                        "#B38B98",
                        D,
                        M,
                        P,
                    ][:len(cost_df)],
                )
            )

            fig.update_layout(
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

            fig.update_yaxes(
                tickprefix="$",
                separatethousands=True,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    with b:
        st.metric(
            "Matrícula UDLA",
            money(
                row.get(
                    "AdMatriculaUDLA"
                )
            ),
        )

        st.metric(
            "Arancel UDLA",
            money(
                row.get(
                    "AdArancelUDLA"
                )
            ),
        )

        st.metric(
            "Total UDLA sin beca",
            money(
                row.get(
                    "AdCostoInstitucion"
                )
            ),
        )

        st.metric(
            "Total UDLA con beca final",
            money(
                row.get(
                    "AdCostoUDLAConBecaFinal"
                )
            ),
        )

        st.metric(
            "Brecha final frente al mercado",
            money(
                row.get(
                    "AdGapFinalVsMercado"
                )
            ),
        )

    st.subheader(
        "Referencia competitiva utilizada"
    )

    ref_u = first_value(
        row,
        [
            "AdUniversidadCompetidoraRef",
            "AdUniversidadReferenciaMercado",
        ],
    )

    min_u = first_value(
        row,
        [
            "AdUniversidadCompetidorMinimo",
            "AdUniversidadCompetidoraMin",
            "AdUniversidadCompetidoraRef",
        ],
    )

    refs = pd.DataFrame(
        {
            "Tipo de referencia": [
                "Referencia central",
                "Competidor más económico",
            ],
            "Universidad": [
                ref_u,
                min_u,
            ],
            "Fuente": [
                txt(
                    row.get(
                        "AdFuenteCostoMercado"
                    )
                ),
                "Precio mínimo disponible",
            ],
            "Costo": [
                get_num(
                    row,
                    "AdCostoRefMercado",
                ),
                get_num(
                    row,
                    "AdCostoCompetidorMinimo",
                ),
            ],
        }
    )

    st.dataframe(
        refs,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Costo": (
                st.column_config.NumberColumn(
                    "Costo",
                    format="$ %.2f",
                )
            )
        },
    )

    st.subheader(
        "Detalle de postulantes"
    )

    mapping = {
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
        "AdCostoUDLAConBecaFinal": (
            "Costo con beca final"
        ),
        "AdResultadoCompetitividadFinal": (
            "Resultado competitivo"
        ),
    }

    table = (
        f[
            [
                x
                for x in mapping
                if x in f.columns
            ]
        ]
        .copy()
        .rename(columns=mapping)
    )

    pcols = [
        "Beca recomendada",
        "Beca lower",
        "Beca upper",
        "Beca final",
    ]

    for c in pcols:
        if c in table.columns:
            table[c] = (
                pd.to_numeric(
                    table[c],
                    errors="coerce",
                )
                * 100
            )

    config = {
        c: st.column_config.NumberColumn(
            c,
            format="%.2f %%",
        )
        for c in pcols
        if c in table.columns
    }

    for c in [
        "Costo ref. mercado",
        "Costo institución",
        "Costo con beca final",
    ]:
        if c in table.columns:
            config[c] = (
                st.column_config.NumberColumn(
                    c,
                    format="$ %.2f",
                )
            )

    if "Días sin gestión" in table.columns:
        config["Días sin gestión"] = (
            st.column_config.NumberColumn(
                "Días sin gestión",
                format="%.0f",
            )
        )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=430,
        column_config=config,
    )

    st.caption(
        f"Registros filtrados: {len(f):,} · "
        f"Postulantes distintos: {applicants:,}"
    )


# ============================== SUMMARY TAB ==============================
with t_summary:
    st.subheader(
        "Indicadores generales"
    )

    adjusted = (
        num(
            f,
            "AdBecaFueAjustada",
        )
        .fillna(0)
        .eq(1)
        .sum()
    )

    validate = (
        num(
            f,
            "AdRequiereValidarIngreso",
        )
        .fillna(0)
        .eq(1)
        .sum()
    )

    review = (
        num(
            f,
            "AdRequiereRevisionComercial",
        )
        .fillna(0)
        .eq(1)
        .sum()
    )

    market_cov = (
        f["AdCostoRefMercado"]
        .notna()
        .mean()
        if "AdCostoRefMercado" in f.columns
        else np.nan
    )

    tariff_cov = (
        f["AdArancelUDLA"]
        .notna()
        .mean()
        if "AdArancelUDLA" in f.columns
        else np.nan
    )


    m = st.columns(6)

    m[0].metric(
        "Registros",
        f"{len(f):,}",
    )

    m[1].metric(
        "Postulantes",
        f"{applicants:,}",
    )

    m[2].metric(
        "Becas ajustadas",
        f"{adjusted:,}",
    )

    m[3].metric(
        "Validar ingreso",
        f"{validate:,}",
    )

    m[4].metric(
        "Revisión comercial",
        f"{review:,}",
    )

    m[5].metric(
        "Cobertura de mercado",
        (
            f"{market_cov:.1%}"
            if pd.notna(market_cov)
            else "Sin información"
        ),
    )

    st.caption(
        "Cobertura de tarifas UDLA: "
        + (
            f"{tariff_cov:.1%}"
            if pd.notna(tariff_cov)
            else "Sin información"
        )
    )

    st.divider()

    a, b = st.columns(2)

    with a:
        st.subheader(
            "Resultado competitivo final"
        )

        if (
            "AdResultadoCompetitividadFinal"
            in f.columns
        ):
            d = (
                f["AdResultadoCompetitividadFinal"]
                .fillna("Sin información")
                .value_counts()
                .rename_axis("Resultado")
                .reset_index(name="Cantidad")
                .sort_values("Cantidad")
            )

            fig = px.bar(
                d,
                x="Cantidad",
                y="Resultado",
                orientation="h",
                text="Cantidad",
                color="Resultado",
                color_discrete_sequence=[
                    P,
                    M,
                    "#C9A5AF",
                    "#7A7F87",
                ],
            )

            fig.update_layout(
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

            fig.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:
            st.info(
                "No está disponible la variable "
                "de resultado competitivo."
            )

    with b:
        st.subheader(
            "Distribución socioeconómica"
        )

        if "AdNivelSocioec" in f.columns:
            d = (
                f["AdNivelSocioec"]
                .fillna("Sin información")
                .value_counts()
                .rename_axis("Nivel")
                .reset_index(name="Cantidad")
            )

            fig = px.pie(
                d,
                names="Nivel",
                values="Cantidad",
                hole=0.45,
                color_discrete_sequence=[
                    D,
                    P,
                    M,
                    "#E6A5B5",
                    "#7A7F87",
                ],
            )

            fig.update_layout(
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
                fig,
                use_container_width=True,
            )

        else:
            st.info(
                "No está disponible la variable "
                "de nivel socioeconómico."
            )

    a, b = st.columns(2)

    with a:
        st.subheader(
            "Carreras con más postulantes"
        )

        if "AdCarrera" in f.columns:
            d = (
                f["AdCarrera"]
                .fillna("Sin carrera")
                .value_counts()
                .head(10)
                .rename_axis("Carrera")
                .reset_index(name="Cantidad")
                .sort_values("Cantidad")
            )

            fig = px.bar(
                d,
                x="Cantidad",
                y="Carrera",
                orientation="h",
                text="Cantidad",
                color_discrete_sequence=[P],
            )

            fig.update_layout(
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

            fig.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:
            st.info(
                "No está disponible "
                "la variable carrera."
            )

    with b:
        st.subheader(
            "Decisión final sobre la beca"
        )

        if "AdMotivoBecaFinal" in f.columns:
            d = (
                f["AdMotivoBecaFinal"]
                .fillna("Sin información")
                .value_counts()
                .head(10)
                .rename_axis("Motivo")
                .reset_index(name="Cantidad")
                .sort_values("Cantidad")
            )

            fig = px.bar(
                d,
                x="Cantidad",
                y="Motivo",
                orientation="h",
                text="Cantidad",
                color_discrete_sequence=[M],
            )

            fig.update_layout(
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

            fig.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:
            st.info(
                "No está disponible el motivo "
                "de la beca final."
            )


# ============================== ADVANCED TAB ==============================
with t_advanced:
    st.subheader(
        "Análisis avanzado del modelo"
    )

    st.caption(
        "Todas las visualizaciones responden "
        "a los filtros superiores."
    )

    # 1. UDLA VS. MERCADO
    st.markdown(
        "### 1. UDLA frente al mercado "
        "antes y después de la beca"
    )

    req = [
        "AdCostoRefMercado",
        "AdCostoInstitucion",
        "AdCostoUDLAConBecaFinal",
    ]

    if all(c in f.columns for c in req):
        d = f.copy()

        d["Mercado"] = num(
            d,
            "AdCostoRefMercado",
        )

        d["UDLA sin beca"] = num(
            d,
            "AdCostoInstitucion",
        )

        d["UDLA con beca"] = num(
            d,
            "AdCostoUDLAConBecaFinal",
        )

        d = d.dropna(
            subset=[
                "Mercado",
                "UDLA sin beca",
                "UDLA con beca",
            ]
        )

        if len(d) > 2500:
            d = d.sample(
                2500,
                random_state=42,
            )

        if d.empty:
            st.info(
                "No existen valores suficientes."
            )

        else:
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    "Antes de la beca",
                    "Después de la beca final",
                ),
                horizontal_spacing=0.10,
            )

            hover = (
                d.get(
                    "AdCarrera",
                    pd.Series(
                        "",
                        index=d.index,
                    ),
                )
                .fillna("")
                .astype(str)
                + "<br>"
                + d.get(
                    "AdNivelSocioec",
                    pd.Series(
                        "",
                        index=d.index,
                    ),
                )
                .fillna("")
                .astype(str)
            )

            fig.add_trace(
                go.Scatter(
                    x=d["Mercado"],
                    y=d["UDLA sin beca"],
                    mode="markers",
                    marker=dict(
                        size=7,
                        opacity=0.55,
                        color=D,
                    ),
                    text=hover,
                    hovertemplate=(
                        "%{text}<br>"
                        "Mercado: $%{x:,.2f}<br>"
                        "UDLA: $%{y:,.2f}"
                        "<extra></extra>"
                    ),
                    name="Sin beca",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=d["Mercado"],
                    y=d["UDLA con beca"],
                    mode="markers",
                    marker=dict(
                        size=7,
                        opacity=0.55,
                        color=P,
                    ),
                    text=hover,
                    hovertemplate=(
                        "%{text}<br>"
                        "Mercado: $%{x:,.2f}<br>"
                        "UDLA con beca: $%{y:,.2f}"
                        "<extra></extra>"
                    ),
                    name="Con beca final",
                ),
                row=1,
                col=2,
            )

            lo = float(
                np.nanmin(
                    [
                        d["Mercado"].min(),
                        d["UDLA sin beca"].min(),
                        d["UDLA con beca"].min(),
                    ]
                )
            )

            hi = float(
                np.nanmax(
                    [
                        d["Mercado"].max(),
                        d["UDLA sin beca"].max(),
                        d["UDLA con beca"].max(),
                    ]
                )
            )

            for j in [1, 2]:
                fig.add_trace(
                    go.Scatter(
                        x=[lo, hi],
                        y=[lo, hi],
                        mode="lines",
                        line=dict(
                            color=G,
                            dash="dash",
                        ),
                        name="Igual al mercado",
                        showlegend=j == 1,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=j,
                )

            fig.update_xaxes(
                title_text="Referencia de mercado",
                tickprefix="$",
            )

            fig.update_yaxes(
                title_text="Costo UDLA",
                tickprefix="$",
                row=1,
                col=1,
            )

            fig.update_yaxes(
                title_text="Costo UDLA con beca",
                tickprefix="$",
                row=1,
                col=2,
            )

            fig.update_layout(
                height=520,
                template="plotly_white",
                margin=dict(
                    l=20,
                    r=20,
                    t=70,
                    b=30,
                ),
                legend_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "Los puntos sobre la diagonal "
                "indican que UDLA cuesta más "
                "que la referencia de mercado."
            )

    else:
        st.info(
            "Faltan columnas de costos "
            "para construir este gráfico."
        )

    # 2. BECA RECOMENDADA VS. FINAL
    st.markdown(
        "### 2. Beca recomendada "
        "frente a beca final"
    )

    if {
        "AdBecaRecomendada",
        "AdBecaFinalSugerida",
    }.issubset(f.columns):

        d = pd.DataFrame(
            {
                "Beca recomendada": (
                    num(
                        f,
                        "AdBecaRecomendada",
                    )
                    * 100
                ),
                "Beca final": (
                    num(
                        f,
                        "AdBecaFinalSugerida",
                    )
                    * 100
                ),
                "Ajustada": np.where(
                    num(
                        f,
                        "AdBecaFueAjustada",
                    )
                    .fillna(0)
                    .eq(1),
                    "Sí",
                    "No",
                ),
                "Carrera": f.get(
                    "AdCarrera",
                    pd.Series(
                        "Sin carrera",
                        index=f.index,
                    ),
                ).fillna("Sin carrera"),
            }
        ).dropna(
            subset=[
                "Beca recomendada",
                "Beca final",
            ]
        )

        if d.empty:
            st.info(
                "No existen becas suficientes."
            )

        else:
            fig = px.scatter(
                d,
                x="Beca recomendada",
                y="Beca final",
                color="Ajustada",
                hover_data=["Carrera"],
                opacity=0.65,
                color_discrete_map={
                    "Sí": P,
                    "No": G,
                },
            )

            mx = max(
                100.0,
                d[
                    [
                        "Beca recomendada",
                        "Beca final",
                    ]
                ]
                .max()
                .max(),
            )

            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=mx,
                y1=mx,
                line=dict(
                    color=D,
                    dash="dash",
                ),
            )

            fig.update_layout(
                height=480,
                template="plotly_white",
                xaxis_title=(
                    "Beca recomendada (%)"
                ),
                yaxis_title=(
                    "Beca final (%)"
                ),
                legend_title="Beca ajustada",
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=30,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    else:
        st.info(
            "Faltan las variables de beca "
            "recomendada o beca final."
        )

    # 3 Y 4
    a, b = st.columns(2)

    with a:
        st.markdown(
            "### 3. Brecha promedio por carrera"
        )

        gap_col = first_col(
            f,
            [
                "AdGapFinalVsMercado",
                "AdGapPostBecaMercado",
            ],
        )

        if (
            gap_col
            and "AdCarrera" in f.columns
        ):
            d = f[
                [
                    "AdCarrera",
                    gap_col,
                ]
            ].copy()

            d[gap_col] = pd.to_numeric(
                d[gap_col],
                errors="coerce",
            )

            s = (
                d.dropna()
                .groupby(
                    "AdCarrera",
                    as_index=False,
                )
                .agg(
                    BrechaPromedio=(
                        gap_col,
                        "mean",
                    ),
                    Casos=(
                        gap_col,
                        "size",
                    ),
                )
            )

            s = s[s["Casos"] >= 3]

            s["Abs"] = (
                s["BrechaPromedio"]
                .abs()
            )

            s = (
                s.nlargest(
                    15,
                    "Abs",
                )
                .sort_values(
                    "BrechaPromedio"
                )
            )

            if s.empty:
                st.info(
                    "No existen suficientes "
                    "casos por carrera."
                )

            else:
                colors = np.where(
                    s["BrechaPromedio"] > 0,
                    RED,
                    GREEN,
                )

                fig = go.Figure(
                    go.Bar(
                        x=s["BrechaPromedio"],
                        y=s["AdCarrera"],
                        orientation="h",
                        marker_color=colors,
                        text=[
                            money(v)
                            for v in s[
                                "BrechaPromedio"
                            ]
                        ],
                        textposition="outside",
                        customdata=s[["Casos"]],
                        hovertemplate=(
                            "%{y}<br>"
                            "Brecha promedio: "
                            "$%{x:,.2f}<br>"
                            "Casos: %{customdata[0]}"
                            "<extra></extra>"
                        ),
                    )
                )

                fig.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color=G,
                )

                fig.update_layout(
                    height=560,
                    template="plotly_white",
                    xaxis_title=(
                        "Brecha promedio frente "
                        "al mercado"
                    ),
                    yaxis_title="",
                    margin=dict(
                        l=20,
                        r=50,
                        t=20,
                        b=30,
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.caption(
                    "Brecha positiva: UDLA permanece "
                    "por encima del mercado."
                )

        else:
            st.info(
                "No están disponibles la carrera "
                "o la brecha final."
            )

    with b:
        st.markdown(
            "### 4. Embudo de cobertura del modelo"
        )

        total = len(f)

        tariff = (
            num(
                f,
                "AdCostoInstitucion",
            )
            .notna()
            .sum()
        )

        market = (
            num(
                f,
                "AdCostoRefMercado",
            )
            .notna()
            .sum()
        )

        rec = (
            num(
                f,
                "AdBecaRecomendada",
            )
            .notna()
            .sum()
        )

        final = (
            num(
                f,
                "AdBecaFinalSugerida",
            )
            .notna()
            .sum()
        )

        if (
            "AdResultadoCompetitividadFinal"
            in f.columns
        ):
            competitive = int(
                (
                    contains(
                        f[
                            "AdResultadoCompetitividadFinal"
                        ],
                        "iguala",
                    )
                    |
                    contains(
                        f[
                            "AdResultadoCompetitividadFinal"
                        ],
                        "mejora",
                    )
                )
                .sum()
            )
        else:
            competitive = 0

        funnel = pd.DataFrame(
            {
                "Etapa": [
                    "Total de registros",
                    "Con tarifa UDLA",
                    "Con referencia de mercado",
                    "Con beca recomendada",
                    "Con beca final",
                    "Iguala o mejora el mercado",
                ],
                "Cantidad": [
                    total,
                    tariff,
                    market,
                    rec,
                    final,
                    competitive,
                ],
            }
        )

        fig = go.Figure(
            go.Funnel(
                y=funnel["Etapa"],
                x=funnel["Cantidad"],
                textinfo=(
                    "value+percent initial"
                ),
                marker=dict(
                    color=[
                        D,
                        P,
                        M,
                        "#D98CA0",
                        "#E6B5C1",
                        GREEN,
                    ]
                ),
            )
        )

        fig.update_layout(
            height=560,
            template="plotly_white",
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=30,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # 5. MATRIZ
    st.markdown(
        "### 5. Matriz de beca final "
        "por carrera y nivel socioeconómico"
    )

    if {
        "AdCarrera",
        "AdNivelSocioec",
        "AdBecaFinalSugerida",
    }.issubset(f.columns):

        top = (
            f["AdCarrera"]
            .dropna()
            .value_counts()
            .head(15)
            .index
        )

        d = f[
            f["AdCarrera"].isin(top)
        ][
            [
                "AdCarrera",
                "AdNivelSocioec",
                "AdBecaFinalSugerida",
            ]
        ].copy()

        d["AdBecaFinalSugerida"] = (
            pd.to_numeric(
                d["AdBecaFinalSugerida"],
                errors="coerce",
            )
            * 100
        )

        matrix = d.pivot_table(
            index="AdCarrera",
            columns="AdNivelSocioec",
            values="AdBecaFinalSugerida",
            aggfunc="mean",
        )

        if matrix.empty:
            st.info(
                "No existen datos suficientes "
                "para construir la matriz."
            )

        else:
            fig = px.imshow(
                matrix,
                text_auto=".1f",
                aspect="auto",
                color_continuous_scale=[
                    [0, "#F9EDF1"],
                    [0.5, M],
                    [1, D],
                ],
                labels=dict(
                    x="Nivel socioeconómico",
                    y="Carrera",
                    color="Beca promedio (%)",
                ),
            )

            fig.update_layout(
                height=650,
                template="plotly_white",
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=30,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    else:
        st.info(
            "Faltan carrera, nivel socioeconómico "
            "o beca final."
        )

    # 6 Y 7
    a, b = st.columns(2)

    with a:
        st.markdown(
            "### 6. Confianza de referencia "
            "y resultado competitivo"
        )

        if {
            "AdConfianzaReferenciaMercado",
            "AdResultadoCompetitividadFinal",
        }.issubset(f.columns):

            d = pd.crosstab(
                f[
                    "AdConfianzaReferenciaMercado"
                ].fillna("Sin información"),
                f[
                    "AdResultadoCompetitividadFinal"
                ].fillna("Sin información"),
            ).reset_index()

            d = d.melt(
                id_vars=(
                    "AdConfianzaReferenciaMercado"
                ),
                var_name="Resultado",
                value_name="Cantidad",
            )

            fig = px.bar(
                d,
                x="AdConfianzaReferenciaMercado",
                y="Cantidad",
                color="Resultado",
                barmode="stack",
                text_auto=True,
                color_discrete_sequence=[
                    GREEN,
                    RED,
                    G,
                    M,
                ],
            )

            fig.update_layout(
                height=520,
                template="plotly_white",
                xaxis_title=(
                    "Confianza de la referencia"
                ),
                yaxis_title="Cantidad",
                legend_title=(
                    "Resultado competitivo"
                ),
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=30,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:
            st.info(
                "Faltan la confianza "
                "o el resultado competitivo."
            )

    with b:
        st.markdown(
            "### 7. Distribución de becas "
            "recomendadas y finales"
        )

        if {
            "AdBecaRecomendada",
            "AdBecaFinalSugerida",
        }.issubset(f.columns):

            bins = [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                75,
                101,
            ]

            labels = [
                "0%–<10%",
                "10%–<20%",
                "20%–<30%",
                "30%–<40%",
                "40%–<50%",
                "50%–<60%",
                "60%–<75%",
                "75%–100%",
            ]

            def dist(series, name):
                x = (
                    series
                    * 100
                ).clip(
                    0,
                    100,
                )

                out = (
                    pd.cut(
                        x,
                        bins=bins,
                        labels=labels,
                        right=False,
                        include_lowest=True,
                    )
                    .value_counts(sort=False)
                    .rename_axis("Rango")
                    .reset_index(
                        name="Cantidad"
                    )
                )

                out["Tipo"] = name

                return out

            d = pd.concat(
                [
                    dist(
                        num(
                            f,
                            "AdBecaRecomendada",
                        ),
                        "Beca recomendada",
                    ),
                    dist(
                        num(
                            f,
                            "AdBecaFinalSugerida",
                        ),
                        "Beca final",
                    ),
                ],
                ignore_index=True,
            )

            fig = px.bar(
                d,
                x="Rango",
                y="Cantidad",
                color="Tipo",
                barmode="group",
                text_auto=True,
                color_discrete_map={
                    "Beca recomendada": G,
                    "Beca final": P,
                },
            )

            fig.update_layout(
                height=520,
                template="plotly_white",
                xaxis_title="Rango de beca",
                yaxis_title="Cantidad",
                legend_title="",
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=60,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:
            st.info(
                "Faltan las variables de beca "
                "recomendada o final."
            )

    # 8. GESTIÓN POR CONSULTOR
    st.markdown(
        "### 8. Gestión por consultor"
    )

    if "AdAsesorNombre" in f.columns:
        d = f.copy()

        d["Dias"] = num(
            d,
            "AdDiasSinGestion",
        )

        d["Revision"] = (
            num(
                d,
                "AdRequiereRevisionComercial",
            )
            .fillna(0)
        )

        d["Ajustada"] = (
            num(
                d,
                "AdBecaFueAjustada",
            )
            .fillna(0)
        )

        if "AdIdentificacion" in d.columns:
            s = (
                d.groupby(
                    "AdAsesorNombre",
                    dropna=False,
                )
                .agg(
                    Postulantes=(
                        "AdIdentificacion",
                        "nunique",
                    ),
                    PromedioDiasSinGestion=(
                        "Dias",
                        "mean",
                    ),
                    CasosRevision=(
                        "Revision",
                        "sum",
                    ),
                    BecasAjustadas=(
                        "Ajustada",
                        "sum",
                    ),
                )
                .reset_index()
            )

        else:
            s = (
                d.groupby(
                    "AdAsesorNombre",
                    dropna=False,
                )
                .agg(
                    Postulantes=(
                        "AdAsesorNombre",
                        "size",
                    ),
                    PromedioDiasSinGestion=(
                        "Dias",
                        "mean",
                    ),
                    CasosRevision=(
                        "Revision",
                        "sum",
                    ),
                    BecasAjustadas=(
                        "Ajustada",
                        "sum",
                    ),
                )
                .reset_index()
            )

        s["AdAsesorNombre"] = (
            s["AdAsesorNombre"]
            .fillna("Sin asesor")
        )

        s = (
            s.nlargest(
                15,
                "Postulantes",
            )
            .sort_values("Postulantes")
        )

        if s.empty:
            st.info(
                "No existen asesores "
                "para construir este gráfico."
            )

        else:
            fig = make_subplots(
                specs=[
                    [
                        {
                            "secondary_y": True
                        }
                    ]
                ]
            )

            fig.add_trace(
                go.Bar(
                    x=s["AdAsesorNombre"],
                    y=s["Postulantes"],
                    name="Postulantes",
                    marker_color=P,
                    text=s["Postulantes"],
                    textposition="outside",
                    customdata=s[
                        [
                            "CasosRevision",
                            "BecasAjustadas",
                        ]
                    ],
                    hovertemplate=(
                        "%{x}<br>"
                        "Postulantes: %{y}<br>"
                        "Revisión comercial: "
                        "%{customdata[0]:.0f}<br>"
                        "Becas ajustadas: "
                        "%{customdata[1]:.0f}"
                        "<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(
                    x=s["AdAsesorNombre"],
                    y=s[
                        "PromedioDiasSinGestion"
                    ],
                    name=(
                        "Promedio días sin gestión"
                    ),
                    mode="lines+markers",
                    line=dict(
                        color=YELLOW,
                        width=3,
                    ),
                    marker=dict(
                        size=8
                    ),
                    hovertemplate=(
                        "%{x}<br>"
                        "Promedio días: %{y:.1f}"
                        "<extra></extra>"
                    ),
                ),
                secondary_y=True,
            )

            fig.update_xaxes(
                title_text="Consultor",
                tickangle=-35,
            )

            fig.update_yaxes(
                title_text="Postulantes",
                secondary_y=False,
            )

            fig.update_yaxes(
                title_text=(
                    "Promedio días sin gestión"
                ),
                secondary_y=True,
            )

            fig.update_layout(
                height=620,
                template="plotly_white",
                legend_title="",
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=140,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            table = (
                s.rename(
                    columns={
                        "AdAsesorNombre": (
                            "Consultor"
                        ),
                        "PromedioDiasSinGestion": (
                            "Promedio días sin gestión"
                        ),
                        "CasosRevision": (
                            "Casos para revisión"
                        ),
                        "BecasAjustadas": (
                            "Becas ajustadas"
                        ),
                    }
                )
                .sort_values(
                    "Postulantes",
                    ascending=False,
                )
            )

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Promedio días sin gestión": (
                        st.column_config.NumberColumn(
                            "Promedio días sin gestión",
                            format="%.1f",
                        )
                    ),
                    "Casos para revisión": (
                        st.column_config.NumberColumn(
                            "Casos para revisión",
                            format="%.0f",
                        )
                    ),
                    "Becas ajustadas": (
                        st.column_config.NumberColumn(
                            "Becas ajustadas",
                            format="%.0f",
                        )
                    ),
                },
            )

    else:
        st.info(
            "No está disponible "
            "el consultor de cierre."
        )


st.divider()

st.caption(
    "Dashboard de Elasticidad de Costos · "
    "Dirección de Inteligencia de la Información"
)

