# =============================================================================
# Etapa 4 — Mapper TDA
# Ácido Fólico (AF) y subgrupos materno-neonatales
#
# Pregunta objetivo:
#   ¿El nivel de consumo total de ácido fólico se asocia con la formación de
#   subgrupos diferenciables en características maternas y neonatales mediante
#   Análisis Topológico de Datos (TDA)?
#
# Diseño:
#   Espacio de features : variables maternas + neonatales  (lo que queremos
#                         saber si se agrupa de forma diferenciada)
#   Lente               : Total mg/d AF suple y pan        (el predictor clave)
#   Post-mapper         : composición Bajo / Medio / Alto por nodo
#
# Outputs:
#   Resultados/mapper_AF.html          — Grafo interactivo coloreado por Total AF
#   Resultados/mapper_cat_AF.html      — Grafo coloreado por categoría (0=Bajo…2=Alto)
#   Resultados/mapper_composicion.png  — Composición de categorías AF por nodo
#   Resultados/mapper_nodos.xlsx       — Estadísticas completas por nodo
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import kmapper as km
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parents[1]
BD_PATH   = BASE_DIR / "BD" / "DB_Limpia_TDA.xlsx"
RES_DIR   = BASE_DIR / "Resultados"
HTML_AF   = RES_DIR / "mapper_AF.html"
HTML_CAT  = RES_DIR / "mapper_cat_AF.html"
PNG_COMP  = RES_DIR / "mapper_composicion.png"
EXCEL_OUT = RES_DIR / "mapper_nodos.xlsx"
RES_DIR.mkdir(exist_ok=True)

# Colores para categorías
CAT_COLORS = {"Bajo": "#4C72B0", "Medio": "#DD8452", "Alto": "#55A868"}
CAT_ORDER  = ["Bajo", "Medio", "Alto"]

# =============================================================================
# 1. CARGA
# =============================================================================
print("=" * 70)
print("CARGANDO BASE LIMPIA")
print("=" * 70)

df = pd.read_excel(BD_PATH)
print(f"  Dimensiones: {df.shape}")

# =============================================================================
# 2. VARIABLES
# =============================================================================
LENTE_COL = "Total mg/d AF  suple y pan"
CAT_COL   = "Total mg/d AF  suple y pan [cat]"   # Bajo / Medio / Alto

MATERNAS   = ["Edad madre", "Educación", "N° embarazo", "Región",
              "Dif peso mamá", "Dif IMC"]
NEONATALES = ["Sexo hijo", "PN hijo (g)", "EG hijo (sem)"]
OUTCOME_COLS = MATERNAS + NEONATALES

# Verificar que existan
for c in [LENTE_COL, CAT_COL] + OUTCOME_COLS:
    assert c in df.columns, f"Columna no encontrada: {c}"

# =============================================================================
# 3. PREPARACIÓN
# =============================================================================
df_work = df[[LENTE_COL, CAT_COL] + OUTCOME_COLS].copy()

# Eliminar filas sin lente
n_antes = len(df_work)
df_work = df_work.dropna(subset=[LENTE_COL]).reset_index(drop=True)
print(f"  Filas eliminadas (NaN en lente): {n_antes - len(df_work)}")
print(f"  Filas para Mapper: {len(df_work)}")

# --- Encoding de variables categóricas / texto ---
print("\n  Encoding de variables categóricas:")

ENCODING_LOG = {}

def encode_if_object(df_in, col):
    """Convierte columna texto a entero ordinal si es tipo object."""
    if df_in[col].dtype == object:
        vals = sorted(df_in[col].dropna().unique())
        enc  = {v: i for i, v in enumerate(vals)}
        df_in[col] = df_in[col].map(enc)
        ENCODING_LOG[col] = enc
        print(f"    '{col}': {enc}")
    else:
        print(f"    '{col}': ya numérica (dtype={df_in[col].dtype})")

encode_if_object(df_work, "Educación")
encode_if_object(df_work, "Región")

# EG hijo (sem): forzar numérico por si contiene texto remanente
if df_work["EG hijo (sem)"].dtype == object:
    df_work["EG hijo (sem)"] = pd.to_numeric(df_work["EG hijo (sem)"], errors="coerce")
    print(f"    'EG hijo (sem)': convertida a numérico")
else:
    print(f"    'EG hijo (sem)': ya numérica (dtype={df_work['EG hijo (sem)'].dtype})")

# Forzar numérico cualquier otra columna de features que quede como object
for col in OUTCOME_COLS:
    if df_work[col].dtype == object:
        df_work[col] = pd.to_numeric(df_work[col], errors="coerce")
        print(f"    '{col}': forzada a numérico")

# Imputar NaN en features con mediana
for col in OUTCOME_COLS:
    if df_work[col].isna().any():
        med = df_work[col].median()
        df_work[col] = df_work[col].fillna(med)
        print(f"  Imputado '{col}' → mediana={med:.4f}")

assert df_work[OUTCOME_COLS].isna().sum().sum() == 0, "NaN restantes en features"

# Categoría y AF continuo (para colorear el grafo)
lente_vals = df_work[LENTE_COL].values
cat_vals   = df_work[CAT_COL].values      # "Bajo" / "Medio" / "Alto"
cat_enc    = np.array([CAT_ORDER.index(c) for c in cat_vals], dtype=float)  # 0/1/2

print(f"\n  Distribución de categorías AF:")
for cat in CAT_ORDER:
    n = (cat_vals == cat).sum()
    print(f"    {cat:5s}: n={n:4d}  ({n/len(cat_vals)*100:.1f}%)")

# Normalizar features [0, 1]
scaler = MinMaxScaler()
X = scaler.fit_transform(df_work[OUTCOME_COLS].values)

# AF normalizado (solo para colorear HTML)
lente_norm = MinMaxScaler().fit_transform(lente_vals.reshape(-1, 1))

print(f"\n  Espacio de features ({len(OUTCOME_COLS)} vars): {OUTCOME_COLS}")
print(f"  Lente: UMAP 2D sobre el espacio de features")

# =============================================================================
# 3b. UMAP — reducción a 2D para usar como lente del Mapper
# =============================================================================
print("\n" + "=" * 70)
print("COMPUTANDO UMAP 2D (lente del Mapper)")
print("=" * 70)

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
UMAP_SEED        = 42

reducer   = umap.UMAP(n_components=2, n_neighbors=UMAP_N_NEIGHBORS,
                      min_dist=UMAP_MIN_DIST, random_state=UMAP_SEED)
X_umap    = reducer.fit_transform(X)
lente_2d  = MinMaxScaler().fit_transform(X_umap)   # normalizar [0,1] para el Cover

print(f"  n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, seed={UMAP_SEED}")
print(f"  Embedding shape: {lente_2d.shape}")

# =============================================================================
# 4. PARÁMETROS DEL MAPPER
# =============================================================================
N_CUBES  = 8     # por dimensión — con lente 2D el cover crea N_CUBES² celdas
OVERLAP  = 0.50
EPS      = 0.42
MIN_SAMP = 4

cover     = km.Cover(n_cubes=N_CUBES, perc_overlap=OVERLAP)
clusterer = DBSCAN(eps=EPS, min_samples=MIN_SAMP)

print(f"\n  Cover  : n_cubes={N_CUBES} (2D → hasta {N_CUBES**2} celdas), overlap={OVERLAP}")
print(f"  DBSCAN : eps={EPS}, min_samples={MIN_SAMP}")

# =============================================================================
# 5. MAPPER
# =============================================================================
print("\n" + "=" * 70)
print("EJECUTANDO MAPPER")
print("=" * 70)

mapper = km.KeplerMapper(verbose=1)
graph  = mapper.map(lente_2d, X, cover=cover, clusterer=clusterer)

n_nodes = len(graph["nodes"])
n_edges = sum(len(v) for v in graph["links"].values())
print(f"\n  Nodos: {n_nodes}  |  Aristas: {n_edges}")

# =============================================================================
# 6. VISUALIZACIONES HTML
# =============================================================================

# A) Coloreado por Total AF continuo
mapper.visualize(
    graph,
    color_values=lente_norm.ravel(),
    color_function_name="Total AF mg/día (normalizado)",
    title="Mapper TDA — Lente: UMAP 2D | Color: Total AF mg/día",
    path_html=str(HTML_AF),
)
print(f"\n  HTML (continuo) : {HTML_AF}")

# B) Coloreado por categoría  0=Bajo · 1=Medio · 2=Alto
mapper.visualize(
    graph,
    color_values=cat_enc,
    color_function_name="Categoría AF  (0=Bajo · 1=Medio · 2=Alto)",
    title="Mapper TDA — Lente: UMAP 2D | Color: Categoría AF (0=Bajo · 1=Medio · 2=Alto)",
    path_html=str(HTML_CAT),
)
print(f"  HTML (categoría): {HTML_CAT}")

# =============================================================================
# 7. ANÁLISIS POST-MAPPER  — composición por nodo
# =============================================================================
print("\n" + "=" * 70)
print("ANÁLISIS POST-MAPPER — composición Bajo/Medio/Alto por nodo")
print("=" * 70)

node_rows = []
for node_id, member_ids in graph["nodes"].items():
    cats_node  = cat_vals[member_ids]
    feats_node = df_work.iloc[member_ids][OUTCOME_COLS]
    row = {
        "Nodo"   : node_id,
        "N"      : len(member_ids),
        "N_Bajo" : int((cats_node == "Bajo").sum()),
        "N_Medio": int((cats_node == "Medio").sum()),
        "N_Alto" : int((cats_node == "Alto").sum()),
        "pct_Bajo" : round((cats_node == "Bajo").mean()  * 100, 1),
        "pct_Medio": round((cats_node == "Medio").mean() * 100, 1),
        "pct_Alto" : round((cats_node == "Alto").mean()  * 100, 1),
        "AF_media" : round(df_work.iloc[member_ids][LENTE_COL].mean(), 4),
    }
    for col in OUTCOME_COLS:
        row[f"{col} (media)"] = round(feats_node[col].mean(), 4)
    node_rows.append(row)

df_nodos = (pd.DataFrame(node_rows)
              .sort_values("AF_media")
              .reset_index(drop=True))

# Etiqueta dominante de categoría
def dominant_cat(r):
    return max(CAT_ORDER, key=lambda c: r[f"N_{c}"])

df_nodos["Cat_dominante"] = df_nodos.apply(dominant_cat, axis=1)

preview_cols = ["Nodo", "N", "AF_media", "Cat_dominante",
                "pct_Bajo", "pct_Medio", "pct_Alto",
                "Edad madre (media)", "PN hijo (g) (media)", "EG hijo (sem) (media)"]
print(df_nodos[preview_cols].to_string(index=False))

# =============================================================================
# 8. GRÁFICO — Composición de categorías AF por nodo
# =============================================================================
fig, ax = plt.subplots(figsize=(max(10, n_nodes * 0.6), 5))

x     = np.arange(n_nodes)
width = 0.7

bot = np.zeros(n_nodes)
for cat in CAT_ORDER:
    vals = df_nodos[f"pct_{cat}"].values
    ax.bar(x, vals, width, bottom=bot, label=cat,
           color=CAT_COLORS[cat], edgecolor="white", linewidth=0.4)
    bot += vals

# Etiquetar N dentro de cada barra
for i, (_, row) in enumerate(df_nodos.iterrows()):
    ax.text(i, 102, f"n={int(row['N'])}", ha="center", va="bottom",
            fontsize=6.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(
    [f"N{i+1}\nAF={r['AF_media']:.1f}" for i, (_, r) in enumerate(df_nodos.iterrows())],
    fontsize=7, rotation=45, ha="right",
)
ax.set_ylabel("Proporción (%)", fontsize=11)
ax.set_ylim(0, 120)
ax.set_title(
    "Composición de Categoría AF (Bajo / Medio / Alto) por Nodo del Mapper",
    fontsize=12, fontweight="bold",
)
ax.legend(title="Categoría AF", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig(PNG_COMP, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Gráfico composición: {PNG_COMP}")

# =============================================================================
# 9. EXPORTAR EXCEL
# =============================================================================
params = {
    "n_filas"              : len(df_work),
    "features"             : ", ".join(OUTCOME_COLS),
    "lente"                : "UMAP 2D sobre espacio de features",
    "umap_n_neighbors"     : UMAP_N_NEIGHBORS,
    "umap_min_dist"        : UMAP_MIN_DIST,
    "umap_seed"            : UMAP_SEED,
    "n_cubes"              : N_CUBES,
    "overlap"              : OVERLAP,
    "dbscan_eps"           : EPS,
    "dbscan_min_samples"   : MIN_SAMP,
    "nodos"                : n_nodes,
    "aristas"              : n_edges,
}

with pd.ExcelWriter(EXCEL_OUT, engine="openpyxl") as writer:
    df_nodos.to_excel(writer, sheet_name="Nodos_Composicion", index=False)
    pd.DataFrame(
        [{"Parámetro": k, "Valor": v} for k, v in params.items()]
    ).to_excel(writer, sheet_name="Parametros", index=False)

print(f"  Excel: {EXCEL_OUT}")

# =============================================================================
# 10. RESUMEN EJECUTIVO
# =============================================================================
print("\n" + "=" * 70)
print("RESUMEN EJECUTIVO")
print("=" * 70)

nodos_alto = df_nodos[df_nodos["Cat_dominante"] == "Alto"]
nodos_bajo = df_nodos[df_nodos["Cat_dominante"] == "Bajo"]

for grupo, sub in [("Alto AF", nodos_alto), ("Bajo AF", nodos_bajo)]:
    if len(sub) == 0:
        continue
    print(f"\n  Nodos dominantes — {grupo}  (n_nodos={len(sub)}):")
    print(f"    AF media       : {sub['AF_media'].mean():.4f} mg/día")
    print(f"    Edad madre     : {sub['Edad madre (media)'].mean():.2f} años")
    print(f"    PN hijo        : {sub['PN hijo (g) (media)'].mean():.1f} g")
    print(f"    EG hijo        : {sub['EG hijo (sem) (media)'].mean():.2f} sem")

print("\n" + "=" * 70)
print("MAPPER COMPLETADO")
print("=" * 70)
print(f"  HTML (continuo)  : {HTML_AF}")
print(f"  HTML (categoría) : {HTML_CAT}")
print(f"  Composición      : {PNG_COMP}")
print(f"  Excel nodos      : {EXCEL_OUT}")
