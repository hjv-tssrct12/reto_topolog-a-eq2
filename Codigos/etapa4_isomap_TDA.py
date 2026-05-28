### Etapa 4 - Isomap TDA
# Geoetría no lineal de subgrupos materno-neonatales asociados al consumo de ácido fólico

## Pregunta objetivo: ¿La geometría intrínseca de las variables maternas y neonatales presenta estructuras
# o subgrupos diferenciables asociados al consumo de ácido fólico meidante reducción de dimensionalidad 
# no lineal con isomap?

## Diseño:
# Espacio de características: variables maternas y neonatales 
# Variable de interpretación:  Total mg/d AF suple y pan
# Método: Isomap para preservar las relaciones geométricas no lineales entre individuos en un embedding 
# de baja dimeensión
# Visualización: Embedding 2D coloreado por consumo continuo de AF y categoría de consumo

## Outputs
# Resultados/isomap_AF_continuo.png - Embedding isomap coloreado por consumo de AF continuo
# Resultados/isomap_AF_categoria.png - Embedding isomap coloreado por categoría de consumo de AF
# Resultados/isomap_embedding.xlsx - Coordenadas del embedding y variables asociadas
# Resultados/isomap_resumen.txt - Resumen descriptivo e interpretación del embedding

## Importar librerías
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

# Rutas 
BASE_DIR = Path(__file__).resolve().parents[1]
BD_PATH = BASE_DIR / "BD" / "DB_Limpia_TDA.xlsx"
RES_DIR = BASE_DIR / "Resultados"
PNG_CONTINUO = RES_DIR / "isomap_AF_continuo.png"
PNG_CAT = RES_DIR / "isomap_AF_categoria.png"
EXCEL_OUT = RES_DIR / "isomap_embedding.xlsx"
TXT_OUT = RES_DIR / "isomap_resumen.txt"
RES_DIR.mkdir(exist_ok=True)

# Colores para categorías
CAT_COLORS = {"Bajo": "#4C72B0", "Medio": "#DD8452", "Alto": "#55A868"}
CAT_ORDER  = ["Bajo", "Medio", "Alto"]

# Cargar los datos
print("Cargando datos... ")
df = pd.read_excel(BD_PATH)
print(f" Dimensiones: {df.shape}")

# Variables
LENTE_COL = "Total mg/d AF  suple y pan"
CAT_COL   = "Total mg/d AF  suple y pan [cat]"   # Bajo / Medio / Alto
MATERNAS   = ["Edad madre", "Educación", "N° embarazo", "Región", "Dif peso mamá", "Dif IMC"]
NEONATALES = ["Sexo hijo", "PN hijo (g)", "EG hijo (sem)"]
OUTCOME_COLS = MATERNAS + NEONATALES

# Verificar que existan
for c in [LENTE_COL, CAT_COL] + OUTCOME_COLS:
    assert c in df.columns, f"Columna no encontrada: {c}"

# PReparación
print("\n" + "=" * 70)
print("PREPARANDO DATOS")
print("=" * 70)
df_work = df[[LENTE_COL, CAT_COL] + OUTCOME_COLS].copy()

# Eliminar filas sin valor de AF
n_antes = len(df_work)
df_work = df_work.dropna(subset=[LENTE_COL]).reset_index(drop=True)
print(f"  Filas eliminadas (NaN en lente): {n_antes - len(df_work)}")
print(f"  Filas para Isomap: {len(df_work)}")

# Imputar faltantes con mediana
for col in OUTCOME_COLS:
    if df_work[col].isna().any():
        med = df_work[col].median()
        df_work[col] = df_work[col].fillna(med)
        print(f"  Imputado '{col}' → mediana={med:.4f}")

# Verificación final
assert df_work[OUTCOME_COLS].isna().sum().sum() == 0, \
    "Persisten NaN en variables de entrada"

## Variables auxiliares
lente_vals = df_work[LENTE_COL].values
cat_vals = df_work[CAT_COL].values
cat_enc = np.array([CAT_ORDER.index(c) for c in cat_vals], dtype=float)
print("\nDistribución categorías AF:")
for cat in CAT_ORDER:
    n = (cat_vals == cat).sum()
    print(f"  {cat:5s}: n={n:4d} ({n/len(cat_vals)*100:.1f}%)")

# ESCALADO
print("ESCALANDO FEATURES")
scaler = StandardScaler()
X = scaler.fit_transform(df_work[OUTCOME_COLS].values)
print(f"  Shape matriz X: {X.shape}")
print(f"\n  Espacio de features ({len(OUTCOME_COLS)} variables):")

for c in OUTCOME_COLS:
    print(f"   - {c}")
print(f"\n  Variable de interpretación: {LENTE_COL}")

## Isomap
print("\n" + "-" * 70)
print("Aplicando Isomap...")
print("\n" + "-" * 70)

## Parámetros del isomap
n_neighbors = 10
n_components = 2

print(f" n_neighbors: {n_neighbors}")
print(f" n_components: {n_components}")

# Modelo
isomap = Isomap(n_neighbors = n_neighbors, n_components = n_components)
# Embedding
X_iso = isomap.fit_transform(X)
print(f" Embedding generado: {X_iso.shape}")

# Separar coordenadas
iso1 = X_iso[:, 0]
iso2 = X_iso[:, 1]

# Varianza del embedding
print("\n  Rangos embedding:")
print(f"    ISO1 → [{iso1.min():.4f}, {iso1.max():.4f}]")
print(f"    ISO2 → [{iso2.min():.4f}, {iso2.max():.4f}]")

# Visualización AF continuo
print("\n" + "-" * 70)
print("Visualizando embedding (AF continuo)...")
print("\n" + "-" * 70)

plt.figure(figsize = (9,7))
scatter = plt.scatter(iso1,iso2, c = lente_vals, cmap = "viridis", s = 28, alpha = 0.8, edgecolor = "k", linewidth = 0.5)
cbar = plt.colorbar(scatter)
cbar.set_label("Total mg/d AF suple y pan")

plt.xlabel("Isomap 1")
plt.ylabel("Isomap 2")
plt.title("Isomap embedding - Coloreado por consumo continuo de AF", fontsize = 13, fontweight = "bold")
plt.grid(alpha = 0.25)
plt.tight_layout()
plt.savefig(PNG_CONTINUO, dpi =  150, bbox_inches = "tight")
plt.close()
print(f"Figura guardada: {PNG_CONTINUO}")

# Visualización AF categoría
print("\n" + "-" * 70)
print("Visualizando embedding (AF categoría)...")
print("\n" + "-" * 70)

cmap_cat = ListedColormap([CAT_COLORS["Bajo"], CAT_COLORS["Medio"], CAT_COLORS["Alto"]])
plt.figure(figsize = (9,7))

scatter = plt.scatter(iso1, iso2, c = cat_enc, cmap = cmap_cat, s = 28, alpha = 0.85, edgecolors = "k", linewidth = 0.5)

handles = [ plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=CAT_COLORS[cat], markersize=8, label=cat) for cat in CAT_ORDER]
plt.legend(handles = handles, title = "Categoría AF")

plt.xlabel("Isomap 1")
plt.ylabel("Isomap 2")
plt.title("Isomap embedding - Categorías de consumo de AF", fontsize = 13, fontweight = "bold")
plt.grid(alpha = 0.25)
plt.tight_layout()
plt.savefig(PNG_CAT, dpi = 150, bbox_inches = "tight")
plt.close()
print(f"Figura guardada: {PNG_CAT}")

# Dataframe del embedding
print("\n" + "-" * 70)
print("Creando dataframe del embedding...")
print("\n" + "-" * 70)
df_embed = df_work.copy()
df_embed["ISO1"] = iso1
df_embed["ISO2"] = iso2
print(df_embed[["ISO1", "ISO2", LENTE_COL, CAT_COL]].head())

# Correlaciones con el embedding
print("\n" + "-" * 70)
print("Correlaciones entre embedding y variables originales...")
print("\n" + "-" * 70)

corr_rows = []
for col in OUTCOME_COLS:
    corr1 = np.corrcoef(df_embed["ISO1"], df_embed[col])[0,1]
    corr2 = np.corrcoef(df_embed["ISO2"], df_embed[col])[0,1]
    corr_rows.append({"Variable": col, "Corr_ISO1": round(corr1, 4), "Corr_ISO2": round(corr2, 4), "Abs_ISO1": abs(corr1), "Abs_ISO2": abs(corr2)})
df_corr = pd.DataFrame(corr_rows)
print(
    df_corr.sort_values("Abs_ISO1", ascending=False)
    [["Variable", "Corr_ISO1", "Corr_ISO2"]]
    .to_string(index=False)
)

# Exportaar el embedding
print("\n" + "-" * 70)
print("Exportando embedding a Excel...")
print("\n" + "-" * 70)

# Guardar correlaciones
df_corr_sorted  = df_corr.sort_values("Abs_ISO1", ascending = False)
with pd.ExcelWriter(EXCEL_OUT, engine='openpyxl') as writer:
    df_embed.to_excel(writer, sheet_name = "Embedding", index = False)
    df_corr_sorted.to_excel(writer, sheet_name = "Correlaciones", index = False)
print(f"Embedding y correlaciones guardados en: {EXCEL_OUT}")

# Clustering sobre Isomap
print("\n" + "-" * 70)
print("Aplicando clustering DBSCAN sobre embedding...")
print("\n" + "-" * 70)

db = DBSCAN(eps = 0.55, min_samples = 15)
clusters = db.fit_predict(X_iso)
df_embed["Cluster"] = clusters
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = (clusters == -1).sum()
print(f" Clusters encontrados: {n_clusters}")
print(f" Noise / outliers: {n_noise}")
print("\nDistribución por cluster:")
print(df_embed["Cluster"].value_counts().sort_index())

# Visualización de los clusters
print("\n" + "-" * 70)
print("Visualizando clusters en el embedding...")
print("\n" + "-" * 70)  

plt.figure(figsize = (9,7))
scatter = plt.scatter(iso1, iso2, c = clusters, cmap = "tab10", s = 30, alpha =0.85, edgecolor = "k", linewidth = 0.5)
plt.xlabel("Isomap 1")
plt.ylabel("Isomap 2")
plt.title("Clusters detectados sobre el embedding Isomap", fontsize = 13, fontweight = "bold")
plt.grid(alpha = 0.25)
plt.tight_layout()
plt.savefig(RES_DIR / "isomap_clusters.png", dpi = 150, bbox_inches = "tight")
plt.close()
print(f"Figura guardada: {RES_DIR / 'isomap_clusters.png'}")

# Perfilado de clusters
print("\n" + "-" * 70)
print("Perfilando clusters encontrados...")
print("\n" + "-" * 70)

cluster_summary =(df_embed.groupby("Cluster")[OUTCOME_COLS].mean().round(3))
print("\nResumen por cluster:")
print(cluster_summary)

print(
    df_embed.groupby("Cluster")[CAT_COL]
    .value_counts(normalize=True)
)