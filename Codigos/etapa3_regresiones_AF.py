# =============================================================================
# Etapa 3 — Regresiones Lineales y Análisis de Colinealidad
# Ácido Fólico (AF) y Equivalentes Dietéticos de Folato (DFE)
#
# Pregunta objetivo:
#   ¿El nivel de consumo total de ácido fólico se asocia con la formación de
#   subgrupos diferenciables en características maternas y neonatales mediante
#   Análisis Topológico de Datos (TDA)?
#
# Outputs:
#   BD/DB_Limpia_TDA.xlsx          — Base limpia con categorías AF/DFE
#   Resultados/Regresiones_AF_DFE.xlsx — Regresiones, colinealidad, descriptivas
#   Resultados/correlacion_AF_DFE.png  — Heatmap de correlaciones AF/DFE
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_PATH  = BASE_DIR / "BD" / "2. Base Datos encuesta AF 2_mas completa para comparar.xlsx"
BD_OUT     = BASE_DIR / "BD" / "DB_Limpia_TDA.xlsx"
RES_DIR    = BASE_DIR / "Resultados"
RES_OUT    = RES_DIR / "Regresiones_AF_DFE.xlsx"
RES_DIR.mkdir(exist_ok=True)

pd.set_option("display.float_format", lambda x: f"{x:.4f}")

# =============================================================================
# 1. DEFINICIÓN DE VARIABLES
# =============================================================================

# Variables AF/DFE — independientes (predictores)
AF_VARS = {
    "Total AF (suple+pan)" : "Total mg/d AF  suple y pan",
    "AF Suple 1T"          : "mgAF/día Suplementos y multivitamínico 1°T",
    "AF Pan"               : "mg/d AF total pan",
    "Total DFE (suple+pan)": "Total mg/d DFE suple y pan",
}

# Variables Maternas — dependientes
MATERNAS = {
    "Edad madre"   : "Edad madre",
    "Educación"    : "Educación",
    "N° embarazo"  : "N° embarazo",
    "Región"       : "Región",
    "Dif peso mamá": "Dif peso mamá",
    "Dif IMC"      : "Dif IMC",
}

# Variables Neonatales — dependientes
NEONATALES = {
    "Sexo hijo"    : "Sexo hijo",
    "PN hijo (g)"  : "PN hijo (g)",
    "EG hijo (sem)": "EG hijo (sem)",
}

af_cols      = list(AF_VARS.values())
af_labels    = list(AF_VARS.keys())
outcome_cols = list(MATERNAS.values())   + list(NEONATALES.values())
outcome_lbl  = list(MATERNAS.keys())     + list(NEONATALES.keys())
outcome_grp  = ["Materna"] * len(MATERNAS) + ["Neonatal"] * len(NEONATALES)
all_cols     = af_cols + outcome_cols

# =============================================================================
# 2. CARGA Y SELECCIÓN
# =============================================================================
print("=" * 70)
print("CARGANDO DATOS")
print("=" * 70)

df_raw = pd.read_excel(DATA_PATH)
print(f"  Dimensiones originales : {df_raw.shape}")

df = df_raw[all_cols].copy()
print(f"  Variables seleccionadas: {len(all_cols)}")
print(f"    AF/DFE (X): {af_labels}")
print(f"    Maternas(Y): {list(MATERNAS.keys())}")
print(f"    Neonat.(Y) : {list(NEONATALES.keys())}")

# =============================================================================
# 2b. CARGA Y UNIÓN CON BASE DATOS ADICIONAL
# =============================================================================
print("\n" + "=" * 70)
print("CARGANDO Y UNIENDO BASE DATOS ADICIONAL")
print("=" * 70)

DATA_PATH2 = BASE_DIR / "BD" / "2. Base Datos encuesta AF 2_mas completa 2.xlsx"

# Leer como texto para detectar y limpiar valores no numéricos (ej. "x")
df_raw2 = pd.read_excel(DATA_PATH2, sheet_name="Base Consenso", dtype=str)
print(f"  Dimensiones base adicional (raw): {df_raw2.shape}")

# Nombres exactos de columnas en la nueva base
_SAF = "SAF mg AF/día "    # tiene espacio al final en el archivo
_MAF = "MAF mg AF/día"
_PAN = "Pan mg AF  día"    # tiene doble espacio en el nombre

_src_cols = [
    "Edad Madre", "Nivel Educacional C", "Número Embarazo", "Region C",
    "Delta Peso Gestacional", "Delta IMC", "Sexo RN C", "Peso RN", "EG RN",
    _SAF, _MAF, _PAN,
]

# Excluir filas con motivo de eliminación (incluye filas de resumen estadístico)
n_antes = len(df_raw2)
df_raw2 = df_raw2[df_raw2["Motivo Eliminación"].isna()].reset_index(drop=True)
print(f"  Filas excluidas (Motivo Eliminación no nulo): {n_antes - len(df_raw2)}")
print(f"  Filas válidas base adicional: {len(df_raw2)}")

df_new = df_raw2[_src_cols].copy()

# Limpiar: "x"/"X" → "0" (celdas que debían ser 0) y convertir a numérico
for col in df_new.columns:
    df_new[col] = df_new[col].str.strip().str.replace(r"^[xX]$", "0", regex=True)
    df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

# Calcular variables AF derivadas (NaN si todos los sumandos son NaN)
df_new["Total mg/d AF  suple y pan"] = (
    df_new[[_SAF, _MAF, _PAN]].sum(axis=1, min_count=1)
)
df_new["mgAF/día Suplementos y multivitamínico 1°T"] = (
    df_new[[_SAF, _MAF]].sum(axis=1, min_count=1)
)
df_new["mg/d AF total pan"] = df_new[_PAN]

# DFE no disponible en la nueva base
df_new["Total mg/d DFE suple y pan"] = np.nan

# Renombrar al esquema de la base actual
df_new.rename(columns={
    "Edad Madre"            : "Edad madre",
    "Nivel Educacional C"   : "Educación",
    "Número Embarazo"       : "N° embarazo",
    "Region C"              : "Región",
    "Delta Peso Gestacional": "Dif peso mamá",
    "Delta IMC"             : "Dif IMC",
    "Sexo RN C"             : "Sexo hijo",
    "Peso RN"               : "PN hijo (g)",
    "EG RN"                 : "EG hijo (sem)",
}, inplace=True)

df_new_sel = df_new[all_cols].copy()
print(f"  Filas base original  : {len(df)}")
print(f"  Filas base adicional : {len(df_new_sel)}")

df = pd.concat([df, df_new_sel], ignore_index=True)
print(f"  Filas combinadas     : {len(df)}")

# =============================================================================
# 3. LIMPIEZA
# =============================================================================
print("\n" + "=" * 70)
print("LIMPIEZA DE DATOS")
print("=" * 70)

ENCODING_LOG = {}

def encode_if_object(df, col):
    """Convierte columna categórica a entero ordinal y guarda el mapa."""
    if df[col].dtype == object:
        vals = sorted(df[col].dropna().unique())
        enc  = {v: i for i, v in enumerate(vals)}
        df[col] = df[col].map(enc)
        ENCODING_LOG[col] = enc
        print(f"  Encoding '{col}': {enc}")

encode_if_object(df, "Sexo hijo")    # binario → 0/1
encode_if_object(df, "Educación")    # ordinal → int
encode_if_object(df, "Región")       # ordinal → int

# Forzar numérico cualquier columna que quede como object
for col in df.columns:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Valores faltantes
miss = pd.DataFrame({
    "N missing" : df.isna().sum(),
    "% missing" : (df.isna().sum() / len(df) * 100).round(2),
})
print("\n  Valores faltantes por variable:")
print(miss.to_string())

# Eliminar filas con missing en predictores AF (excluye DFE, ausente en base2)
af_cols_required = [c for c in af_cols if "DFE" not in c]
df_clean = df.dropna(subset=af_cols_required).copy()
df_clean.reset_index(drop=True, inplace=True)
print(f"\n  Filas originales    : {len(df)}")
print(f"  Filas tras limpieza : {len(df_clean)}  "
      f"(eliminadas: {len(df) - len(df_clean)})")

# =============================================================================
# 4. ESTADÍSTICAS DESCRIPTIVAS
# =============================================================================
print("\n" + "=" * 70)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 70)
desc = df_clean[af_cols + outcome_cols].describe().T.round(4)
desc.index.name = "Variable"
print(desc.to_string())

# =============================================================================
# 5. CATEGORIZACIÓN AF/DFE  (Bajo / Medio / Alto)
# =============================================================================
print("\n" + "=" * 70)
print("CATEGORIZACIÓN AF/DFE — Percentiles 0-33 / 33-66 / 66-100")
print("=" * 70)

cat_summary_rows = []
cat_col_map = {}   # col → nombre de la columna categoría

for label, col in AF_VARS.items():
    q33 = df_clean[col].quantile(0.33)
    q66 = df_clean[col].quantile(0.66)

    cat_col = col + " [cat]"
    cat_col_map[col] = cat_col
    df_clean[cat_col] = pd.cut(
        df_clean[col],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["Bajo", "Medio", "Alto"],
    )

    counts = df_clean[cat_col].value_counts().sort_index()
    print(f"\n  {label}")
    print(f"    p33 = {q33:.6f}  |  p66 = {q66:.6f}")
    for cat, n in counts.items():
        pct = n / len(df_clean) * 100
        print(f"    {cat:5s}: n={n:4d}  ({pct:.1f}%)")
        cat_summary_rows.append({
            "Variable": label, "Columna": col, "Categoría": cat,
            "p33": round(q33, 6), "p66": round(q66, 6),
            "n": n, "%": round(pct, 1),
        })

cat_summary_df = pd.DataFrame(cat_summary_rows)

# =============================================================================
# 6. GUARDAR BASE LIMPIA
# =============================================================================
df_clean.to_excel(BD_OUT, index=False)
print(f"\nBase limpia guardada: {BD_OUT}  |  {df_clean.shape}")

# =============================================================================
# 7. ANÁLISIS DE COLINEALIDAD ENTRE VARIABLES AF/DFE
# =============================================================================
print("\n" + "=" * 70)
print("COLINEALIDAD — Correlación y VIF entre Variables AF/DFE")
print("=" * 70)

# 7.1 Correlación de Pearson
corr = df_clean[af_cols].corr()
corr.index   = af_labels
corr.columns = af_labels
print("\nMatriz de correlación de Pearson:")
print(corr.round(4).to_string())

# Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
    vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5,
    xticklabels=af_labels, yticklabels=af_labels,
)
ax.set_title("Correlación entre Variables AF/DFE", fontsize=13, fontweight="bold")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
heatmap_path = RES_DIR / "correlacion_AF_DFE.png"
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nHeatmap guardado: {heatmap_path}")

# 7.2 VIF
sub_af = df_clean[af_cols].dropna()
X_vif  = sm.add_constant(sub_af)
vif_vals = [variance_inflation_factor(X_vif.values, i + 1) for i in range(len(af_cols))]
vif_df = pd.DataFrame({
    "Etiqueta": af_labels,
    "Columna" : af_cols,
    "VIF"     : [round(v, 4) for v in vif_vals],
})
print("\nFactor de Inflación de Varianza (VIF):")
print("  Referencia: >5 moderado | >10 severo")
print(vif_df.to_string(index=False))

# =============================================================================
# 8. REGRESIONES LINEALES SIMPLES  (AF/DFE → cada outcome)
# =============================================================================
print("\n" + "=" * 70)
print(f"REGRESIONES LINEALES  "
      f"({len(af_cols)} predictores × {len(outcome_cols)} outcomes = "
      f"{len(af_cols)*len(outcome_cols)} modelos)")
print("=" * 70)
print("Nota: Sexo hijo y Región son categóricas — OLS es exploratorio aquí.\n")

results_rows = []

for af_col, af_label in zip(af_cols, af_labels):
    for out_col, out_label, out_grupo in zip(outcome_cols, outcome_lbl, outcome_grp):
        sub = df_clean[[af_col, out_col]].dropna()
        n   = len(sub)
        if n < 10:
            print(f"  SKIP [{af_label} → {out_label}]: n={n} insuficiente")
            continue
        try:
            X     = sm.add_constant(sub[af_col])
            model = sm.OLS(sub[out_col], X).fit()
            ci    = model.conf_int()
            pv    = model.pvalues[af_col]
            sig   = ("***" if pv < 0.001 else
                     "**"  if pv < 0.01  else
                     "*"   if pv < 0.05  else "ns")
            results_rows.append({
                "Variable AF/DFE": af_label,
                "Outcome"        : out_label,
                "Grupo"          : out_grupo,
                "n"              : int(model.nobs),
                "beta"           : round(model.params[af_col], 6),
                "SE"             : round(model.bse[af_col], 6),
                "t"              : round(model.tvalues[af_col], 4),
                "p-valor"        : round(pv, 6),
                "IC95_inf"       : round(ci.loc[af_col, 0], 6),
                "IC95_sup"       : round(ci.loc[af_col, 1], 6),
                "R2"             : round(model.rsquared, 4),
                "R2_ajust"       : round(model.rsquared_adj, 4),
                "Significancia"  : sig,
            })
        except Exception as e:
            print(f"  ERROR [{af_label} → {out_label}]: {e}")

df_results = pd.DataFrame(results_rows)
n_sig = (df_results["Significancia"] != "ns").sum()
print(f"Completadas: {len(df_results)}  |  Significativas (p<0.05): {n_sig}")

# Imprimir resumen por predictor
cols_show = ["Outcome", "Grupo", "beta", "p-valor", "R2", "Significancia"]
for af_label in af_labels:
    sub = df_results[df_results["Variable AF/DFE"] == af_label]
    n_s = (sub["Significancia"] != "ns").sum()
    print(f"\n  {af_label}  —  {n_s}/{len(sub)} significativas")
    print(sub[cols_show].to_string(index=False))

# =============================================================================
# 9. EXPORTAR A EXCEL
# =============================================================================
print("\n" + "=" * 70)
print("EXPORTANDO RESULTADOS")
print("=" * 70)

# Nombres de hoja por predictor (max 31 chars Excel)
sheet_per_af = {
    "Total AF (suple+pan)" : "Reg_TotalAF",
    "AF Suple 1T"          : "Reg_AF_Suple_1T",
    "AF Pan"               : "Reg_AF_Pan",
    "Total DFE (suple+pan)": "Reg_TotalDFE",
}

sig_only = df_results[df_results["Significancia"] != "ns"].sort_values("p-valor")

with pd.ExcelWriter(RES_OUT, engine="openpyxl") as writer:

    # ── Regresiones ──────────────────────────────────────────────────────────
    df_results.to_excel(writer, sheet_name="Resumen_Regresiones", index=False)
    for af_label, sheet in sheet_per_af.items():
        df_results[df_results["Variable AF/DFE"] == af_label].to_excel(
            writer, sheet_name=sheet, index=False)
    sig_only.to_excel(writer, sheet_name="Solo_Significativas", index=False)

    # ── Colinealidad ─────────────────────────────────────────────────────────
    vif_df.to_excel(writer, sheet_name="VIF", index=False)
    corr.to_excel(writer, sheet_name="Correlacion_AF_DFE")

    # ── Categorización ───────────────────────────────────────────────────────
    cat_summary_df.to_excel(writer, sheet_name="Categorizacion_AF", index=False)

    # ── Descriptivas ─────────────────────────────────────────────────────────
    desc.to_excel(writer, sheet_name="Estadisticas_Descriptivas")

    # ── Base limpia ──────────────────────────────────────────────────────────
    df_clean.to_excel(writer, sheet_name="Base_Limpia", index=False)

print(f"\nArchivo guardado: {RES_OUT}")
print("\nHojas disponibles:")
import openpyxl
wb = openpyxl.load_workbook(RES_OUT, read_only=True)
for s in wb.sheetnames:
    print(f"  - {s}")
wb.close()

# =============================================================================
# 10. GRÁFICO — Regresión Lineal: Total AF vs Edad madre
# =============================================================================
print("\n" + "=" * 70)
print("GRÁFICO — Total AF (suple+pan) → Edad madre")
print("=" * 70)

_x_col = "Total mg/d AF  suple y pan"
_y_col = "Edad madre"

_sub = df_clean[[_x_col, _y_col]].dropna()
_X   = sm.add_constant(_sub[_x_col])
_mod = sm.OLS(_sub[_y_col], _X).fit()
_row = df_results[
    (df_results["Variable AF/DFE"] == "Total AF (suple+pan)") &
    (df_results["Outcome"]         == "Edad madre")
].iloc[0]

_x_range = np.linspace(_sub[_x_col].min(), _sub[_x_col].max(), 200)
_y_pred  = _mod.params["const"] + _mod.params[_x_col] * _x_range

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(_sub[_x_col], _sub[_y_col], alpha=0.45, s=30,
           color="#4C72B0", edgecolors="white", linewidths=0.4, label="Observaciones")
ax.plot(_x_range, _y_pred, color="#C44E52", linewidth=2, label="Línea de regresión")

_sig = _row["Significancia"]
_ann = (f"β = {_row['beta']:.4f}\n"
        f"R² = {_row['R2']:.4f}\n"
        f"p = {_row['p-valor']:.4f}  {_sig}\n"
        f"n = {int(_row['n'])}")
ax.text(0.97, 0.97, _ann, transform=ax.transAxes,
        fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))

ax.set_xlabel("Total AF (suplementos + pan)  [mg/día]", fontsize=11)
ax.set_ylabel("Edad madre  [años]", fontsize=11)
ax.set_title("Regresión Lineal: Total AF → Edad madre", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()

_reg_plot_path = RES_DIR / "reg_TotalAF_EdadMadre.png"
plt.savefig(_reg_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Gráfico guardado: {_reg_plot_path}")

print("\n" + "=" * 70)
print("ANÁLISIS COMPLETADO")
print("=" * 70)
print(f"  BD limpia         : {BD_OUT}")
print(f"  Resultados Excel  : {RES_OUT}")
print(f"  Heatmap           : {heatmap_path}")
print(f"  Gráfico regresión : {_reg_plot_path}")
