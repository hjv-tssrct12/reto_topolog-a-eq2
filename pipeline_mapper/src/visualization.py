"""Phase 7: HTML and PNG visualisations of the Mapper graph."""

import logging
from pathlib import Path
from typing import Optional

import kmapper as km
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import CAT_COLORS, CAT_ORDER, OUTPUTS_DIR
from src.mapper_multiscale import Graph, _graph_to_nx

log = logging.getLogger(__name__)


def _build_tooltips(graph: Graph, df_analysis: pd.DataFrame, n_patients: int) -> np.ndarray:
    """Build HTML tooltip string per patient (indexed by original patient id).

    kmapper's visualize() indexes custom_tooltips by patient id, not node id.

    Parameters
    ----------
    graph       : kmapper Graph
    df_analysis : output of analysis.analyze_nodes
    n_patients  : total number of patients in the dataset.

    Returns
    -------
    tooltips : np.ndarray of str, shape (n_patients,).
    """
    tooltips = np.empty(n_patients, dtype=object)
    tooltips[:] = ""

    tip_map = {}
    for _, row in df_analysis.iterrows():
        sig = "★ significativo" if row.get("nodo_significativo", False) else ""
        tip_map[row["Nodo"]] = (
            f"<b>{row['Nodo']}</b><br>"
            f"N={row['N']}<br>"
            f"AF medio={row['AF_media']:.2f} mg/día "
            f"[{row.get('CI_inf', '?'):.2f}–{row.get('CI_sup', '?'):.2f}]<br>"
            f"p_FDR={row.get('p_FDR', float('nan')):.4f}<br>"
            f"Cliff delta={row.get('cliffs_delta', 0):.3f}<br>"
            f"Bajo={row['pct_Bajo']}% Medio={row['pct_Medio']}% Alto={row['pct_Alto']}%<br>"
            f"{sig}"
        )

    for node_id, members in graph["nodes"].items():
        tip = tip_map.get(node_id, node_id)
        for patient_id in members:
            if patient_id < n_patients:
                tooltips[patient_id] = tip

    return tooltips


def make_html_continuous(
    mapper: km.KeplerMapper,
    graph: Graph,
    lente_norm: np.ndarray,
    df_analysis: pd.DataFrame,
    output_path: Path,
    n_patients: int = 0,
) -> None:
    """HTML graph coloured by continuous AF value."""
    tooltips = _build_tooltips(graph, df_analysis, n_patients)
    mapper.visualize(
        graph,
        color_values=lente_norm.ravel(),
        color_function_name="Total AF mg/día (normalizado)",
        title="Mapper TDA — Lente: UMAP 2D | Color: AF continuo",
        custom_tooltips=tooltips,
        path_html=str(output_path),
    )
    log.info("HTML (continuo): %s", output_path)


def make_html_categorical(
    mapper: km.KeplerMapper,
    graph: Graph,
    cat_enc: np.ndarray,
    df_analysis: pd.DataFrame,
    output_path: Path,
    n_patients: int = 0,
) -> None:
    """HTML graph coloured by AF category (0=Bajo, 1=Medio, 2=Alto)."""
    tooltips = _build_tooltips(graph, df_analysis, n_patients)
    mapper.visualize(
        graph,
        color_values=cat_enc,
        color_function_name="Categoría AF (0=Bajo · 1=Medio · 2=Alto)",
        title="Mapper TDA — Lente: UMAP 2D | Color: Categoría AF",
        custom_tooltips=tooltips,
        path_html=str(output_path),
    )
    log.info("HTML (categoría): %s", output_path)


def make_html_density(
    mapper: km.KeplerMapper,
    graph: Graph,
    df_analysis: pd.DataFrame,
    output_path: Path,
    n_patients: int = 0,
) -> None:
    """HTML graph coloured by node size (density of patients)."""
    tooltips = _build_tooltips(graph, df_analysis, n_patients)
    # color_values must be indexed by patient id, not node id
    density_per_patient = np.zeros(n_patients, dtype=float)
    for members in graph["nodes"].values():
        size = float(len(members))
        for pid in members:
            if pid < n_patients:
                density_per_patient[pid] = max(density_per_patient[pid], size)
    node_sizes_norm = MinMaxScaler().fit_transform(density_per_patient.reshape(-1, 1)).ravel()
    mapper.visualize(
        graph,
        color_values=node_sizes_norm,
        color_function_name="Densidad de pacientes (normalizada)",
        title="Mapper TDA — Lente: UMAP 2D | Color: Densidad de nodo",
        custom_tooltips=tooltips,
        path_html=str(output_path),
    )
    log.info("HTML (densidad): %s", output_path)


def make_png_graph(
    graph: Graph,
    color_values: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str = "RdYlBu_r",
) -> None:
    """Static PNG of the Mapper graph using networkx + matplotlib.

    Parameters
    ----------
    graph        : kmapper Graph
    color_values : np.ndarray, shape (n_nodes,) — values used for colouring nodes.
    title        : plot title.
    output_path  : destination PNG path.
    cmap         : matplotlib colormap name.
    """
    G = _graph_to_nx(graph)
    if G.number_of_nodes() == 0:
        log.warning("Empty graph — skipping PNG: %s", output_path)
        return

    node_list = list(graph["nodes"].keys())
    sizes = [max(50, len(graph["nodes"][n]) * 2) for n in node_list]

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42)
    nc = ax.scatter(
        [pos[n][0] for n in node_list],
        [pos[n][1] for n in node_list],
        c=color_values,
        cmap=cmap,
        s=sizes,
        zorder=2,
    )
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
    plt.colorbar(nc, ax=ax, shrink=0.7)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PNG graph: %s", output_path)


def make_composition_bar(df_analysis: pd.DataFrame, output_path: Path) -> None:
    """Stacked bar chart of Bajo/Medio/Alto composition per node."""
    n = len(df_analysis)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))
    x   = np.arange(n)
    bot = np.zeros(n)

    for cat in CAT_ORDER:
        vals = df_analysis[f"pct_{cat}"].values
        ax.bar(x, vals, 0.7, bottom=bot, label=cat,
               color=CAT_COLORS[cat], edgecolor="white", linewidth=0.4)
        bot += vals

    for i, (_, row) in enumerate(df_analysis.iterrows()):
        marker = "★" if row.get("nodo_significativo", False) else ""
        ax.text(i, 102, f"n={int(row['N'])}{marker}", ha="center", va="bottom",
                fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"N{i+1}\nAF={r['AF_media']:.1f}" for i, (_, r) in enumerate(df_analysis.iterrows())],
        fontsize=7, rotation=45, ha="right",
    )
    ax.set_ylabel("Proporción (%)", fontsize=11)
    ax.set_ylim(0, 120)
    ax.set_title("Composición AF por nodo — ★ = nodo significativo (FDR p<0.05 & |Δ|>0.33)",
                 fontsize=11, fontweight="bold")
    ax.legend(title="Categoría AF", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Composition bar chart: %s", output_path)


def make_all_visualizations(
    graph: Graph,
    lente_vals: np.ndarray,
    cat_enc: np.ndarray,
    df_analysis: pd.DataFrame,
) -> None:
    """Generate all 3 HTMLs + 3 PNGs + composition bar."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    mapper = km.KeplerMapper(verbose=0)
    lente_norm = MinMaxScaler().fit_transform(lente_vals.reshape(-1, 1))
    n_patients = len(lente_vals)

    node_list   = list(graph["nodes"].keys())
    n_nodes     = len(node_list)

    # Per-node colour arrays (same order as graph["nodes"])
    af_node_mean  = np.array([lente_vals[list(graph["nodes"][n])].mean() for n in node_list])
    cat_node_mean = np.array([
        np.mean([cat_enc[i] for i in graph["nodes"][n]]) for n in node_list
    ])
    density       = np.array([len(graph["nodes"][n]) for n in node_list], dtype=float)
    density_norm  = MinMaxScaler().fit_transform(density.reshape(-1, 1)).ravel()
    af_norm_nodes = MinMaxScaler().fit_transform(af_node_mean.reshape(-1, 1)).ravel()

    make_html_continuous(mapper, graph, lente_norm, df_analysis,
                         OUTPUTS_DIR / "mapper_AF.html", n_patients)
    make_html_categorical(mapper, graph, cat_enc, df_analysis,
                          OUTPUTS_DIR / "mapper_cat_AF.html", n_patients)
    make_html_density(mapper, graph, df_analysis,
                      OUTPUTS_DIR / "mapper_densidad.html", n_patients)

    make_png_graph(graph, af_norm_nodes,
                   "Mapper — Color: AF medio del nodo",
                   OUTPUTS_DIR / "mapper_AF.png")
    make_png_graph(graph, cat_node_mean,
                   "Mapper — Color: Categoría AF media del nodo",
                   OUTPUTS_DIR / "mapper_cat_AF.png", cmap="viridis")
    make_png_graph(graph, density_norm,
                   "Mapper — Color: Densidad de pacientes",
                   OUTPUTS_DIR / "mapper_densidad.png", cmap="plasma")

    make_composition_bar(df_analysis, OUTPUTS_DIR / "mapper_composicion.png")
