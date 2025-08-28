#!/usr/bin/env python3

"""
Dash + Scanpy UMAP app
- Press "Run Pipeline" to fetch example data, run the pipeline, and render UMAP.
- Use the slider to choose the number of highly variable genes.
"""

# ---- Imports ----
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import pooch
import scvelo as scv


from dash import Dash, html, dcc, Input, Output, State, no_update
import plotly.graph_objects as go

# ---- Scanpy settings ----
sc.settings.set_figure_params(dpi=50, facecolor="white")
sc.settings.verbosity = 2


# ---- Prepare example data once at startup (downloaded & cached by pooch) ----
def load_example_adata(dataset) -> ad.AnnData:
    if dataset == "PBMC3k":
        base_url = "doi:10.6084/m9.figshare.16447278.v1/"
        EXAMPLE_DATA = pooch.create(
            path=pooch.os_cache("scverse_tutorials"),
            base_url=base_url,
        )
        EXAMPLE_DATA.load_registry_from_doi()
        h5ad_path = EXAMPLE_DATA.fetch("scanpy-pbmc3k.h5ad")
        adata = sc.read_h5ad(h5ad_path)
                
        # making a series containing the cluster names and associated cell IDs
        pbmc3k = sc.datasets.pbmc3k_processed()
        cluster_names_series = pd.Series(pbmc3k.obs["louvain"])
        
        # resetting the adata to its raw counts
        adata = adata.raw.to_adata()
        adata.obs["cell_type"] = adata.obs.index.map(lambda x: cluster_names_series[x])
        
    elif dataset == "DG":
        adata = scv.datasets.dentategyrus()
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # storing raw counts into a layer for reproducible re-runs
    adata.layers["counts"] = adata.X.copy()

    # QC flags
    adata.var["mt"] = adata.var_names.str.startswith("MT-")          # human mito
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

    # Basic filtering (same as your script)
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)

    # Save raw counts for reproducible re-runs
    adata.layers["counts"] = adata.X.copy()

    return adata


def run_pipeline(dataset: str, n_hvg: int, n_pcas: int, n_neighbs: int) -> ad.AnnData:
    """Run the Scanpy pipeline, parameterized by n_hvg, n_pcas, and n_neighbs."""
    # Work on a fresh copy so multiple runs are consistent
    adata = load_example_adata(dataset).copy()

    # Reset to raw counts, make sure values are finite, then normalize + log1p
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata)   # target_sum default is fine
    sc.pp.log1p(adata)

    # HVGs (batch-aware across 'sample')
    # Only use batch_key if multiple samples exist
    batch_key = "sample" if ("sample" in adata.obs) else None # and adata.obs["sample"].nunique() > 1
    sc.pp.highly_variable_genes(adata, n_top_genes=int(n_hvg), batch_key=batch_key)

    # IMPORTANT: subset to HVGs so the slider actually affects PCA/UMAP
    adata = adata[:, adata.var["highly_variable"]].copy()

    # PCA → Neighbors → UMAP
    
    if n_pcas >= n_hvg:
        n_pcas = n_hvg - 1 # makes sure num PCAs does not exceed num HVGs
        
    # Adding variable to select number of PCAs
    sc.tl.pca(adata, n_comps=int(n_pcas))

    # Adding variable to select number for KNN
    sc.pp.neighbors(adata, n_neighbors=int(n_neighbs))
    
    
    sc.tl.umap(adata) # Has a default random state of zero
    
    # Leiden clustering if not already included in the data
    if "leiden" not in adata.obs:
        sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)

    return adata


def umap_figure_from_adata(adata: ad.AnnData, dataset) -> go.Figure: ### DATASET
    """Convert adata.obsm['X_umap'] + adata.obs['leiden'] into a Plotly scatter."""
    if "X_umap" not in adata.obsm_keys():
        raise ValueError("UMAP coordinates not found.")

    if dataset == "PBMC3k":
        cell_type_index = "cell_type"
    elif dataset == "DG":
        cell_type_index = "clusters"
    coords = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata.obs_names)
    coords["cluster"] = adata.obs[cell_type_index].astype(str).values

    fig = go.Figure()
    for cluster in sorted(coords["cluster"].unique()):
        sub = coords[coords["cluster"] == cluster]
        fig.add_trace(
            go.Scattergl(
                x=sub["UMAP1"],
                y=sub["UMAP2"],
                mode="markers",
                name=str(cluster),
                marker={"size": 4, "opacity": 0.8},
                hovertemplate=f"cluster: {cluster}<extra></extra>"
            )
        )

    fig.update_layout(
        title=f"UMAP colored by cell type",
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        template="plotly_white",
        legend_title_text="cell type label",
        margin=dict(l=20, r=20, t=40, b=20),
        dragmode="pan",
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1, showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    return fig


# ---- Dash app ----
app = Dash(__name__)
server = app.server  # for production servers (e.g., gunicorn)

app.title = "Interactive Scanpy"

app.layout = html.Div(
    style={"maxWidth": "900px", "margin": "0 auto", "fontFamily": "system-ui, sans-serif", "marginBottom": "24px"},
    children=[
        html.H2("Interactive scRNA-seq Scanpy pipeline"),
        html.P(
            "Toggle HVG, PC, and KNN parameters and produce the resulting UMAP using the scanpy analysis pipeline."
        ),
        html.Div(
            [
                html.Label("Dataset"),
                dcc.Dropdown(
                    id="dataset-dropdown",
                    options=[
                        {"label": "PBMC3k", "value": "PBMC3k"},
                        {"label": "Dentate Gyrus neurogenesis", "value": "DG"}
                    ],
                    value="PBMC3k",       # default
                    clearable=False,
                ),
            ],
            style={"marginBottom": "24px"},
        ),
        html.Div(
            style={"marginBottom": "24px"},
            children=html.P(
                "New datasets coming soon...",
                style={"fontStyle": "italic"}
                )
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr auto", "gap": "16px", "alignItems": "center", "marginBottom": "24px"},
            children=[
                html.Div(
                    children=[
                        html.Label("Highly variable genes"),
                        dcc.Slider(
                            id="hvg-slider",
                            min=5,
                            max=15000,
                            step=100,
                            value=2000,  # default as requested
                            marks={10: "10", 500: "500", 1000: "1000", 2000: "2000", 3000: "3000", 4000: "4000", 5000: "5000", 10000: "10000", 15000: "15000"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ]
                ),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr auto", "gap": "16px", "alignItems": "center", "marginBottom": "24px"},
            children=[
                html.Div(
                    children=[
                        html.Label("Principal components"),
                        dcc.Slider(
                            id="pca-slider",
                            min=2,
                            max=100,
                            step=1,
                            value=50,  # default as requested
                            marks={2: "2", 10: "10", 20: "20", 50: "50", 100: "100"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ]
                ),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr auto", "gap": "16px", "alignItems": "center", "marginBottom": "24px"},
            children=[
                html.Div(
                    children=[
                        html.Label("Number of nearest neighbors (for UMAP)"),
                        dcc.Slider(
                            id="neighbor-slider",
                            min=2,
                            max=100,
                            step=1,
                            value=15,  # default as requested
                            marks={2: "2", 10: "10", 15: "15", 25: "25", 30: "30", 50: "50", 100: "100"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ]
                ),
            ],
        ),
        html.Div(
            html.Button("Run Pipeline", id="run-btn", n_clicks=0, style={"height": "44px", "minWidth": "200px"}),
            style={"display": "flex", "justifyContent": "center", "margin": "8px 0 24px"}
            ),
        html.Div(id="status", style={"marginTop": "10px", "fontSize": "0.95rem", "color": "#444"}),
        dcc.Loading(
            id="loading",
            type="default",
            children=dcc.Graph(
                id="umap-graph",
                figure=go.Figure(layout={"template": "plotly_white"}),
                style={"height": "640px", "marginTop": "10px"},
                config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]},
            ),
        ),
        html.Div([
            html.P([
                "Dentate Gyrus neurogenesis dataset from ",
                html.A("scVelo", href="https://scvelo.readthedocs.io/en/stable/scvelo.datasets.dentategyrus.html", target="_blank", rel="noopener noreferrer"),
                ". PBMC3k dataset from ",
                html.A("scanpy", href="https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html", target="_blank", rel="noopener noreferrer"),
                "."
            ]),
            html.P("Tip: rerun with different HVG, PC, and KNN settings to see stability."),
            html.P("Typical ranges:"),
            html.P("HVG: 1,000–5,000 depending on tissue & depth  |  PCs: 10-100, increasing with dataset size  |  KNN: 5-15 for small datasets, but can go up to 60s for very large datasets"),
            html.P("Warning: If set number of HVGs > PCs, PCs will be automatically set to HVGs - 1")
            ], style={"marginTop": "8px", "color": "#666", "fontSize": "0.9rem"},
        ),
    ],
)


@app.callback(
    Output("umap-graph", "figure"),
    Output("status", "children"),
    Input("run-btn", "n_clicks"),
    State("dataset-dropdown", "value"),
    State("hvg-slider", "value"),
    State("pca-slider", "value"),
    State("neighbor-slider", "value"),
    prevent_initial_call=True,
)
def on_run(n_clicks: int, dataset: str, n_hvg: int,  n_pcas: int, n_neighbs: int):
    try:
        status = f"Running pipeline on {dataset} (n_hvg={int(n_hvg)}, PCs={int(n_pcas)}, kNN={int(n_neighbs)})…"
        # Run the analysis
        adata = run_pipeline(dataset=dataset, n_hvg=int(n_hvg), n_pcas=int(n_pcas), n_neighbs=int(n_neighbs))
        fig = umap_figure_from_adata(adata, dataset)
        status = f"Done. Dataset: {dataset}  |  Cells: {adata.n_obs:,}  |  Genes (HVGs): {adata.n_vars:,}  |  PCs: {min(n_pcas, adata.n_vars)}  |  kNN: {n_neighbs}"
        return fig, status
    except Exception as e:
        # Report the error to the UI
        err = f"Error: {type(e).__name__}: {e}"
        return no_update, err


if __name__ == "__main__":
    # Bind to localhost by default; set debug=True if you like
    app.run(host="127.0.0.1", port=8050, debug=False)
