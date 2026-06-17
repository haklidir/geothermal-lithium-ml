"""
make_fig9_workflow.py — Generate the ML workflow schematic (Figure 9).

Uses Graphviz to render the inputs → 4 regressors → outputs diagram in
the same visual style as the published figure. Updated to include the
Ridge baseline (4 models total, mor oval) and "empirical prediction
interval" terminology in the output box.

Requires:
    pip install graphviz
    plus a system-level Graphviz install (`apt-get install graphviz` or
    equivalent), since the python package is a thin wrapper around the
    command-line tool.

Output: results/figures/Fig9_workflow.png (300+ dpi)
"""

import argparse
import pathlib
from graphviz import Digraph


def build_fig9(out_path):
    dot = Digraph("inputs_outputs_ml", format="png")
    dot.attr(rankdir="LR", splines="ortho", nodesep="0.7", ranksep="1.0",
             fontname="Arial")
    dot.graph_attr.update(dpi="600")

    # Input box (record with two sections)
    dot.attr("node", shape="record", style="rounded,filled",
             color="#c78b2b", fillcolor="#ffe2a8",
             fontsize="11", penwidth="1.4")
    dot.node(
        "IN",
        """{
        Inputs|
        {<chem> Primary geochemistry:\\lT (\u00b0C), pH, EC (\u00b5S/cm), Na, K, Cl, B, SiO2 (mg/L)\\l|
         <eng> Engineered ratios:\\lNa/K, Na+K, Cl/EC\\l}
        }"""
    )

    # ML models cluster (dashed box) with 4 ovals
    with dot.subgraph(name="cluster_models") as cm:
        cm.attr(label="ML models", labelloc="t", fontsize="12",
                style="rounded", color="#bbbbbb", penwidth="1.0")
        cm.attr("node", shape="ellipse", style="filled",
                fontsize="11", penwidth="1.4")
        cm.node("DNN",   "DNN",            color="#2b5caa", fillcolor="#a8c8ff")
        cm.node("Ridge", "Ridge",          color="#7a4c9e", fillcolor="#dcc8ee")
        cm.node("RF",    "Random Forest",  color="#3b7b3a", fillcolor="#bfe8bf")
        cm.node("XGB",   "XGBoost",        color="#c25757", fillcolor="#ffc2c2")
        cm.attr(rank="same")

    # Output box
    dot.attr("node", shape="box", style="rounded,filled",
             color="#8a1f1f", fillcolor="#ffb3b3",
             fontsize="11", penwidth="1.6")
    dot.node("OUT", "Predicted Li (mg/L)\n+ 90% empirical prediction interval")

    # Input → models (model-colored)
    dot.edge("IN", "DNN",   color="#2b5caa", penwidth="1.6")
    dot.edge("IN", "Ridge", color="#7a4c9e", penwidth="1.6")
    dot.edge("IN", "RF",    color="#3b7b3a", penwidth="1.6")
    dot.edge("IN", "XGB",   color="#c25757", penwidth="1.6")

    # Models → output
    dot.edge("DNN",   "OUT", color="#2b5caa", penwidth="1.8")
    dot.edge("Ridge", "OUT", color="#7a4c9e", penwidth="1.8")
    dot.edge("RF",    "OUT", color="#3b7b3a", penwidth="1.8")
    dot.edge("XGB",   "OUT", color="#c25757", penwidth="1.8")

    dot.attr(label="Inputs and outputs of the ML models",
             labelloc="t", fontsize="14")

    out_stem = str(out_path).rsplit(".", 1)[0]
    rendered = dot.render(out_stem, cleanup=True)
    print(f"Saved: {rendered}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="results/figures/Fig9_workflow.png",
        help="Output path for the rendered figure",
    )
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_fig9(out)
