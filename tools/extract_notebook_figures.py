import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "FedProxyStart.ipynb"
FIGURE_ROOT = ROOT / "results" / "figures"

# Notebook cell/output locations for each saved plot.
FIGURES = {
    (34, 0): "core/uq_core_results.png",
    (34, 2): "core/uq_selective_and_gate_results.png",
    (36, 0): "uncertainty/mcar/predictive_uncertainty_decomposition_mcar.png",
    (36, 2): "uncertainty/mcar/epistemic_uncertainty_mcar.png",
    (36, 4): "uncertainty/mcar/epistemic_share_mcar.png",
    (36, 6): "uncertainty/mcar/macro_f1_and_epistemic_mcar.png",
    (36, 8): "uncertainty/mcar/aleatoric_uncertainty_mcar.png",
    (36, 10): "uncertainty/mcar/auarc_epistemic_mcar.png",
    (36, 14): "uncertainty/mnar/predictive_uncertainty_decomposition_mnar.png",
    (36, 16): "uncertainty/mnar/epistemic_uncertainty_mnar.png",
    (36, 18): "uncertainty/mnar/epistemic_share_mnar.png",
    (36, 20): "uncertainty/mnar/macro_f1_and_epistemic_mnar.png",
    (36, 22): "uncertainty/mnar/aleatoric_uncertainty_mnar.png",
    (36, 24): "uncertainty/mnar/auarc_epistemic_mnar.png",
    (37, 1): "diagnostics/calibration_all_modality_missingness.png",
    (37, 3): "diagnostics/intended_vs_realized_missingness.png",
    (37, 5): "diagnostics/modality_impact_at_50pct.png",
    (37, 7): "diagnostics/gate_behavior_at_50pct.png",
    (37, 9): "diagnostics/mean_gate_weight_mcar.png",
    (37, 11): "diagnostics/mean_gate_weight_mnar.png",
    (37, 13): "diagnostics/epistemic_error_recognition.png",
    (41, 0): "presentation/01_degradation_three_panel.png",
    (41, 2): "presentation/02_uq_bald_six_panel.png",
    (41, 4): "presentation/03_aleatoric_vs_bald.png",
    (45, 1): "diagnostics/multiseed_modality_importance.png",
    (49, 0): "diagnostics/paired_mnar_minus_mcar.png",
    (50, 0): "presentation/01_degradation_three_panel_natural.png",
    (50, 3): "presentation/02_uq_bald_six_panel_natural.png",
    (54, 1): "sweeps/mu_sweep_macro_f1_and_ece.png",
    (58, 3): "sweeps/local_epochs_macro_f1_and_ece.png",
}


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    written = 0
    for (cell_index, output_index), relative_path in FIGURES.items():
        output = notebook["cells"][cell_index]["outputs"][output_index]
        encoded = output.get("data", {}).get("image/png")
        if encoded is None:
            raise RuntimeError(f"No PNG found at cell {cell_index}, output {output_index}")
        if isinstance(encoded, list):
            encoded = "".join(encoded)
        destination = FIGURE_ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(base64.b64decode(encoded))
        written += 1
    print(f"Extracted {written} figures to {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
