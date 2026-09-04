from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def summarize(path: Path, group: str, metrics: list[str]) -> None:
    frame = pd.read_csv(path)
    columns = [name for name in metrics if name in frame.columns]
    print(f"\n{path.name}")
    print(frame.groupby(group)[columns].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    summarize(
        ROOT / "results" / "mu_sweep_reported.csv",
        "mu",
        ["best_validation_macro_f1", "clean_test_macro_f1", "mnar50_macro_f1"],
    )
    summarize(
        ROOT / "results" / "local_epochs_reported.csv",
        "local_epochs",
        ["best_validation_macro_f1", "clean_test_macro_f1", "mnar50_macro_f1"],
    )
