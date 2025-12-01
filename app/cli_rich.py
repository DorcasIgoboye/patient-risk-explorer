# app/cli_rich.py
# A polished terminal CLI using Rich for presentation and interactivity

import os
from joblib import load
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, FloatPrompt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
console = Console()

def load_pipeline(path):
    d = load(path)
    if isinstance(d, dict) and "model" in d and "scaler" in d and "features" in d:
        return d["model"], d["scaler"], d["features"]
    raise ValueError(f"Invalid pipeline at {path}")

def prompt_for_features(features):
    console.print(Panel.fit("Enter patient values (leave blank for 0)"))
    values = []
    for f in features:
        val = Prompt.ask(f, default="")
        try:
            values.append(float(val) if val != "" else 0.0)
        except ValueError:
            values.append(0.0)
    return np.array(values, dtype=float)

def main():
    model_d, scaler_d, feat_d = load_pipeline(os.path.join(MODEL_DIR, "diabetes.joblib"))
    model_h, scaler_h, feat_h = load_pipeline(os.path.join(MODEL_DIR, "heart.joblib"))

    console.print(Panel.fit("Patient Risk Explorer (CLI)", style="bold cyan"))
    console.print(f"[green]Diabetes features:[/green] {feat_d}")
    console.print(f"[green]Heart features:[/green] {feat_h}")

    console.print("\n[bold]Enter values for diabetes model[/bold]")
    Xd = prompt_for_features(feat_d).reshape(1, -1)
    pd_prob = float(model_d.predict_proba(scaler_d.transform(Xd))[0, 1])

    console.print("\n[bold]Enter values for heart model[/bold]")
    Xh = prompt_for_features(feat_h).reshape(1, -1)
    ph_prob = float(model_h.predict_proba(scaler_h.transform(Xh))[0, 1])

    table = Table(title="Estimated Risks")
    table.add_column("Condition", style="bold")
    table.add_column("Risk (%)")
    table.add_row("Diabetes", f"{pd_prob*100:.1f}")
    table.add_row("Heart disease", f"{ph_prob*100:.1f}")
    console.print(table)

if __name__ == "__main__":
    main()
