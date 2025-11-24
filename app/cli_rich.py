# app/cli_rich.py
# A polished terminal CLI using Rich for presentation and interactivity

import os, sys
from joblib import load
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, FloatPrompt
from app.utils import DIABETES_FEATURES, HEART_FEATURES
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT,"models")
console = Console()

def load_pipeline(path):
    d = load(path)
    if isinstance(d, dict) and "model" in d and "scaler" in d:
        return d["model"], d["scaler"], d["features"]
