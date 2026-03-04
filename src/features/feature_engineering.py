"""
Feature Engineering — Correlation Analysis & Index Visualizations
Computes top-5 correlated field pairs for each composite index
and generates heatmaps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from src.data.preprocessing import preprocess_data, create_composite_indices


def get_top_correlations(df: pd.DataFrame, cols: list, n: int = 5) -> list:
    """Return top-n correlated column pairs within a given column set."""
    corr_matrix = df[cols].corr().abs()
    pairs = []
    for col_a, col_b in combinations(cols, 2):
        pairs.append((col_a, col_b, corr_matrix.loc[col_a, col_b]))
    return sorted(pairs, key=lambda x: x[2], reverse=True)[:n]


def plot_index_heatmap(df: pd.DataFrame, cols: list, title: str, save_path: str = None):
    """Plot and optionally save a correlation heatmap for an index."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(title, fontsize=14, pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# Column definitions for each index
SUPPORT_COLS = [
    "Does your employer provide mental health benefits as part of healthcare coverage?",
    "Do you know the options for mental health care available under your employer-provided coverage?",
    "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?",
    "Does your employer offer resources to learn more about mental health concerns and options for seeking help?",
    "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:",
]

STIGMA_COLS = [
    "Do you think that discussing a mental health disorder with your employer would have negative consequences?",
    "Would you feel comfortable discussing a mental health disorder with your coworkers?",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?",
    "Do you feel that being identified as a person with a mental health issue would hurt your career?",
    "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?",
]

OPENNESS_COLS = [
    "Do you feel that your employer takes mental health as seriously as physical health?",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?",
    "Do you have medical coverage (private insurance or state-provided) which includes treatment of  mental health issues?",
    "Would you bring up a mental health issue with a potential employer in an interview?",
    "Would you be willing to bring up a physical health issue with a potential employer in an interview?",
]
