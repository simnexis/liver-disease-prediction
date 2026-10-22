"""Exploratory Data Analysis (EDA) visualizations and summary statistics."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_eda_figures(df: pd.DataFrame, output_dir: str = "reports/figures") -> dict:
    """Generate high-resolution EDA figures saved to reports/figures/."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Class Balance
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["target"].value_counts()
    labels = ["Liver Disease (1)", "No Liver Disease (0)"]
    bars = ax.bar([labels[0], labels[1]], [counts.get(1, 0), counts.get(0, 0)], color=["#e74c3c", "#3498db"], width=0.5)
    total = len(df)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 5,
            f"{h} ({h/total:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("Target Class Distribution (Indian Liver Patient Dataset)", fontsize=12, pad=12)
    ax.set_ylabel("Patient Count")
    ax.set_ylim(0, total * 0.85)
    plt.tight_layout()
    class_fig_path = os.path.join(output_dir, "eda_class_balance.png")
    plt.savefig(class_fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Skewed Blood Markers Distributions
    skewed_markers = [
        ("total_bilirubin", "Total Bilirubin (TB)"),
        ("direct_bilirubin", "Direct Bilirubin (DB)"),
        ("alkaline_phosphatase", "Alkaline Phosphatase (Alkphos)"),
        ("alamine_aminotransferase", "Alamine Aminotransferase (SGPT)"),
        ("aspartate_aminotransferase", "Aspartate Aminotransferase (SGOT)"),
        ("albumin_globulin_ratio", "A/G Ratio"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, (col, title) in enumerate(skewed_markers):
        if col in df.columns:
            sns.histplot(
                df[col].dropna(),
                kde=True,
                ax=axes[i],
                color="#8e44ad",
                bins=25,
            )
            skew_val = df[col].skew()
            axes[i].set_title(f"{title}\nSkew: {skew_val:.2f}", fontsize=10)
            axes[i].set_xlabel("Value")
    plt.suptitle("Biomarker Distribution Skewness (Routine Hepatic Panel)", fontsize=14, y=1.02)
    plt.tight_layout()
    dist_fig_path = os.path.join(output_dir, "eda_distributions.png")
    plt.savefig(dist_fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Correlation Matrix (Collinearity)
    num_cols = [
        "age",
        "total_bilirubin",
        "direct_bilirubin",
        "alkaline_phosphatase",
        "alamine_aminotransferase",
        "aspartate_aminotransferase",
        "total_proteins",
        "albumin",
        "albumin_globulin_ratio",
    ]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        "Biomarker Correlation Matrix\n(Highlighting Severe Collinearity: TB vs DB r=0.87, SGPT vs SGOT r=0.79)",
        fontsize=11,
        pad=12,
    )
    plt.tight_layout()
    corr_fig_path = os.path.join(output_dir, "eda_correlation_matrix.png")
    plt.savefig(corr_fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    # 4. Demographic Breakdown (Gender & Age Imbalance)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Gender imbalance
    gender_counts = df["gender"].value_counts()
    ax1.pie(
        gender_counts.values,
        labels=[f"{k} ({v})" for k, v in gender_counts.items()],
        autopct="%1.1f%%",
        colors=["#2980b9", "#e84393"],
        startangle=140,
        explode=(0.05, 0),
    )
    ax1.set_title("Cohort Gender Distribution\n(Notable Imbalance: ~75% Male)", fontsize=11)

    # Age by Disease Status
    sns.boxplot(
        x="target",
        y="age",
        hue="target",
        legend=False,
        data=df,
        palette=["#3498db", "#e74c3c"],
        ax=ax2,
        width=0.4,
    )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["No Disease (0)", "Liver Disease (1)"])
    ax2.set_title("Age Distribution by Disease Status", fontsize=11)
    ax2.set_ylabel("Age (Years)")
    plt.tight_layout()
    demographics_path = os.path.join(output_dir, "eda_demographics.png")
    plt.savefig(demographics_path, dpi=200, bbox_inches="tight")
    plt.close()

    tb_db_corr = float(df["total_bilirubin"].corr(df["direct_bilirubin"]))
    sgpt_sgot_corr = float(df["alamine_aminotransferase"].corr(df["aspartate_aminotransferase"]))

    eda_summary = {
        "class_proportions": {
            "disease_positive_pct": float(round((df["target"] == 1).mean() * 100, 2)),
            "disease_negative_pct": float(round((df["target"] == 0).mean() * 100, 2)),
        },
        "gender_proportions": {
            "male_pct": float(round((df["gender"] == "Male").mean() * 100, 2)),
            "female_pct": float(round((df["gender"] == "Female").mean() * 100, 2)),
        },
        "collinearities": {
            "tb_db_correlation": round(tb_db_corr, 4),
            "sgpt_sgot_correlation": round(sgpt_sgot_corr, 4),
        },
        "skewness": {
            col: round(float(df[col].skew()), 2)
            for col in num_cols
        },
    }
    return eda_summary
