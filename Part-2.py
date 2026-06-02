"""
==============================================================================
Part 2: Success Rate Logic & Cohort Analysis
==============================================================================
"""

import os
import ast
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_FILE = "SampleDateExtract.xlsx"
SHEET_NAME = "1000_inteventional_trials"
OUTPUT_DIR = "output"

PHASE_STANDARD_MAP = {
    "PHASE1":        "Phase I",
    "PHASE2":        "Phase II",
    "PHASE3":        "Phase III",
    "PHASE4":        "Phase IV",
    "PHASE1/PHASE2": "Phase I/II",
    "PHASE2/PHASE3": "Phase II/III",
    "EARLY_PHASE1":  "Early Phase I",
}

PHASE_NUMERIC_MAP = {
    "EARLY_PHASE1":  0.5,   # sub-phase of Phase I; 0.5 reflects proximity to Phase I
    "PHASE1":        1.0,
    "PHASE1/PHASE2": 1.5,
    "PHASE2":        2.0,
    "PHASE2/PHASE3": 2.5,
    "PHASE3":        3.0,
    "PHASE4":        4.0,
}

CENSORED_STATUSES = {
    "RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION",
}

ARRAY_COLUMNS = [
    "indications",
    "interventions_drugs",
    "drugs_datalake",
    "main_technologies",
    "specific_technologies",
    "target_names",
    "target_abbreviations",
]

PHASE_ORDER = [
    "Early Phase I", "Phase I", "Phase I/II",
    "Phase II", "Phase II/III", "Phase III", "Phase IV",
]

PHASE_NUMERIC_TO_LABEL = {
    0.5: "Early Phase I",
    1.0: "Phase I",
    1.5: "Phase I/II",
    2.0: "Phase II",
    2.5: "Phase II/III",
    3.0: "Phase III",
    4.0: "Phase IV",
}

def prepare_analytical_dataframe(file_path: str, sheet_name: str) -> pd.DataFrame:

    print("=" * 78)
    print("  PREPARING ANALYTICAL DATAFRAME  (Part 1 pipeline replay)")
    print("=" * 78)

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"  Loaded {df.shape[0]} rows × {df.shape[1]} columns\n")

    # Parse stringified arrays
    for col in ARRAY_COLUMNS:
        df[col] = df[col].apply(lambda x: ast.literal_eval(str(x)))

    # Standardise phases
    df["standardized_phase"] = df["phase"].map(PHASE_STANDARD_MAP).fillna(
        df["phase"].apply(
            lambda x: "Unspecified Phase" if pd.isna(x) else PHASE_STANDARD_MAP.get(x, x)
        )
    )
    df["phase_numeric"] = df["phase"].map(PHASE_NUMERIC_MAP)

    # Impute enrollment type
    mask = (
        df["enrollment_type"].isna()
        & df["enrollment"].notna()
        & (df["enrollment"] > 0)
        & df["recruitment_status"].isin(["COMPLETED", "TERMINATED"])
    )
    df.loc[mask, "enrollment_type"] = "ACTUAL"

    # Engineer duration (with withdrawn trial date cleanup)
    withdrawn_zero = (
        (df["recruitment_status"] == "WITHDRAWN") &
        (df["enrollment"].fillna(0) == 0)
    )
    df.loc[withdrawn_zero, "completion_date"]         = pd.NaT
    df.loc[withdrawn_zero, "primary_completion_date"] = pd.NaT

    effective_completion = df["completion_date"].fillna(df["primary_completion_date"])
    duration = (effective_completion - df["start_date"]).dt.days
    df["trial_duration_days"] = duration.where(duration >= 0, other=np.nan)
    df["start_year"] = df["start_date"].dt.year

    df["is_right_censored"]  = df["recruitment_status"].isin(CENSORED_STATUSES)
    df["is_outcome_unknown"] = df["recruitment_status"] == "UNKNOWN"

    print(f"  is_right_censored=True  : {df['is_right_censored'].sum()} trials "
          f"(Active/Ongoing — excluded from success rate denominators)")
    print(f"  is_outcome_unknown=True : {df['is_outcome_unknown'].sum()} trials "
          f"(UNKNOWN status — excluded from success rate denominators)")
    print("  ✅ Analytical DataFrame prepared.\n")
    return df

def apply_binary_success_rules(df: pd.DataFrame) -> pd.Series:
    conditions = [
        df["recruitment_status"] == "COMPLETED",
        df["recruitment_status"] == "TERMINATED",
    ]
    choices = [1.0, 0.0]

    result = pd.Series(
        np.select(conditions, choices, default=np.nan),
        index=df.index,
        name="binary_success",
    )

    censored_count = df["is_right_censored"].sum()
    unknown_count  = df["is_outcome_unknown"].sum()
    other_excluded = result.isna().sum() - censored_count - unknown_count
    print(f"  Excluded from binary success denominator:")
    print(f"     {censored_count:>4d}  right-censored (Active/Ongoing)")
    print(f"     {unknown_count:>4d}  outcome-unknown (UNKNOWN status)")
    if other_excluded > 0:
        print(f"     {other_excluded:>4d}  other (SUSPENDED, WITHDRAWN, etc.)")

    return result


def apply_tiered_success_rules(df: pd.DataFrame) -> tuple[pd.Series, float]:
    """
    Assigns trials to a success tier (0–3) based on status, enrollment,
    and enrollment type:

        Tier 3 (High Success)     : COMPLETED + ACTUAL enrollment ≥ median
        Tier 2 (Operational Pass) : COMPLETED but small or estimated enrollment
        Tier 1 (Operational Fail) : TERMINATED or SUSPENDED with enrollment > 0
                                    Also: WITHDRAWN with enrollment > 0
                                    (e.g. NCT00149019, enrollment=12)
        Tier 0 (Pre-start Fail)   : WITHDRAWN with enrollment == 0
        NaN   (Excluded)          : RECRUITING, ACTIVE_NOT_RECRUITING,
                                    NOT_YET_RECRUITING, ENROLLING_BY_INVITATION,
                                    UNKNOWN
    """
    completed_actual = df[
        (df["recruitment_status"] == "COMPLETED")
        & (df["enrollment_type"] == "ACTUAL")
        & (df["enrollment"].notna())
    ]
    median_enrollment = completed_actual["enrollment"].median()

    conditions = [
        # Tier 3: Completed, actual enrollment ≥ median
        (df["recruitment_status"] == "COMPLETED")
        & (df["enrollment_type"] == "ACTUAL")
        & (df["enrollment"] >= median_enrollment),

        # Tier 2: Completed but below median or estimated enrollment
        (df["recruitment_status"] == "COMPLETED"),

        # Tier 1: Terminated/Suspended with enrollment > 0
        # Also: WITHDRAWN with enrollment > 0 (e.g. NCT00149019)
        (
            df["recruitment_status"].isin(["TERMINATED", "SUSPENDED"])
            & (df["enrollment"].fillna(0) > 0)
        )
        | (
            (df["recruitment_status"] == "WITHDRAWN")
            & (df["enrollment"].fillna(0) > 0)
        ),

        # Tier 0: Withdrawn with zero enrollment
        (df["recruitment_status"] == "WITHDRAWN")
        & (df["enrollment"].fillna(0) == 0),
    ]
    choices = [3.0, 2.0, 1.0, 0.0]

    tiered = pd.Series(
        np.select(conditions, choices, default=np.nan),
        index=df.index,
        name="tiered_success",
    )
    return tiered, median_enrollment


def operationalise_success(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function for Phase A: applies both binary and tiered success
    rules, prints a detailed report, and returns the enriched DataFrame.
    """
    print("=" * 78)
    print("  PHASE A: OPERATIONALISING 'SUCCESS'")
    print("=" * 78)

    df = df.copy()

    # ── Binary Success ───────────────────────────────────────────────────
    df["binary_success"] = apply_binary_success_rules(df)

    print("\n📊 BINARY SUCCESS PROXY DEFINITION")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  Success  (1)   : COMPLETED                            │")
    print("  │  Failure  (0)   : TERMINATED                           │")
    print("  │  Excluded (NaN) : All other statuses                   │")
    print("  └─────────────────────────────────────────────────────────┘")
    print("\n  Binary outcome distribution:")
    binary_counts = df["binary_success"].value_counts(dropna=False)
    for val, cnt in binary_counts.items():
        label = {1.0: "Success (1)", 0.0: "Failure (0)"}.get(val, "Excluded (NaN)")
        print(f"     {label:<20s}  {cnt:>4d}  ({cnt/len(df)*100:5.1f}%)")

    # ── Tiered Success ───────────────────────────────────────────────────
    df["tiered_success"], median_enroll = apply_tiered_success_rules(df)

    print(f"\n📊 TIERED SUCCESS PROXY DEFINITION  (median enrollment = {median_enroll})")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  Tier 3 : COMPLETED + ACTUAL enrollment ≥ median       │")
    print("  │  Tier 2 : COMPLETED (small or estimated enrollment)    │")
    print("  │  Tier 1 : TERMINATED/SUSPENDED/WITHDRAWN w/ enroll > 0 │")
    print("  │  Tier 0 : WITHDRAWN with enrollment == 0               │")
    print("  │  NaN    : Active / Recruiting / Unknown                │")
    print("  └─────────────────────────────────────────────────────────┘")
    print("\n  Tiered outcome distribution:")
    tier_counts = df["tiered_success"].value_counts(dropna=False).sort_index()
    tier_labels = {
        0.0: "Tier 0 (Pre-start Fail)",
        1.0: "Tier 1 (Operational Fail)",
        2.0: "Tier 2 (Operational Pass)",
        3.0: "Tier 3 (High Success)",
    }
    for val, cnt in tier_counts.items():
        label = tier_labels.get(val, "Excluded (NaN)")
        print(f"     {label:<30s}  {cnt:>4d}  ({cnt/len(df)*100:5.1f}%)")

    # ── Ambiguous Status Handling ────────────────────────────────────────
    print("\n📊 AMBIGUOUS STATUS HANDLING")
    ambiguous_statuses = [
        "SUSPENDED", "UNKNOWN", "NOT_YET_RECRUITING",
        "RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION",
    ]
    for status in ambiguous_statuses:
        count = (df["recruitment_status"] == status).sum()
        binary_val = "Excluded (NaN)"
        tiered_val = "Tier 1 (if enrolled > 0)" if status == "SUSPENDED" else "Excluded (NaN)"
        print(f"     {status:<30s}  n={count:>3d}  binary={binary_val:<15s}  tiered={tiered_val}")

    # ── Proxy Positioning Commentary ─────────────────────────────────────
    print("\n📝 PROXY POSITIONING ON THE SUCCESS SPECTRUM")
    print("  ┌────────────────────────────────────────────────────────────────┐")
    print("  │  Trial          Operational          Therapeutic    Drug      │")
    print("  │  Completion  →  Success           →  Efficacy    → Approval  │")
    print("  │                     ▲                                        │")
    print("  │              OUR PROXY SITS HERE                             │")
    print("  │                                                              │")
    print("  │  We measure operational completion, NOT therapeutic efficacy. │")
    print("  │  A completed trial may still fail its primary endpoint.      │")
    print("  │  A terminated trial may (rarely) be stopped for efficacy.    │")
    print("  └────────────────────────────────────────────────────────────────┘")

    # ── Validation Checks ────────────────────────────────────────────────
    print("\n  🔍 VALIDATION CHECKS (Phase A):")

    # V1: Withdrawn (0 enroll) correctly excluded from binary
    withdrawn_zero_enroll = df[
        (df["recruitment_status"] == "WITHDRAWN") &
        (df["enrollment"].fillna(0) == 0)
    ]
    v1_pass = withdrawn_zero_enroll["binary_success"].isna().all()
    print(f"     V1 - Withdrawn (0 enroll) excluded from binary : "
          f"{'✅ PASS' if v1_pass else '❌ FAIL'}")

    # V2: Binary defined ≤ total rows
    binary_defined = df["binary_success"].notna().sum()
    v2_pass = binary_defined <= len(df)
    print(f"     V2 - Binary defined ({binary_defined}) ≤ total ({len(df)}) : "
          f"{'✅ PASS' if v2_pass else '❌ FAIL'}")

    # V3: NCT00149019 edge case (withdrawn with enrollment > 0 → Tier 1)
    nct_edge = df[df["nct_id"] == "NCT00149019"]
    if len(nct_edge) > 0:
        tier_val = nct_edge["tiered_success"].values[0]
        v3_pass = tier_val == 1.0
        print(f"     V3 - NCT00149019 (withdrawn, enroll=12) → Tier 1 : "
              f"{'✅ PASS' if v3_pass else '❌ FAIL'} (got Tier {tier_val})")
    else:
        print("     V3 - NCT00149019 not found in dataset (skipped)")

    # V4: is_right_censored count
    v4_pass = df["is_right_censored"].sum() == 259
    print(f"     V4 - is_right_censored count == 259 : "
          f"{'✅ PASS' if v4_pass else '❌ FAIL'} "
          f"(actual={df['is_right_censored'].sum()})")

    # V5: is_outcome_unknown count
    v5_pass = df["is_outcome_unknown"].sum() == 121
    print(f"     V5 - is_outcome_unknown count == 121 : "
          f"{'✅ PASS' if v5_pass else '❌ FAIL'} "
          f"(actual={df['is_outcome_unknown'].sum()})")

    # V6: No censored/unknown trial has a non-NaN binary_success
    censored_or_unknown = df["is_right_censored"] | df["is_outcome_unknown"]
    v6_pass = df.loc[censored_or_unknown, "binary_success"].isna().all()
    print(f"     V6 - Censored/unknown trials have NaN binary_success : "
          f"{'✅ PASS' if v6_pass else '❌ FAIL'}")

    return df

def compute_wilson_ci(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Calculates the Wilson Score confidence interval for a binomial proportion.

    Parameters
    ----------
    successes : int   – number of successes
    total     : int   – total number of trials in the cohort
    confidence: float – confidence level (default 0.95 for 95% CI)

    Returns
    -------
    (ci_lower, ci_upper) : tuple of floats
    """
    if total == 0:
        return (0.0, 0.0)

    z         = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat     = successes / total
    n         = total
    denominator = 1 + z**2 / n
    centre    = (p_hat + z**2 / (2 * n)) / denominator
    margin    = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denominator

    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


def calculate_cohort_success_rates(
    df: pd.DataFrame,
    dimension_col: str,
    success_col: str = "binary_success",
    min_cohort_size: int = 3,
) -> pd.DataFrame:
    evaluable = df[df[success_col].notna()].copy()

    grouped = (
        evaluable.groupby(dimension_col)
        .agg(
            total_trials=(success_col, "count"),
            success_count=(success_col, "sum"),
        )
        .reset_index()
    )

    grouped["failure_count"] = grouped["total_trials"] - grouped["success_count"]
    grouped["success_rate"]  = (grouped["success_count"] / grouped["total_trials"]).round(4)

    ci_bounds = grouped.apply(
        lambda row: compute_wilson_ci(int(row["success_count"]), int(row["total_trials"])),
        axis=1,
    )
    grouped["ci_lower"] = ci_bounds.apply(lambda x: x[0])
    grouped["ci_upper"] = ci_bounds.apply(lambda x: x[1])
    grouped["low_sample_flag"] = grouped["total_trials"] < 5

    grouped = grouped[grouped["total_trials"] >= min_cohort_size]
    grouped = grouped.sort_values("success_rate", ascending=False).reset_index(drop=True)

    return grouped


def calculate_multidim_success_rates(
    df: pd.DataFrame,
    dim_cols: list[str],
    success_col: str = "binary_success",
    min_cohort_size: int = 3,
) -> pd.DataFrame:
    """
    Multi-dimensional cohort analysis (e.g. Indication × Phase).
    Same logic as single-dimension but groups on multiple columns.
    """
    evaluable = df[df[success_col].notna()].copy()

    grouped = (
        evaluable.groupby(dim_cols)
        .agg(
            total_trials=(success_col, "count"),
            success_count=(success_col, "sum"),
        )
        .reset_index()
    )

    grouped["failure_count"] = grouped["total_trials"] - grouped["success_count"]
    grouped["success_rate"]  = (grouped["success_count"] / grouped["total_trials"]).round(4)

    ci_bounds = grouped.apply(
        lambda row: compute_wilson_ci(int(row["success_count"]), int(row["total_trials"])),
        axis=1,
    )
    grouped["ci_lower"] = ci_bounds.apply(lambda x: x[0])
    grouped["ci_upper"] = ci_bounds.apply(lambda x: x[1])
    grouped["low_sample_flag"] = grouped["total_trials"] < 5

    grouped = grouped[grouped["total_trials"] >= min_cohort_size]
    grouped = grouped.sort_values("success_rate", ascending=False).reset_index(drop=True)

    return grouped


# ── Explode helpers ───────────────────────────────────────────────────────────

def explode_main_technology(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()

    def flatten_tech(tech_list):
        flat = []
        for sub in tech_list:
            if isinstance(sub, list):
                flat.extend(sub)
            else:
                flat.append(sub)
        return flat if flat else ["Unspecified"]

    df["main_technology_flat"] = df["main_technologies"].apply(flatten_tech)
    df = df.explode("main_technology_flat").rename(
        columns={"main_technology_flat": "main_technology"}
    )
    df = df.drop_duplicates(subset=["ID-datalake", "main_technology"])
    return df


def explode_target_abbreviation(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    def flatten_targets(tgt_list):
        flat = []
        for sub in tgt_list:
            if isinstance(sub, list):
                flat.extend(sub)
            else:
                flat.append(sub)
        return flat if flat else ["Unspecified Target"]

    df["target_class"] = df["target_abbreviations"].apply(flatten_targets)
    df = df.explode("target_class")
    df = df.drop_duplicates(subset=["ID-datalake", "target_class"])
    return df


def explode_indications(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()
    df["indication"] = df["indications"].apply(
        lambda x: list(set(x)) if x else ["Unspecified"]
    )
    df = df.explode("indication")
    df = df.drop_duplicates(subset=["ID-datalake", "indication"])
    return df


# ── Visualisation functions ───────────────────────────────────────────────────

def plot_success_heatmap(
    cohort_df: pd.DataFrame,
    row_col: str,
    col_col: str,
    output_path: str,
    top_n: int = 15,
):
   
    ind_totals = cohort_df.groupby(row_col)["total_trials"].sum()
    top_inds   = ind_totals.nlargest(top_n).index
    filtered   = cohort_df[cohort_df[row_col].isin(top_inds)]

    pivot = filtered.pivot_table(
        index=row_col, columns=col_col, values="success_rate", aggfunc="first"
    )
    annot_pivot = filtered.pivot_table(
        index=row_col, columns=col_col, values="total_trials", aggfunc="sum"
    )

    annot = pd.DataFrame("", index=pivot.index, columns=pivot.columns, dtype=object)
    for r in annot.index:
        for c in annot.columns:
            rate = pivot.loc[r, c] if pd.notna(pivot.loc[r, c]) else np.nan
            n    = annot_pivot.loc[r, c] if pd.notna(annot_pivot.loc[r, c]) else 0
            if pd.notna(rate):
                annot.loc[r, c] = f"{rate:.0%}\n(n={int(n)})"

    ordered_cols = [p for p in PHASE_ORDER + ["Unspecified Phase"] if p in pivot.columns]
    pivot = pivot[ordered_cols]
    annot = annot[ordered_cols]

    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.55)))
    sns.heatmap(
        pivot.astype(float),
        annot=annot.values,
        fmt="",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        ax=ax,
        cbar_kws={"label": "Success Rate", "shrink": 0.7},
    )
    ax.set_title(
        "Oncology Trial Success Rate: Indication × Phase\n"
        "(Operational Completion Proxy)",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_ylabel("Indication", fontsize=11)
    ax.set_xlabel("Clinical Phase", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Heatmap saved: {output_path}")


def plot_forest_ci(
    cohort_df: pd.DataFrame,
    dimension_col: str,
    output_path: str,
    top_n: int = 20,
):
    """
    Forest plot showing success rates with 95% Wilson CI error bars
    for each modality/technology type.
    """
    plot_df = cohort_df.nlargest(top_n, "total_trials").copy()
    plot_df = plot_df.sort_values("success_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.4)))

    xerr_lower = plot_df["success_rate"] - plot_df["ci_lower"]
    xerr_upper = plot_df["ci_upper"] - plot_df["success_rate"]
    y_pos = range(len(plot_df))

    ax.errorbar(
        plot_df["success_rate"], y_pos,
        xerr=[xerr_lower.values, xerr_upper.values],
        fmt="o", color="#2563EB", ecolor="#94A3B8",
        elinewidth=1.5, capsize=4, markersize=7,
        markerfacecolor="#2563EB", markeredgecolor="white", markeredgewidth=1,
    )

    for i, (_, row) in enumerate(plot_df.iterrows()):
        flag = " ⚠️" if row["low_sample_flag"] else ""
        ax.text(
            row["ci_upper"] + 0.02, i,
            f"n={int(row['total_trials'])}{flag}",
            va="center", fontsize=8, color="#64748B",
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df[dimension_col].values, fontsize=9)
    ax.set_xlabel("Success Rate (95% Wilson CI)", fontsize=11)
    ax.set_title(
        f"Success Rate by {dimension_col.replace('_', ' ').title()}\n"
        f"(with 95% Wilson Score Confidence Intervals)",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.axvline(x=0.5, color="#E2E8F0", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlim(-0.05, 1.15)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Forest plot saved: {output_path}")


def plot_trial_phase_funnel(df: pd.DataFrame, output_path: str):
   
    evaluable = df[df["binary_success"].notna()].copy()

    # Note unspecified phase exclusion
    unspecified_n = (evaluable["standardized_phase"] == "Unspecified Phase").sum()

    phase_data = []
    for phase in PHASE_ORDER:
        subset = evaluable[evaluable["standardized_phase"] == phase]
        total  = len(subset)
        successes = int(subset["binary_success"].sum())
        if total > 0:
            rate = successes / total
            ci   = compute_wilson_ci(successes, total)
        else:
            rate = 0
            ci   = (0, 0)
        phase_data.append({
            "phase": phase, "total_trials": total, "successes": successes,
            "success_rate": rate, "ci_lower": ci[0], "ci_upper": ci[1],
        })

    funnel = pd.DataFrame(phase_data)
    funnel = funnel[funnel["total_trials"] > 0]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = range(len(funnel))

    ax1.bar(x, funnel["total_trials"], 0.6, color="#3B82F6", alpha=0.7,
            label="Total Evaluable Trials", edgecolor="white", linewidth=1)
    ax1.bar(x, funnel["successes"], 0.6, color="#10B981", alpha=0.85,
            label="Completed (Success)", edgecolor="white", linewidth=1)

    ax1.set_xlabel("Clinical Phase", fontsize=11)
    ax1.set_ylabel("Number of Trials", fontsize=11, color="#3B82F6")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(funnel["phase"].values, rotation=15, fontsize=10)

    for i, (_, row) in enumerate(funnel.iterrows()):
        ax1.text(i, row["total_trials"] + 2, f"n={int(row['total_trials'])}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(list(x), funnel["success_rate"].values, "o-", color="#F59E0B",
             linewidth=2.5, markersize=8, markerfacecolor="#F59E0B",
             markeredgecolor="white", markeredgewidth=1.5, label="Success Rate")
    ax2.fill_between(list(x), funnel["ci_lower"].values, funnel["ci_upper"].values,
                     alpha=0.15, color="#F59E0B")
    ax2.set_ylabel("Success Rate", fontsize=11, color="#F59E0B")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=9, framealpha=0.9)

    note = f"Note: 'Unspecified Phase' trials (n={unspecified_n}) excluded — phase position unknown."
    ax1.set_title(
        "Clinical Phase Attrition Funnel (Trial-Level)\n"
        "(Operational Completion Proxy with 95% Wilson CI)",
        fontsize=13, fontweight="bold", pad=15,
    )
    fig.text(0.5, -0.02, note, ha="center", fontsize=8, color="#64748B", style="italic")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Trial-level funnel saved: {output_path}")
    if unspecified_n > 0:
        print(f"     Note: {unspecified_n} 'Unspecified Phase' trials excluded from funnel chart.")


def plot_drug_phase_funnel(drug_phase_summary: pd.DataFrame, output_path: str):
  
    # Map numeric phase to label if needed
    if "max_phase_label" not in drug_phase_summary.columns:
        drug_phase_summary = drug_phase_summary.copy()
        drug_phase_summary["max_phase_label"] = drug_phase_summary["max_phase_numeric"].map(
            PHASE_NUMERIC_TO_LABEL
        )

    phase_counts = (
        drug_phase_summary.groupby("max_phase_label")
        .size()
        .reset_index(name="n_drugs")
    )

    # Order by PHASE_ORDER
    phase_counts["_order"] = phase_counts["max_phase_label"].map(
        {p: i for i, p in enumerate(PHASE_ORDER)}
    )
    phase_counts = phase_counts.dropna(subset=["_order"]).sort_values("_order")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(phase_counts))

    bars = ax.bar(
        x, phase_counts["n_drugs"], 0.6,
        color="#8B5CF6", alpha=0.8,
        label="Unique Drugs", edgecolor="white", linewidth=1,
    )

    ax.set_xlabel("Highest Phase Reached", fontsize=11)
    ax.set_ylabel("Number of Unique Drugs", fontsize=11, color="#8B5CF6")
    ax.set_xticks(list(x))
    ax.set_xticklabels(phase_counts["max_phase_label"].values, rotation=15, fontsize=10)

    for i, (_, row) in enumerate(phase_counts.iterrows()):
        ax.text(i, row["n_drugs"] + 0.5, f"n={int(row['n_drugs'])}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title(
        "Drug Program Phase Distribution (Drug-Level)\n"
        "(Highest Phase Reached per Drug — 1,000-trial extract)",
        fontsize=13, fontweight="bold", pad=15,
    )
    note = ("Drug-level funnel shows phase distribution, not attrition rates.\n"
            "Success rates require drug-level outcome data not available in this extract.")
    fig.text(0.5, -0.04, note, ha="center", fontsize=8, color="#64748B", style="italic")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Drug-level funnel saved: {output_path}")


def run_cohort_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Master function for Phase B: computes stratified success rates across
    multiple dimensions with confidence intervals and generates plots.
    """
    print("\n" + "=" * 78)
    print("  PHASE B: STRATIFIED SUCCESS RATES & COHORT ANALYSIS")
    print("=" * 78)

    results: dict[str, pd.DataFrame] = {}

    # ── Dimension 1: Indication × Phase ──────────────────────────────────
    print("\n─── DIMENSION 1: Indication × Phase ──────────────────────────")
    df_ind = explode_indications(df)
    cohort_ind_phase = calculate_multidim_success_rates(
        df_ind, ["indication", "standardized_phase"], min_cohort_size=3
    )
    results["indication_x_phase"] = cohort_ind_phase
    print(f"  Cohorts with n ≥ 3: {len(cohort_ind_phase)}")
    print("\n  Top 10 cohorts by success rate:")
    for _, row in cohort_ind_phase.head(10).iterrows():
        flag = "⚠️" if row["low_sample_flag"] else "  "
        print(f"  {flag} {row['indication']:<30s} {row['standardized_phase']:<15s}  "
              f"rate={row['success_rate']:.1%}  "
              f"CI=[{row['ci_lower']:.1%}, {row['ci_upper']:.1%}]  "
              f"n={int(row['total_trials'])}")

    plot_success_heatmap(
        cohort_ind_phase, "indication", "standardized_phase",
        os.path.join(OUTPUT_DIR, "heatmap_indication_phase.png"), top_n=15,
    )

    # ── Dimension 2: Main Technology Type ────────────────────────────────
    print("\n─── DIMENSION 2: Main Technology Type ────────────────────────")
    df_tech = explode_main_technology(df)
    cohort_tech = calculate_cohort_success_rates(
        df_tech, "main_technology", min_cohort_size=3
    )
    results["technology_type"] = cohort_tech
    print(f"  Technology cohorts with n ≥ 3: {len(cohort_tech)}")
    print("\n  Success rates by technology:")
    for _, row in cohort_tech.iterrows():
        flag = "⚠️" if row["low_sample_flag"] else "  "
        print(f"  {flag} {row['main_technology']:<30s}  "
              f"rate={row['success_rate']:.1%}  "
              f"CI=[{row['ci_lower']:.1%}, {row['ci_upper']:.1%}]  "
              f"n={int(row['total_trials'])}")

    plot_forest_ci(
        cohort_tech, "main_technology",
        os.path.join(OUTPUT_DIR, "forest_technology.png"), top_n=20,
    )

    # ── Dimension 3: Target Class ────────────────────────────────────────
    print("\n─── DIMENSION 3: Target Class (Abbreviation) ─────────────────")
    df_tgt = explode_target_abbreviation(df)
    cohort_target = calculate_cohort_success_rates(
        df_tgt, "target_class", min_cohort_size=3
    )
    results["target_class"] = cohort_target
    print(f"  Target cohorts with n ≥ 3: {len(cohort_target)}")
    print("\n  Top 15 target classes by success rate:")
    for _, row in cohort_target.head(15).iterrows():
        flag = "⚠️" if row["low_sample_flag"] else "  "
        print(f"  {flag} {row['target_class']:<25s}  "
              f"rate={row['success_rate']:.1%}  "
              f"CI=[{row['ci_lower']:.1%}, {row['ci_upper']:.1%}]  "
              f"n={int(row['total_trials'])}")

    plot_forest_ci(
        cohort_target, "target_class",
        os.path.join(OUTPUT_DIR, "forest_target_class.png"), top_n=20,
    )

    # ── Dimension 4: Phase-level success (trial-level funnel) ─────────────
    print("\n─── DIMENSION 4: Phase Attrition Funnel (Trial-Level) ────────")
    cohort_phase = calculate_cohort_success_rates(
        df, "standardized_phase", min_cohort_size=1
    )
    results["phase"] = cohort_phase
    print("  Success rates by phase (binary proxy):")
    for phase in PHASE_ORDER:
        row = cohort_phase[cohort_phase["standardized_phase"] == phase]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"     {phase:<18s}  rate={r['success_rate']:.1%}  "
                  f"CI=[{r['ci_lower']:.1%}, {r['ci_upper']:.1%}]  "
                  f"n={int(r['total_trials'])}")

    plot_trial_phase_funnel(df, os.path.join(OUTPUT_DIR, "funnel_trial_phase.png"))

    drug_phase_path = os.path.join(OUTPUT_DIR, "drug_phase_summary.csv")
    if os.path.exists(drug_phase_path):
        drug_phase_summary = pd.read_csv(drug_phase_path)
        print(f"\n  Loaded drug_phase_summary: {len(drug_phase_summary)} unique drugs")
        plot_drug_phase_funnel(
            drug_phase_summary,
            os.path.join(OUTPUT_DIR, "funnel_drug_phase.png"),
        )
        results["drug_phase_summary"] = drug_phase_summary
    else:
        print("\n  ⚠️  drug_phase_summary.csv not found — run Part 1 first to generate it.")
        print("       Drug-level funnel skipped.")

    # ── Dimension 5: Tiered success by phase ─────────────────────────────
    print("\n─── DIMENSION 5: Tiered Success by Phase ─────────────────────")
    print("  NOTE: Tiered success (0–3 scale) is averaged per phase.")
    print("  Tier 3 = fully-enrolled completion; Tier 0 = pre-start withdrawal.")
    print("  'success_rate' here = mean tier score, NOT a binary success rate.")
    cohort_tiered_phase = calculate_cohort_success_rates(
        df[df["tiered_success"].notna()],
        "standardized_phase",
        success_col="tiered_success",
        min_cohort_size=1,
    )
    results["tiered_phase"] = cohort_tiered_phase
    print("\n  Mean tier score by phase (compare with binary rates above):")
    for phase in PHASE_ORDER:
        row_b = cohort_phase[cohort_phase["standardized_phase"] == phase]
        row_t = cohort_tiered_phase[cohort_tiered_phase["standardized_phase"] == phase]
        if len(row_b) > 0 and len(row_t) > 0:
            rb = row_b.iloc[0]
            rt = row_t.iloc[0]
            print(f"     {phase:<18s}  binary={rb['success_rate']:.1%}  "
                  f"mean_tier={rt['success_rate']:.2f}  "
                  f"n_tiered={int(rt['total_trials'])}")

    # ── Validation Checks (Phase B) ──────────────────────────────────────
    print("\n  🔍 VALIDATION CHECKS (Phase B):")

    # V1: success_rate = success_count / total_trials
    for name, cohort in results.items():
        if "success_count" not in cohort.columns:
            continue
        recalc = cohort["success_count"] / cohort["total_trials"]
        match  = np.allclose(cohort["success_rate"], recalc, atol=1e-4, equal_nan=True)
        print(f"     V1 ({name}) - rate = count/total : {'✅ PASS' if match else '❌ FAIL'}")

    # V2: CI width decreases with larger n
    for name, cohort in results.items():
        if "ci_upper" not in cohort.columns or len(cohort) < 5:
            continue
        cohort_check = cohort.copy()
        cohort_check["ci_width"] = cohort_check["ci_upper"] - cohort_check["ci_lower"]
        corr   = cohort_check[["total_trials", "ci_width"]].corr().iloc[0, 1]
        v2_pass = corr < 0
        print(f"     V2 ({name}) - CI width ↓ as n ↑ : "
              f"{'✅ PASS' if v2_pass else '⚠️ WEAK'} (corr={corr:.3f})")

    df_tech_check = explode_main_technology(df)
    max_dups = df_tech_check.groupby("main_technology")["ID-datalake"].apply(
        lambda x: x.duplicated().sum()
    ).max()
    v7_pass = max_dups == 0
    print(f"     V7 - No duplicate (trial, technology) rows after dedup : "
          f"{'✅ PASS' if v7_pass else '❌ FAIL'} (max_dups={max_dups})")

    return results

def main():
    """Run the full Part 2 pipeline: Success Logic → Cohort Analysis."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Prepare data (Part 1 replay) ─────────────────────────────────────
    df = prepare_analytical_dataframe(INPUT_FILE, SHEET_NAME)

    # ── PHASE A: Operationalise success ──────────────────────────────────
    df = operationalise_success(df)

    # Export enriched trial data
    export_cols = [
        "ID-datalake", "nct_id", "brief_title", "phase",
        "standardized_phase", "phase_numeric", "recruitment_status",
        "start_date", "completion_date", "trial_duration_days",
        "start_year", "enrollment", "enrollment_type",
        "is_right_censored", "is_outcome_unknown",
        "binary_success", "tiered_success",
    ]
    df[export_cols].to_csv(
        os.path.join(OUTPUT_DIR, "trials_with_success_flags.csv"), index=False
    )
    print(f"\n  💾 Saved: {os.path.join(OUTPUT_DIR, 'trials_with_success_flags.csv')}")

    # ── PHASE B: Cohort analysis ─────────────────────────────────────────
    cohort_results = run_cohort_analysis(df)

    # Export cohort tables
    for name, cohort_df in cohort_results.items():
        out_path = os.path.join(OUTPUT_DIR, f"cohort_{name}.csv")
        cohort_df.to_csv(out_path, index=False)
        print(f"  💾 Saved: {out_path}")

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  ✅  PART 2 COMPLETE")
    print("=" * 78)
    print(f"\n  Total trials in dataset          : {len(df)}")
    evaluable  = df["binary_success"].notna().sum()
    successes  = int(df["binary_success"].sum())
    print(f"  Evaluable trials (binary proxy)  : {evaluable}")
    print(f"  Overall success rate             : {successes}/{evaluable} = "
          f"{successes/evaluable:.1%}")
    ci = compute_wilson_ci(successes, evaluable)
    print(f"  Overall 95% Wilson CI            : [{ci[0]:.1%}, {ci[1]:.1%}]")

    # ── Documented Assumptions & Limitations ─────────────────────────────
    print("\n📝 DOCUMENTED ASSUMPTIONS & LIMITATIONS:")
    print("  1. 'Success' is an OPERATIONAL COMPLETION proxy — not a measure of")
    print("     therapeutic efficacy. A completed trial may fail its primary")
    print("     endpoint; a terminated trial may (rarely) show efficacy.")
    print("  2. WITHDRAWN trials with 0 enrollment → NaN binary, Tier 0 tiered.")
    print("  3. SUSPENDED → excluded from binary (NaN); SUSPENDED with enrollment")
    print("     > 0 → Tier 1 in tiered model. UNKNOWN → excluded from both.")
    print("  4. Registry datasets suffer from PUBLICATION BIAS — successful trials")
    print("     are more likely to be fully reported, potentially inflating rates.")
    print("  5. Wilson Score CIs used (not Wald) for small-sample robustness;")
    print("     cohorts with n < 5 are flagged with low_sample_flag=True.")
    print("  6. is_right_censored (259 trials) and is_outcome_unknown (121 trials)")
    print("     are excluded from all success rate denominators. Rates for recent")
    print("     cohorts are likely underestimated due to right-censoring.")
    print("  7. After exploding multi-valued fields (technology, target, indication),")
    print("     rows are deduplicated per (trial_id, dimension_value) so each trial")
    print("     contributes exactly one success/failure observation per unique value.")
    print("     A trial with 3 drugs all classified as 'Antibody' counts once.")
    print("  8. min_cohort_size=3 for all stratified tables. Phase funnel uses")
    print("     min_cohort_size=1 to preserve funnel shape. Cohorts with n < 5")
    print("     are flagged.")
    print("  9. Drug-level funnel (funnel_drug_phase.png) shows phase distribution")
    print("     only — not attrition rates. Drug-level success/failure outcomes are")
    print("     not available in this extract. Requires drug_phase_summary.csv from")
    print("     Part 1 output.")
    print(" 10. median_enrollment for Tier 3 threshold is computed on COMPLETED +")
    print("     ACTUAL trials in this extract only and may not generalise to the")
    print("     full registry.")

    print("\n  📊 Visualisations saved to ./output/:")
    print("     • heatmap_indication_phase.png   (Indication × Phase)")
    print("     • forest_technology.png           (Technology Type)")
    print("     • forest_target_class.png         (Target Class)")
    print("     • funnel_trial_phase.png          (Trial-Level Phase Funnel)")
    print("     • funnel_drug_phase.png           (Drug-Level Phase Distribution)")
    print("       └─ Requires drug_phase_summary.csv from Part 1 run")

    return df, cohort_results


if __name__ == "__main__":
    enriched_df, all_cohorts = main()
