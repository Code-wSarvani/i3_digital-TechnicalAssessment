"""
==============================================================================
Part 1: Data Ingestion & Schema Design
==============================================================================
  Phase A  –  Parse, profile, and generate a structured Data Quality Report
  Phase B  –  Design and implement a normalised analytical schema

Author : Oncology Trial Analytics Pipeline
Input  : SampleDateExtract.xlsx  (sheet: 1000_inteventional_trials)
Output : Console report + normalised CSV tables in  ./output/
==============================================================================
"""

import os
import ast
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_FILE = "C:/Users/Sarvani/Desktop/i3_digital/SampleDateExtract.xlsx"
SHEET_NAME = "1000_inteventional_trials"
OUTPUT_DIR = "output"

# Columns that contain stringified Python lists
ARRAY_COLUMNS = [
    "indications",
    "interventions_drugs",
    "drugs_datalake",
    "main_technologies",
    "specific_technologies",
    "target_names",
    "target_abbreviations",
]

# Controlled vocabulary: phase standardisation
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

# Controlled vocabulary: recruitment status grouping
STATUS_GROUP_MAP = {
    "COMPLETED":               "Completed",
    "RECRUITING":              "Active/Ongoing",
    "ACTIVE_NOT_RECRUITING":   "Active/Ongoing",
    "NOT_YET_RECRUITING":      "Active/Ongoing",
    "ENROLLING_BY_INVITATION": "Active/Ongoing",
    "TERMINATED":              "Halted/Terminated",
    "SUSPENDED":               "Halted/Terminated",
    "WITHDRAWN":               "Withdrawn",
    "UNKNOWN":                 "Ambiguous",
}

# FIX 1 — Right-censoring: statuses where the trial outcome is not yet known
# because the trial is still running. These 259 trials are excluded from
# success rate denominators in Part 2. Rates computed without them will
# underestimate true success rates for recent cohorts.
CENSORED_STATUSES = {
    "RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION",
}

# FIX 7 — Immunotherapy boundary (module-level constant, single source of truth)
# Includes: immune-activating cell and vaccine modalities only.
# Excluded: plain "Antibody" — too broad; captures anti-EGFR, anti-HER2, ADCs
#           which are not conventionally immunotherapy.
# Excluded: "Antibody Drug Conjugate (ADC)" — cytotoxic payload, not immune activation.
# Note: "CAR-T" does NOT match any value in main_technologies; full string required.
IMMUNOTHERAPY_TECHS = {
    "Chimeric Antigen Receptor T-Cell Therapy (CAR-T)",
    "Chimeric Antigen Receptor NK-Cell Therapy (CAR-NK)",
    "Chimeric Antigen Receptor Gamma Delta (\u03b3\u03b4) T-Cell Therapy (CAR-\u03b3\u03b4T)",
    "Tumor Infiltrating Lymphocyte Therapy (TIL)",
    "T-Cell Replacement Therapy",
    "Cancer Vaccine",
    "Cell Therapy",
}

# Phase label lookup used by build_drug_phase_summary()
PHASE_LABEL_MAP = {v: k for k, v in {
    0.5: "Early Phase I",
    1.0: "Phase I",
    1.5: "Phase I/II",
    2.0: "Phase II",
    2.5: "Phase II/III",
    3.0: "Phase III",
    4.0: "Phase IV",
}.items()}
PHASE_NUMERIC_TO_LABEL = {
    0.5: "Early Phase I",
    1.0: "Phase I",
    1.5: "Phase I/II",
    2.0: "Phase II",
    2.5: "Phase II/III",
    3.0: "Phase III",
    4.0: "Phase IV",
}


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                    PHASE A – PARSE & PROFILE RAW DATA                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝


def load_and_profile_dataset(
    file_path: str, sheet_name: str
) -> pd.DataFrame:
    """
    Loads the target Excel sheet and prints basic shape, dtypes, and null
    statistics. Returns the raw DataFrame.
    """
    print("=" * 78)
    print("  PHASE A: DATA INGESTION & PROFILING")
    print("=" * 78)
    print(f"\n📂 Loading '{file_path}' — sheet '{sheet_name}' ...")

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    print(f"   ✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns\n")
    print("─" * 78)
    print("  BASIC SCHEMA OVERVIEW")
    print("─" * 78)
    for col in df.columns:
        print(f"  {col:<28s}  dtype={str(df[col].dtype):<16s}")
    print()
    return df


# ── A.1  Field completeness, null distribution, cardinality ──────────────────

def profile_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame summarising, for every column:
      - total values, non-null count, null count, completeness %
      - number of unique non-null values (cardinality)
    """
    records = []
    for col in df.columns:
        total     = len(df)
        non_null  = df[col].notna().sum()
        null_count = df[col].isna().sum()
        completeness = round(non_null / total * 100, 2)
        cardinality  = df[col].dropna().nunique()
        records.append({
            "column":        col,
            "total":         total,
            "non_null":      non_null,
            "null_count":    null_count,
            "completeness_%": completeness,
            "cardinality":   cardinality,
        })
    return pd.DataFrame(records)


def check_field_cardinality(
    df: pd.DataFrame, categorical_cols: list[str]
) -> dict[str, pd.Series]:
    """
    Returns frequency tables for each categorical column and detects
    potential spelling/casing variations.
    """
    return {col: df[col].value_counts(dropna=False) for col in categorical_cols}


# ── A.2  Dirty values: casing, free-text synonyms, date formats ─────────────

def detect_dirty_values(df: pd.DataFrame) -> dict[str, Any]:
    """
    Inspects categorical fields for inconsistent capitalisation and
    free-text synonyms. Checks date columns for mixed format issues.
    """
    report: dict[str, Any] = {}

    # Phase casing/synonym check
    raw_phases    = df["phase"].dropna().unique().tolist()
    unknown_phases = [p for p in raw_phases if p not in PHASE_STANDARD_MAP]
    report["phase_raw_values"]     = raw_phases
    report["phase_unknown_values"] = unknown_phases

    # Status casing/synonym check
    raw_statuses    = df["recruitment_status"].dropna().unique().tolist()
    unknown_statuses = [s for s in raw_statuses if s not in STATUS_GROUP_MAP]
    report["status_raw_values"]     = raw_statuses
    report["status_unknown_values"] = unknown_statuses

    # Date format consistency
    for dcol in ["start_date", "completion_date", "primary_completion_date"]:
        if df[dcol].dtype != "datetime64[ns]":
            report[f"{dcol}_format_issue"] = (
                f"Expected datetime64[ns], got {df[dcol].dtype}"
            )

    return report


# ── A.3  Structural anomalies ────────────────────────────────────────────────

def check_structural_anomalies(df: pd.DataFrame) -> dict[str, Any]:
    """
    Checks for:
      - Duplicate trial IDs (ID-datalake, nct_id)
    """
    return {
        "duplicate_ID-datalake": int(df["ID-datalake"].duplicated().sum()),
        "duplicate_nct_id":      int(df["nct_id"].duplicated().sum()),
    }


def validate_stringified_arrays(
    df: pd.DataFrame, array_cols: list[str]
) -> dict[str, Any]:
    """
    Safely parses stringified Python lists using ast.literal_eval.
    Reports parse failures per column and checks element-wise length
    alignment across drug/technology/target arrays.

    NOTE: ast.literal_eval is used (not json.loads) because the raw data
    uses Python-style single-quoted strings — valid Python literals but
    invalid JSON.
    """
    report: dict[str, Any] = {"parse_errors": {}, "alignment_errors": []}
    parsed: dict[str, list] = {}

    for col in array_cols:
        errors, results = 0, []
        for val in df[col]:
            try:
                results.append(ast.literal_eval(str(val)))
            except (ValueError, SyntaxError):
                errors += 1
                results.append(None)
        report["parse_errors"][col] = errors
        parsed[col] = results

    # Element-wise alignment: drugs_datalake must align with tech/target columns
    aligned_cols = [
        "drugs_datalake", "main_technologies", "specific_technologies",
        "target_names", "target_abbreviations",
    ]
    misaligned_rows = []
    for idx in range(len(df)):
        lengths = [
            len(parsed[col][idx]) if parsed[col][idx] is not None else -1
            for col in aligned_cols
        ]
        if len(set(lengths)) > 1:
            misaligned_rows.append({
                "row_index": idx,
                "nct_id":    df["nct_id"].iloc[idx],
                "lengths":   dict(zip(aligned_cols, lengths)),
            })
    report["alignment_errors"] = misaligned_rows
    return report


# ── A.4  Date and enrollment anomalies ──────────────────────────────────────

def detect_date_anomalies(df: pd.DataFrame) -> dict[str, Any]:
    """
    Flags:
      - Completion dates in the future for COMPLETED trials
      - Withdrawn trials with future completion dates
      - Negative trial durations
    """
    report: dict[str, Any] = {}
    today = pd.Timestamp.now().normalize()

    completed        = df[df["recruitment_status"] == "COMPLETED"]
    future_completion = completed[completed["completion_date"] > today]
    report["completed_with_future_completion"] = len(future_completion)
    if len(future_completion):
        report["completed_future_nct_ids"] = future_completion["nct_id"].tolist()

    withdrawn        = df[df["recruitment_status"] == "WITHDRAWN"]
    withdrawn_future = withdrawn[withdrawn["completion_date"] > today]
    report["withdrawn_with_future_completion"] = len(withdrawn_future)
    if len(withdrawn_future):
        report["withdrawn_future_nct_ids"] = withdrawn_future["nct_id"].tolist()

    has_both = df.dropna(subset=["start_date", "completion_date"])
    negative = has_both[has_both["completion_date"] < has_both["start_date"]]
    report["negative_duration_count"] = len(negative)

    return report


def detect_enrollment_anomalies(df: pd.DataFrame) -> dict[str, Any]:
    """
    Flags:
      - Non-null enrollment with missing enrollment_type
      - Withdrawn trials with enrollment > 0
    """
    report: dict[str, Any] = {}

    enroll_no_type = df[df["enrollment"].notna() & df["enrollment_type"].isna()]
    report["enrollment_without_type"] = len(enroll_no_type)
    if len(enroll_no_type):
        report["enrollment_without_type_nct_ids"] = enroll_no_type["nct_id"].tolist()

    withdrawn_enrolled = df[
        (df["recruitment_status"] == "WITHDRAWN") & (df["enrollment"] > 0)
    ]
    report["withdrawn_with_enrollment"] = len(withdrawn_enrolled)
    if len(withdrawn_enrolled):
        report["withdrawn_enrolled_nct_ids"] = withdrawn_enrolled["nct_id"].tolist()

    return report


def generate_dq_report(df: pd.DataFrame) -> dict[str, Any]:
    """
    Master function: runs all profiling checks and compiles a unified
    Data Quality Report.
    """
    print("\n" + "─" * 78)
    print("  DATA QUALITY REPORT")
    print("─" * 78)

    # 1. Completeness
    completeness = profile_completeness(df)
    print("\n📊 1. FIELD COMPLETENESS & CARDINALITY")
    print(completeness.to_string(index=False))

    # 2. Cardinality of categorical fields
    cat_cols   = ["phase", "recruitment_status", "enrollment_type"]
    cardinality = check_field_cardinality(df, cat_cols)
    print("\n📊 2. CATEGORICAL FIELD DISTRIBUTIONS")
    for col, freq in cardinality.items():
        print(f"\n  ── {col} ──")
        for val, cnt in freq.items():
            label = val if pd.notna(val) else "<NULL>"
            print(f"     {str(label):<30s}  {cnt:>4d}")

    # 3. Dirty value detection
    dirty = detect_dirty_values(df)
    print("\n📊 3. DIRTY VALUE DETECTION")
    print(f"  Phase raw values       : {dirty['phase_raw_values']}")
    print(f"  Phase unknown values   : {dirty['phase_unknown_values']}")
    print(f"  Status raw values      : {dirty['status_raw_values']}")
    print(f"  Status unknown values  : {dirty['status_unknown_values']}")

    # 4. Structural anomalies
    structural = check_structural_anomalies(df)
    print("\n📊 4. STRUCTURAL ANOMALIES")
    print(f"  Duplicate ID-datalake  : {structural['duplicate_ID-datalake']}")
    print(f"  Duplicate nct_id       : {structural['duplicate_nct_id']}")

    # 5. Array validation
    array_report = validate_stringified_arrays(df, ARRAY_COLUMNS)
    print("\n📊 5. STRINGIFIED ARRAY VALIDATION")
    print("  Parse errors per column:")
    for col, err_cnt in array_report["parse_errors"].items():
        status = "✅" if err_cnt == 0 else "❌"
        print(f"     {status}  {col:<28s}  errors={err_cnt}")
    print(
        f"  Alignment mismatches (drug ↔ tech ↔ target): "
        f"{len(array_report['alignment_errors'])}"
    )

    # 6. Date anomalies
    date_report = detect_date_anomalies(df)
    print("\n📊 6. DATE ANOMALIES")
    print(f"  Completed trials w/ future completion : {date_report['completed_with_future_completion']}")
    print(f"  Withdrawn trials w/ future completion : {date_report['withdrawn_with_future_completion']}")
    if date_report.get("withdrawn_future_nct_ids"):
        for nct in date_report["withdrawn_future_nct_ids"]:
            print(f"     ⚠️  {nct}")
    print(f"  Trials with negative duration         : {date_report['negative_duration_count']}")

    # 7. Enrollment anomalies
    enroll_report = detect_enrollment_anomalies(df)
    print("\n📊 7. ENROLLMENT ANOMALIES")
    print(f"  Enrollment present, type missing      : {enroll_report['enrollment_without_type']}")
    print(f"  Withdrawn trials with enrollment > 0  : {enroll_report['withdrawn_with_enrollment']}")
    if enroll_report.get("withdrawn_enrolled_nct_ids"):
        for nct in enroll_report["withdrawn_enrolled_nct_ids"]:
            print(f"     ⚠️  {nct}")

    print("\n" + "─" * 78)
    print("  END OF DATA QUALITY REPORT")
    print("─" * 78)

    return {
        "completeness":        completeness,
        "cardinality":         cardinality,
        "dirty_values":        dirty,
        "structural_anomalies": structural,
        "array_validation":    array_report,
        "date_anomalies":      date_report,
        "enrollment_anomalies": enroll_report,
    }


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║              PHASE B – NORMALISED ANALYTICAL SCHEMA DESIGN               ║
# ╚════════════════════════════════════════════════════════════════════════════╝


def parse_string_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all stringified Python list columns from text to actual
    Python lists using ast.literal_eval.

    NOTE: ast.literal_eval is used (not json.loads) because the raw data
    uses single-quoted strings — valid Python literals but invalid JSON.
    """
    df = df.copy()
    for col in ARRAY_COLUMNS:
        df[col] = df[col].apply(lambda x: ast.literal_eval(str(x)))
    return df


def standardize_trial_phases(
    phase_series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Maps raw phase strings to:
      1. standardized_phase  – human-readable label (e.g., 'Phase I/II')
      2. phase_numeric       – float for ordering (e.g., 1.5)

    EARLY_PHASE1 → 0.5 (not 0): it is a sub-phase of Phase I, not a full
    phase step below it. 0.5 reflects its proximity to Phase I on the
    development ladder and is consistent with PHASE_NUMERIC_MAP.

    Null phases map to 'Unspecified Phase' / NaN.
    """
    standardized = phase_series.map(PHASE_STANDARD_MAP).fillna(
        phase_series.apply(
            lambda x: "Unspecified Phase" if pd.isna(x) else PHASE_STANDARD_MAP.get(x, x)
        )
    )
    numeric = phase_series.map(PHASE_NUMERIC_MAP)
    return standardized, numeric


def standardize_recruitment_status(status_series: pd.Series) -> pd.Series:
    """Maps raw recruitment_status values to a controlled vocabulary group."""
    return status_series.map(STATUS_GROUP_MAP)


def engineer_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates trial_duration_days using a date fallback strategy:
      1. completion_date – start_date
      2. If completion_date is null → fall back to primary_completion_date
      3. If both completion dates are null → duration = NaN

    Also engineers start_year from start_date.

    Edge case: WITHDRAWN trials with 0 enrollment have completion dates
    nulled to prevent artificial duration calculations (these trials never
    started; any completion date is a data entry artefact).
    """
    df = df.copy()

    withdrawn_zero = (
        (df["recruitment_status"] == "WITHDRAWN") &
        (df["enrollment"].fillna(0) == 0)
    )
    df.loc[withdrawn_zero, "completion_date"]         = pd.NaT
    df.loc[withdrawn_zero, "primary_completion_date"] = pd.NaT

    effective_completion = df["completion_date"].fillna(df["primary_completion_date"])
    duration = (effective_completion - df["start_date"]).dt.days
    duration = duration.where(duration >= 0, other=np.nan)

    df["trial_duration_days"] = duration
    df["start_year"]          = df["start_date"].dt.year
    return df


def impute_enrollment_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Edge case: if enrollment > 0 and enrollment_type is missing, impute
    as 'ACTUAL' when the trial is COMPLETED or TERMINATED — enrollment was
    clearly achieved, so the number is an actual count.
    """
    df   = df.copy()
    mask = (
        df["enrollment_type"].isna() &
        df["enrollment"].notna() &
        (df["enrollment"] > 0) &
        df["recruitment_status"].isin(["COMPLETED", "TERMINATED"])
    )
    df.loc[mask, "enrollment_type"] = "ACTUAL"
    return df


def deduplicate_indication_lists(indications: list) -> list:
    """Remove duplicate indications within a single trial's list."""
    seen, deduped = set(), []
    for ind in indications:
        normed = ind.strip()
        if normed not in seen:
            seen.add(normed)
            deduped.append(normed)
    return deduped


def build_dim_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the DimTrials fact table with all standardised and engineered
    fields, including the two new analytical flags:

      is_right_censored  — True for the 259 Active/Ongoing trials whose
                           outcome is not yet known. These trials must be
                           excluded from success rate denominators in Part 2.
                           Rates computed without them underestimate true
                           success rates for recent cohorts.

      is_outcome_unknown — True for the 121 UNKNOWN trials (12.1% of dataset).
                           Outcome is unobservable from registry data alone.
                           Excluded from success rate denominators in Part 2.

    FIX 5: enrollment_per_month is now NaN for ESTIMATED enrollment.
           Dividing a projected enrollment count by trial duration produces
           a misleading rate. Only ACTUAL enrollment counts are used.

    FIX 6: drug_count uses drugs_datalake (catalogued therapeutic agents only),
           not interventions_drugs (which includes procedures, imaging, and
           supportive care that are not drugs).
    """
    trials = df[[
        "ID-datalake", "nct_id", "brief_title", "official_title",
        "phase", "standardized_phase", "phase_numeric",
        "recruitment_status", "standardized_status",
        "start_date", "completion_date", "primary_completion_date",
        "trial_duration_days", "start_year",
        "enrollment", "enrollment_type",
    ]].copy()

    trials = trials.rename(columns={"ID-datalake": "trial_id", "phase": "raw_phase"})

    # FIX 1 — is_right_censored
    trials["is_right_censored"] = df["recruitment_status"].isin(CENSORED_STATUSES)

    # FIX 3 — is_outcome_unknown
    trials["is_outcome_unknown"] = df["recruitment_status"] == "UNKNOWN"

    # FIX 6 — drug_count from drugs_datalake (not interventions_drugs)
    trials["drug_count"] = df["drugs_datalake"].apply(len)
    trials["is_combination_therapy"] = trials["drug_count"] > 1

    # Indication count (deduplicated)
    trials["indication_count"] = df["indications"].apply(
        lambda v: len(set(v)) if isinstance(v, list) else 0
    )

    # Stage flags
    trials["is_late_stage"]  = df["phase"].isin({"PHASE3", "PHASE4"})
    trials["is_early_phase"] = df["phase"].isin({"EARLY_PHASE1", "PHASE1"})

    # FIX 7 — is_immunotherapy using corrected IMMUNOTHERAPY_TECHS set
    def _has_immunotherapy(tech_list: Any) -> bool:
        if not isinstance(tech_list, list):
            return False
        for drug_techs in tech_list:
            if isinstance(drug_techs, list):
                if any(t in IMMUNOTHERAPY_TECHS for t in drug_techs):
                    return True
            elif drug_techs in IMMUNOTHERAPY_TECHS:
                return True
        return False

    trials["is_immunotherapy"] = df["main_technologies"].apply(_has_immunotherapy)

    # FIX 5 — enrollment_per_month restricted to ACTUAL enrollment only
    actual_enrollment = np.where(
        df["enrollment_type"] == "ACTUAL", df["enrollment"], np.nan
    )
    trials["enrollment_per_month"] = np.where(
        (df["enrollment_type"] == "ACTUAL") & (df["trial_duration_days"] > 0),
        actual_enrollment / (df["trial_duration_days"] / 30.44),
        np.nan,
    )

    return trials


def normalize_therapies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts DimTherapies and FactTrialTherapies from the parsed array
    columns, preserving element-wise index alignment.

    FIX 2 — DUMMY_AGENT removed: trials with empty drugs_datalake ([]) are
    skipped entirely. They remain in dim_trials and fact_trial_indications
    for indication and phase analysis, but produce no rows here. This
    prevents 77 heterogeneous trials (procedural, herbal, uncatalogued)
    from polluting technology-level success rate calculations in Part 2.
    """
    therapy_records  = []
    mapping_records  = []
    skipped_no_drug  = 0

    for _, row in df.iterrows():
        trial_id = row["ID-datalake"]
        drugs    = row["drugs_datalake"]
        main_tech = row["main_technologies"]
        spec_tech = row["specific_technologies"]
        tgt_names = row["target_names"]
        tgt_abbrs = row["target_abbreviations"]

        # FIX 2: skip trials with no catalogued drug — no placeholder created
        if len(drugs) == 0:
            skipped_no_drug += 1
            continue

        for j, drug_id in enumerate(drugs):
            m_tech = main_tech[j] if j < len(main_tech) else []
            s_tech = spec_tech[j] if j < len(spec_tech) else []
            t_name = tgt_names[j] if j < len(tgt_names) else []
            t_abbr = tgt_abbrs[j] if j < len(tgt_abbrs) else []

            m_tech_str = ", ".join(m_tech) if isinstance(m_tech, list) else str(m_tech)
            s_tech_str = ", ".join(s_tech) if isinstance(s_tech, list) else str(s_tech)
            t_name_str = ", ".join(t_name) if isinstance(t_name, list) else str(t_name)
            t_abbr_str = ", ".join(t_abbr) if isinstance(t_abbr, list) else str(t_abbr)

            therapy_records.append({
                "therapy_id":           drug_id,
                "main_technology":      m_tech_str,
                "specific_technology":  s_tech_str,
                "target_name":          t_name_str,
                "target_abbreviation":  t_abbr_str,
            })
            mapping_records.append({
                "trial_id":      trial_id,
                "therapy_id":    drug_id,
                "sequence_order": j,
            })

    print(f"     Skipped {skipped_no_drug} trials with empty drugs_datalake "
          f"(retained in dim_trials and indication tables).")

    dim_therapies = (
        pd.DataFrame(therapy_records)
        .drop_duplicates(subset=["therapy_id"])
        .reset_index(drop=True)
    )
    fact_trial_therapies = pd.DataFrame(mapping_records)

    return dim_therapies, fact_trial_therapies


def normalize_indications(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts DimIndications and FactTrialIndications by exploding the
    parsed indications list. Deduplicates within each trial first.
    All 1,000 trials are included regardless of drug catalogue status.
    """
    indication_records = []
    mapping_records    = []
    indication_set: dict[str, str] = {}

    for _, row in df.iterrows():
        trial_id       = row["ID-datalake"]
        raw_indications = row["indications"]
        clean_list     = deduplicate_indication_lists(raw_indications)

        for ind in clean_list:
            if ind not in indication_set:
                ind_id = f"IND_{abs(hash(ind)) % 100000:05d}"
                indication_set[ind] = ind_id
                indication_records.append({
                    "indication_id":          ind_id,
                    "standardized_indication": ind,
                })
            mapping_records.append({
                "trial_id":     trial_id,
                "indication_id": indication_set[ind],
            })

    return pd.DataFrame(indication_records), pd.DataFrame(mapping_records)


def build_drug_phase_summary(
    fact_trial_therapies: pd.DataFrame,
    dim_trials: pd.DataFrame,
) -> pd.DataFrame:
    """
    FIX 4 — Drug-level phase funnel support.

    Computes the highest phase reached per drug (therapy_id) across all
    trials it appears in. This is the correct basis for a phase transition
    funnel — a trial-level funnel conflates independent trials with
    sequential development programs for the same drug.

    Example: Pembrolizumab appears in Phase I, II, and III trials. A
    trial-level count would show 3 separate entries; this table shows
    one entry with max_phase_numeric=3.0.

    NOTE: max_phase reflects the highest phase observed in this 1,000-trial
    extract only — not the drug's full global development history.
    """
    merged = fact_trial_therapies.merge(
        dim_trials[["trial_id", "phase_numeric"]],
        on="trial_id",
        how="left",
    )
    summary = (
        merged.groupby("therapy_id")["phase_numeric"]
        .max()
        .reset_index()
        .rename(columns={"phase_numeric": "max_phase_numeric"})
    )
    summary["max_phase_label"] = summary["max_phase_numeric"].map(
        PHASE_NUMERIC_TO_LABEL
    )
    # Count how many distinct trials each drug appears in
    trial_counts = (
        fact_trial_therapies.groupby("therapy_id")["trial_id"]
        .nunique()
        .reset_index()
        .rename(columns={"trial_id": "n_trials"})
    )
    summary = summary.merge(trial_counts, on="therapy_id", how="left")
    return summary.sort_values("max_phase_numeric", ascending=False).reset_index(drop=True)


def get_normalized_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Master function that builds the full normalised schema:
      - dim_trials              (fact table — all 1,000 trials)
      - dim_therapies           (unique drugs with tech/target attributes)
      - fact_trial_therapies    (trial ↔ drug mapping)
      - dim_indications         (unique indications)
      - fact_trial_indications  (trial ↔ indication mapping)
      - drug_phase_summary      (highest phase per drug — for funnel analysis)
    """
    print("\n" + "=" * 78)
    print("  PHASE B: NORMALISED ANALYTICAL SCHEMA")
    print("=" * 78)

    # ① Parse stringified arrays
    print("\n  ① Parsing stringified arrays ...")
    df = parse_string_arrays(df)

    # ② Standardise phases
    print("  ② Standardising clinical phases ...")
    df["standardized_phase"], df["phase_numeric"] = standardize_trial_phases(df["phase"])
    print(f"     Phase mapping applied. Numeric range: "
          f"[{df['phase_numeric'].min()}, {df['phase_numeric'].max()}]")

    # ③ Standardise recruitment status
    print("  ③ Standardising recruitment statuses ...")
    df["standardized_status"] = standardize_recruitment_status(df["recruitment_status"])

    # ④ Impute enrollment type
    print("  ④ Imputing missing enrollment types ...")
    before_impute = df["enrollment_type"].isna().sum()
    df = impute_enrollment_type(df)
    after_impute  = df["enrollment_type"].isna().sum()
    print(f"     Imputed {before_impute - after_impute} enrollment_type values")

    # ⑤ Engineer duration features
    print("  ⑤ Engineering trial duration and start year ...")
    df = engineer_duration_features(df)
    print(f"     Valid durations: {df['trial_duration_days'].notna().sum()}, "
          f"Null: {df['trial_duration_days'].isna().sum()}")

    # ⑥ Build DimTrials
    print("  ⑥ Building DimTrials table ...")
    dim_trials = build_dim_trials(df)
    print(f"     DimTrials: {dim_trials.shape[0]} rows × {dim_trials.shape[1]} cols")
    print(f"     is_right_censored=True  : {dim_trials['is_right_censored'].sum()} trials "
          f"(Active/Ongoing — excluded from success rate denominators)")
    print(f"     is_outcome_unknown=True : {dim_trials['is_outcome_unknown'].sum()} trials "
          f"(UNKNOWN status — excluded from success rate denominators)")

    # ⑦ Normalise therapies
    print("  ⑦ Normalising therapies (drug ↔ technology ↔ target) ...")
    dim_therapies, fact_trial_therapies = normalize_therapies(df)
    print(f"     DimTherapies: {dim_therapies.shape[0]} unique therapies")
    print(f"     FactTrialTherapies: {fact_trial_therapies.shape[0]} mappings")

    # ⑧ Normalise indications
    print("  ⑧ Normalising indications ...")
    dim_indications, fact_trial_indications = normalize_indications(df)
    print(f"     DimIndications: {dim_indications.shape[0]} unique indications")
    print(f"     FactTrialIndications: {fact_trial_indications.shape[0]} mappings")

    # ⑨ Build drug-level phase summary (FIX 4)
    print("  ⑨ Building drug-level phase summary ...")
    drug_phase_summary = build_drug_phase_summary(fact_trial_therapies, dim_trials)
    print(f"     drug_phase_summary: {drug_phase_summary.shape[0]} unique drugs")

    # ── Validation checks ────────────────────────────────────────────────
    print("\n  🔍 VALIDATION CHECKS:")

    # V1: Therapy row count — no DUMMY_AGENT, so expected = sum of actual drug counts
    expected_therapy_rows = df["drugs_datalake"].apply(len).sum()
    actual_therapy_rows   = len(fact_trial_therapies)
    v1_pass = expected_therapy_rows == actual_therapy_rows
    print(f"     V1 - Therapy row count match       : {'✅ PASS' if v1_pass else '❌ FAIL'} "
          f"(expected={expected_therapy_rows}, actual={actual_therapy_rows})")

    # V2: Numeric phase range [0.5, 4.0]
    valid_phases = df["phase_numeric"].dropna()
    v2_pass = valid_phases.min() >= 0.5 and valid_phases.max() <= 4.0
    print(f"     V2 - Phase numeric range [0.5, 4.0]: {'✅ PASS' if v2_pass else '❌ FAIL'}")

    # V3: No negative durations
    v3_pass = (df["trial_duration_days"].dropna() >= 0).all() if df["trial_duration_days"].notna().any() else True
    print(f"     V3 - No negative durations         : {'✅ PASS' if v3_pass else '❌ FAIL'}")

    # V4: is_right_censored count (FIX 1)
    v4_pass = dim_trials["is_right_censored"].sum() == 259
    print(f"     V4 - is_right_censored count == 259: {'✅ PASS' if v4_pass else '❌ FAIL'} "
          f"(actual={dim_trials['is_right_censored'].sum()})")

    # V5: is_outcome_unknown count (FIX 3)
    v5_pass = dim_trials["is_outcome_unknown"].sum() == 121
    print(f"     V5 - is_outcome_unknown count == 121: {'✅ PASS' if v5_pass else '❌ FAIL'} "
          f"(actual={dim_trials['is_outcome_unknown'].sum()})")

    # V6: No DUMMY_AGENT in therapy tables (FIX 2)
    v6_pass = "DUMMY_AGENT" not in fact_trial_therapies["therapy_id"].values
    print(f"     V6 - No DUMMY_AGENT in therapies   : {'✅ PASS' if v6_pass else '❌ FAIL'}")

    tables = {
        "dim_trials":             dim_trials,
        "dim_therapies":          dim_therapies,
        "fact_trial_therapies":   fact_trial_therapies,
        "dim_indications":        dim_indications,
        "fact_trial_indications": fact_trial_indications,
        "drug_phase_summary":     drug_phase_summary,
    }

    # ── Schema summary ───────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("  NORMALISED SCHEMA SUMMARY")
    print("─" * 78)
    for name, table in tables.items():
        print(f"\n  📋 {name}  ({table.shape[0]} rows × {table.shape[1]} cols)")
        print(f"     Columns: {table.columns.tolist()}")
        print(f"     Sample (first 2 rows):")
        for line in table.head(2).to_string(index=False).split("\n"):
            print(f"       {line}")

    return tables


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                            MAIN EXECUTION                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝


def main():
    """Run the full Part 1 pipeline: Ingestion → Profiling → Normalisation."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── PHASE A ──────────────────────────────────────────────────────────
    df        = load_and_profile_dataset(INPUT_FILE, SHEET_NAME)
    dq_report = generate_dq_report(df)

    dq_report["completeness"].to_csv(
        os.path.join(OUTPUT_DIR, "dq_completeness_report.csv"), index=False
    )

    # ── PHASE B ──────────────────────────────────────────────────────────
    tables = get_normalized_tables(df)

    for name, table in tables.items():
        out_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        table.to_csv(out_path, index=False)
        print(f"  💾 Saved: {out_path}")

    print("\n" + "=" * 78)
    print("  ✅  PART 1 COMPLETE — All tables exported to ./output/")
    print("=" * 78)

    # ── Documented Assumptions & Limitations ─────────────────────────────
    print("\n📝 DOCUMENTED ASSUMPTIONS & LIMITATIONS:")
    print("  1. ast.literal_eval is used (not json.loads) — raw data uses")
    print("     single-quoted Python strings, which are invalid JSON.")
    print("  2. Transitional phases mapped to floats (e.g., Phase I/II → 1.5)")
    print("     assumes a linear clinical progression model.")
    print("  3. EARLY_PHASE1 → 0.5 (not 0): it is a sub-phase of Phase I,")
    print("     not a full phase step below it.")
    print("  4. Withdrawn trials with 0 enrollment have completion dates nulled")
    print("     to prevent artificial duration calculations.")
    print("  5. Missing enrollment_type imputed as ACTUAL only for COMPLETED or")
    print("     TERMINATED trials with non-zero enrollment.")
    print("  6. 77 trials with empty drugs_datalake are excluded from therapy")
    print("     and technology tables but retained in dim_trials and indication")
    print("     tables. No DUMMY_AGENT placeholder is used.")
    print("  7. 121 UNKNOWN trials (12.1%) are excluded from success rate")
    print("     denominators in Part 2. Flag: is_outcome_unknown=True.")
    print("  8. 259 Active/Ongoing trials are right-censored — outcome not yet")
    print("     known. Excluded from success rate denominators. Rates for recent")
    print("     cohorts are likely underestimated. Flag: is_right_censored=True.")
    print("  9. enrollment_per_month is NaN for ESTIMATED enrollment — dividing")
    print("     a projected count by duration produces a misleading rate.")
    print(" 10. drug_count uses drugs_datalake (catalogued therapeutic agents),")
    print("     not interventions_drugs (which includes procedures and imaging).")
    print(" 11. IMMUNOTHERAPY_TECHS excludes plain 'Antibody' (too broad) and")
    print("     ADCs (cytotoxic payload). Includes CAR-T, CAR-NK, TIL, Cancer")
    print("     Vaccine, Cell Therapy, T-Cell Replacement Therapy, CAR-γδT.")
    print(" 12. drug_phase_summary reflects the highest phase in this 1,000-trial")
    print("     extract only — not each drug's full global development history.")

    return df, dq_report, tables


if __name__ == "__main__":
    raw_df, quality_report, normalized_tables = main()
