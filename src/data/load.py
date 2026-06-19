"""Load and join Indiana University CXR CSV files, validate dataset integrity.

IU dataset structure (Kaggle: raddar/chest-xrays-indiana-university):
  indiana_reports.csv    — uid, findings, impression, indication, MeSH, Problems
  indiana_projections.csv — uid, filename, projection (Frontal / Lateral)

One row per study (uid) is the working unit: frontal + lateral images of the
same patient study must always land in the same split.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

REPORTS_FILE = "indiana_reports.csv"
PROJECTIONS_FILE = "indiana_projections.csv"


def load_reports(raw_dir: Path) -> pd.DataFrame:
    path = Path(raw_dir) / REPORTS_FILE
    df = pd.read_csv(path)
    logger.info("Loaded %d report rows from %s", len(df), path)
    return df


def load_projections(raw_dir: Path) -> pd.DataFrame:
    path = Path(raw_dir) / PROJECTIONS_FILE
    df = pd.read_csv(path)
    logger.info("Loaded %d projection rows from %s", len(df), path)
    return df


def build_study_df(reports: pd.DataFrame, projections: pd.DataFrame) -> pd.DataFrame:
    """One row per study. Image filenames are collapsed to lists by projection type."""
    proj_grouped = (
        projections.groupby(["uid", "projection"])["filename"]
        .apply(list)
        .unstack(fill_value=[])
        .reset_index()
    )
    # Normalise projection column names (e.g. "Frontal" -> "frontal")
    proj_grouped.columns = [c.lower() for c in proj_grouped.columns]
    df = reports.merge(proj_grouped, on="uid", how="left")
    return df


def validate_integrity(df: pd.DataFrame) -> dict:
    """Log and return integrity counts. Does not drop rows."""
    n_total = len(df)

    n_no_frontal = 0
    if "frontal" in df.columns:
        n_no_frontal = df["frontal"].apply(lambda x: not x if isinstance(x, list) else pd.isna(x)).sum()

    findings_missing = df["findings"].isna() | (df["findings"].astype(str).str.strip() == "")
    n_no_findings = int(findings_missing.sum())

    report = {
        "total_studies": n_total,
        "studies_without_frontal_image": int(n_no_frontal),
        "studies_with_empty_findings": n_no_findings,
    }
    for k, v in report.items():
        level = logger.warning if v > 0 and k != "total_studies" else logger.info
        level("%s: %s", k, v)
    return report


def load_dataset(
    raw_dir: Path,
    images_dir: Optional[Path] = None,
    drop_empty_findings: bool = True,
) -> pd.DataFrame:
    """Load, join, and validate the IU CXR dataset.

    Args:
        raw_dir: Directory containing the two CSV files.
        images_dir: If given, absolute image paths are added as
            ``frontal_paths`` / ``lateral_paths`` columns.
        drop_empty_findings: Drop studies with no Findings text (cannot be
            used as generation targets).

    Returns:
        DataFrame with one row per study (uid).
    """
    reports = load_reports(raw_dir)
    projections = load_projections(raw_dir)

    df = build_study_df(reports, projections)
    validate_integrity(df)

    if drop_empty_findings:
        mask = df["findings"].notna() & (df["findings"].astype(str).str.strip() != "")
        before = len(df)
        df = df[mask].reset_index(drop=True)
        logger.info("Dropped %d studies with empty findings (%d remain)", before - len(df), len(df))

    if images_dir is not None:
        images_dir = Path(images_dir)
        for proj in ("frontal", "lateral"):
            if proj in df.columns:
                df[f"{proj}_paths"] = df[proj].apply(
                    lambda fnames: [str(images_dir / f) for f in fnames] if isinstance(fnames, list) else []
                )

    return df


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    raw_dir = Path(params["data"]["raw_dir"])
    processed_dir = Path(params["data"]["processed_dir"])
    images_dir = Path(params["data"]["images_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(raw_dir, images_dir=images_dir)
    out = processed_dir / "dataset.parquet"
    df.to_parquet(out, index=False)
    logger.info("Saved %d studies to %s", len(df), out)
