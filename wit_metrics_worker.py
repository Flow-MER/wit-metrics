"""
wit_metrics_worker.py

USAGE:
Run this module directly as a standalone Python script. It reads configuration
from config.py and does not require the legacy notebook.

"""

import io
import logging
import multiprocessing
import sys
import time
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO

import fiona
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely import geometry
from tqdm.auto import tqdm

# Add project root to sys.path to handles cases where the script is in a subfolder
current_path = Path(__file__).resolve().parent
project_root = current_path if (current_path / "config.py").exists() else current_path.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from config import WITMetricsConfig, load_config

logger = logging.getLogger(__name__)


def setup_logging(log_level: str | int = logging.INFO) -> None:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _running_in_jupyter_kernel() -> bool:
    """Return whether progress is being rendered inside a Jupyter kernel."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def progress(iterable, **kwargs):
    """Create a progress bar suitable for notebooks, terminals, and redirected logs."""
    kwargs.setdefault("disable", False if _running_in_jupyter_kernel() else None)
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(iterable, **kwargs)


# ------------------------------------------------------------------------------
# validate inputs
# ------------------------------------------------------------------------------


def validate_config(config: WITMetricsConfig) -> None:
    """
    Validates configuration before processing starts.
    Catches issues early rather than failing hours into a batch job.
    """
    errors = []
    warnings = []

    # 1. Check critical paths exist
    if not config.wit_csv_path.exists():
        errors.append(f"wit_csv_path is not a directory: {config.wit_csv_path}")

    # Check if shapefile exists (only if area lookup is enabled)
    if config.shapefile_path is not None:
        if not config.shapefile_path.exists():
            errors.append(f"shapefile_path does not exist: {config.shapefile_path}")
        if config.shapefile_path.suffix != ".shp":
            warnings.append(f"shapefile_path doesn't have .shp extension: {config.shapefile_path}")

    # Check value ranges
    if not 0 <= config.pc_missing_threshold <= 1:
        errors.append(
            f"pc_missing_threshold must be between 0 and 1, got: {config.pc_missing_threshold}"
        )

    if config.batch_size < 1:
        errors.append(f"BATCH_SIZE must be >= 1, got: {config.batch_size}")

    if config.threshold_percentile < 0 or config.threshold_percentile > 1:
        errors.append(
            f"threshold_percentile must be between 0 and 1 (0.3 recommended), got: {config.threshold_percentile}"
        )

    if config.min_threshold < 0 or config.min_threshold > 1:
        errors.append(
            f"min_threshold must be between 0 and 1 (0.05 recommended), got: {config.min_threshold}"
        )

    if config.max_threshold < 0 or config.max_threshold > 1:
        errors.append(
            f"max_threshold must be between 0 and 1 (0.5 recommended), got: {config.max_threshold}"
        )

    # Check MONTHLY_SUBSET validity
    if config.monthly_subset is not None:
        required_cols = ["feature_id", "date"]
        missing = [col for col in required_cols if col not in config.monthly_subset]
        if missing or len(config.monthly_subset) <= 2:
            errors.append(
                f"monthly_subset must include {required_cols} and at least one metric column"
            )

    # Report results
    if warnings:
        for w in warnings:
            logger.warning("Config warning: %s", w)

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)


# ------------------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------------------


def shape_list(
    key: str, values: Iterable, shapefile: str | Path
) -> Iterator[tuple[Any, dict[str, Any]]]:
    """
    Yields features from a shapefile that match the given keys.

    Args:
        key (str): The property name in the shapefile to filter by.
        values (iterable): A list or set of values to match against the key.
        shapefile (str): Path to the shapefile.

    Yields:
        tuple: A tuple containing (key_value, feature_object).
    """
    values = set(values)
    with fiona.open(shapefile) as src:
        for feat in src:
            k = feat["properties"].get(key)
            if k in values:
                yield k, feat


def get_areas(polygons: Iterable[tuple[Any, dict[str, Any]]], pkey: str) -> pd.DataFrame | None:
    """
    Calculates the area of each feature in hectares.

    Args:
        polygons (iterable): An iterable of (id, feature_object) tuples from shape_list.
        pkey (str): The column name to use for the ID in the output DataFrame.

    Returns:
        pd.DataFrame: A DataFrame indexed by pkey containing the 'area' column,
                      or None if no features are provided.
    """
    rows = []
    for poly_id, poly in polygons:
        # Calculate area and convert to hectares (1e4 m^2 = 1 ha)
        area = geometry.shape(poly["geometry"]).area / 1e4
        rows.append((poly_id, area))
    if not rows:
        return None
    return pd.DataFrame(rows, columns=[pkey, "area"]).set_index(pkey)


# ------------------------------------------------------------------------------
# Aggregation helpers
# ------------------------------------------------------------------------------


def _add_combined_members(df: pd.DataFrame, members: list[Any]) -> pd.DataFrame:
    """
    Adds combined metric columns to the DataFrame (e.g., 'water+wet').

    Args:
        df (pd.DataFrame): The input DataFrame.
        members (list): A list of members to process. Can be strings or lists of strings.
                        If a list is provided (e.g., ['water', 'wet']), a new column
                        'water+wet' is created by summing them.

    Returns:
        pd.DataFrame: The DataFrame with additional combined columns.
    """
    for m in members:
        if isinstance(m, list):
            name = "+".join(m)
            if name not in df:
                df[name] = df[m].sum(axis=1)
    return df


def _resample_metrics(df: pd.DataFrame, freq: str, pkey: str) -> pd.DataFrame:
    """
    Standard resampling logic.
    Calculates each stat independently and merges them at the end.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        freq (str): The frequency string for resampling (e.g., 'YS' for year start).
        pkey (str): The column name to group by (e.g., 'feature_id').

    Returns:
        pd.DataFrame: A DataFrame with resampled statistics, reset index.
    """
    # 1. Group and Resample
    g = df.groupby(pkey).resample(freq, include_groups=False)

    # 2. Get the count (as a DataFrame so we can concat to it)
    count_df = g.size().astype("int32").to_frame(name="count")

    # 3. Identify numeric columns
    numeric_cols = df.select_dtypes(include="number").columns

    # 4. Explicitly calculate each block
    stats_dfs = [count_df]
    stats_to_run = ["max", "min", "mean", "median"]

    for stat in stats_to_run:
        # Get the stat (e.g., g.max(), g.mean())
        # getattr(g[numeric_cols], stat)() is equivalent to g[numeric_cols].max()
        res_stat = getattr(g[numeric_cols], stat)()

        # Rename: 'water' becomes 'water_max'
        res_stat.columns = [f"{c}_{stat}" for c in res_stat.columns]

        stats_dfs.append(res_stat)

    # 5. Concatenate all stats horizontally
    # Since they all share the same (pkey, date) index, they align perfectly
    res = pd.concat(stats_dfs, axis=1)

    return res.reset_index()


# ------------------------------------------------------------------------------
# Annual / Monthly metrics
# ------------------------------------------------------------------------------


def annual_metrics(
    wit_data: pd.DataFrame,
    output_path: Path,
    members: list[Any] | None = None,
    pkey: str = "feature_id",
) -> pd.DataFrame:
    """
    Computes annual statistics for WIT members using optimized single-pass resampling.

    Args:
        wit_data (pd.DataFrame): The input WIT data.
        members (list, optional): List of metrics to compute. Defaults to standard set.
        pkey (str, optional): Primary key column. Defaults to "feature_id".

    Returns:
        pd.DataFrame: The computed annual metrics.
    """
    chunk = int(wit_data["chunk"].iat[0])
    members = members or [
        "pv",
        "wet",
        "water",
        "bs",
        "npv",
        ["npv", "pv", "wet"],
        ["pv", "wet"],
        ["water", "wet"],
    ]

    # 1. Create a local copy to avoid side-effects on the main wit_data
    # Use errors='ignore' in case columns were already dropped by monthly_metrics
    df = wit_data.drop(columns=["chunk", "pc_missing"], errors="ignore").set_index("date")

    # 2. Add composite columns (e.g., water+wet)
    df = _add_combined_members(df, members)

    # 3. Resample using the optimized single-pass _resample_metrics
    # "YS" = Year Start (Jan 1st)
    res = _resample_metrics(df, "YS", pkey)

    # 4. Insert a clean 'year' column for easier end-user filtering
    res.insert(1, "year", res["date"].dt.year)

    # 5. Save output
    out = output_path / f"WIT_yearly_metrics{chunk}.parquet"
    write_batch_parquet(res, out)

    return res


def monthly_metrics(
    wit_data: pd.DataFrame,
    output_path: Path,
    members: list[Any] | None = None,
    pkey: str = "feature_id",
) -> pd.DataFrame:
    """
    Computes monthly statistics for WIT members.

    Args:
        wit_data (pd.DataFrame): The input WIT data.
        members (list, optional): List of metrics to compute.
        pkey (str, optional): Primary key column.

    Returns:
        pd.DataFrame: The computed monthly metrics.
    """
    chunk = int(wit_data["chunk"].iat[0])
    members = members or [
        "pv",
        "wet",
        "water",
        "bs",
        "npv",
        ["npv", "pv", "wet"],
        ["pv", "wet"],
        ["water", "wet"],
    ]

    # Drop non-essential columns and set date as index for resampling
    df = wit_data.drop(columns=["chunk", "pc_missing"], errors="ignore").set_index("date")
    df = _add_combined_members(df, members)

    # Use the optimized resampler
    res = _resample_metrics(df, "MS", pkey)

    # Add helper columns for easy filtering later
    res.insert(1, "month", res["date"].dt.month)
    res.insert(1, "year", res["date"].dt.year)

    out = output_path / f"WIT_monthly_metrics{chunk}.parquet"
    write_batch_parquet(res, out)
    return res


# ------------------------------------------------------------------------------
# Inundation detection (vectorised, no apply)
# ------------------------------------------------------------------------------


def _event_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Generates a table of inundation events and gaps based on a threshold.

    Logic:
    1. Identifies periods where 'water+wet' > threshold.
    2. Groups contiguous periods into events.
    3. Calculates duration (days) and aggregated area-days for events.
    4. Calculates gaps between events.
    5. Ensures the total duration + gaps equals the total record span (Golden Rule).

    Args:
        df (pd.DataFrame): Input DataFrame with 'date' and 'water+wet' columns.
        threshold (float): The threshold value to define an event.

    Returns:
        pd.DataFrame: DataFrame with columns [start_date, end_date, duration, gap, area_days].
    """
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["start_date", "end_date", "duration", "gap", "area_days"])

    record_start = df["date"].min()
    record_end = df["date"].max()

    df["above"] = df["water+wet"] > threshold

    # CASE 1: Never Wet
    if not df["above"].any():
        total_span = (record_end - record_start).days + 1
        return pd.DataFrame(
            [
                {
                    "start_date": None,
                    "end_date": None,
                    "duration": 0,
                    "gap": total_span,
                    "area_days": 0.0,
                }
            ]
        )

    # Identify contiguous event blocks
    # Added 'area_days' to the aggregation as the sum of fractions
    df["group_id"] = (df["above"] != df["above"].shift()).cumsum()
    event_groups = (
        df[df["above"]]
        .groupby("group_id")
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            area_days=(
                "water+wet",
                "sum",
            ),  # <--- Sum of fractions (e.g., 0.8 + 0.9 + 0.4 = 2.1)
        )
        .sort_values("start_date")
        .reset_index(drop=True)
    )

    result = []

    # CASE 2: The Leading Gap
    first_event_start = event_groups.iloc[0]["start_date"]
    leading_gap = (first_event_start - record_start).days
    if leading_gap > 0:
        result.append(
            {
                "start_date": None,
                "end_date": None,
                "duration": 0,
                "gap": leading_gap,
                "area_days": 0.0,
            }
        )

    # CASE 3: Events and their Following Gaps
    num_events = len(event_groups)
    for i in range(num_events):
        curr_s = event_groups.iloc[i]["start_date"]
        curr_e = event_groups.iloc[i]["end_date"]
        # Pull the aggregated sum
        area_days = event_groups.iloc[i]["area_days"]
        duration = (curr_e - curr_s).days + 1

        if i < num_events - 1:
            next_s = event_groups.iloc[i + 1]["start_date"]
            gap = (next_s - curr_e).days - 1
        else:
            gap = (record_end - curr_e).days

        result.append(
            {
                "start_date": curr_s,
                "end_date": curr_e,
                "duration": duration,
                "gap": gap,
                "area_days": area_days,
            }
        )

    return pd.DataFrame(result)


# ------------------------------------------------------------------------------
# Inundation detection with robust stats
# ------------------------------------------------------------------------------


def inundation_metrics(
    wit_data: pd.DataFrame,
    threshold_df: pd.DataFrame | float,
    shapefile_path: Path,
    shapefile_key: str,
    output_path: Path,
    *,
    debug_event_times: bool = False,
    pkey: str = "feature_id",
) -> pd.DataFrame:
    """
    Calculates inundation metrics for each feature in the batch.

    For each feature:
    1. Retrieves the adaptive threshold.
    2. Generates the event table using `_event_table`.
    3. Calculates statistics for each event (max/mean water, cumulative area-days).
    4. Merges event details with statistics.

    Args:
        wit_data (pd.DataFrame): Batch of WIT data.
        threshold_df (pd.DataFrame | float): Thresholds for each feature.
        shapefile_path (Path): Path to the shapefile for area lookup.
        shapefile_key (str): Key in shapefile to match feature_id.
        debug_event_times (bool): If True, saves intermediate event timing data.
        pkey (str): Primary key column name.

    Returns:
        pd.DataFrame: DataFrame containing inundation events and metrics.
    """
    chunk = int(wit_data["chunk"].iat[0])

    df = wit_data[[pkey, "date", "water", "wet"]].copy()
    df["water+wet"] = df["water"] + df["wet"]

    # Area lookup
    area = None
    if Path(shapefile_path).is_file():
        polys = shape_list(shapefile_key, df[pkey].unique(), shapefile_path)
        area = get_areas(polys, pkey)

    events = []
    event_times = []
    stats = []

    # Local threshold reference to avoid modifying the input variable in loop
    for feature_id, group_df in df.groupby(pkey, group_keys=False, sort=False):
        # Clean threshold lookup
        if isinstance(threshold_df, pd.DataFrame):
            # Use 'water+wet' if it exists, otherwise the first numeric column
            thresh_col = (
                "water+wet" if "water+wet" in threshold_df.columns else threshold_df.columns[0]
            )
            current_threshold = threshold_df.loc[feature_id, thresh_col]
        else:
            # Fallback for scalar input
            current_threshold = threshold_df

        event_df = _event_table(group_df, current_threshold)
        if event_df.empty:
            continue

        # Ensure consistent dtypes to avoid FutureWarning in pd.concat
        # None values cause the warning come from Leading Gaps—periods of dryness at the start of the record.
        # "Never Wet" Sites: The entire record is one big gap. start_date is None.
        # Sites that Start Dry: If a site is dry for the first month and then floods, the code creates a "Leading Gap" row where start_date is None,
        # this converts None to NaT (not a time) which is compatible with wet sites for concatenation
        event_df["start_date"] = pd.to_datetime(event_df["start_date"])
        event_df["end_date"] = pd.to_datetime(event_df["end_date"])

        # Add metadata
        event_df.insert(0, pkey, feature_id)
        event_df.insert(1, "threshold", current_threshold)
        events.append(event_df)

        if debug_event_times:
            event_times.append(
                event_df[[pkey, "threshold", "start_date", "end_date", "duration", "gap"]]
            )

        # Compute water metrics
        gvals = group_df.set_index("date")["water+wet"]
        feat_area = (
            area.loc[feature_id, "area"]
            if area is not None and feature_id in area.index
            else np.nan
        )

        for _, r in event_df.iterrows():
            # GOLDEN RULE GUARD: Skip stats calculation for gap-only rows
            if pd.isna(r.start_date) or r.duration == 0:
                stats.append(
                    {
                        pkey: feature_id,
                        "start_date": r.start_date,
                        "max_water+wet": np.nan,
                        "mean_water+wet": np.nan,
                        "max_wet_area": np.nan,
                        "mean_wet_area": np.nan,
                        "cum_area_days": 0.0,
                    }
                )
                continue

            # Standard stats for actual inundation events
            event_slice = gvals.loc[r.start_date : r.end_date]
            max_wet = event_slice.max()
            mean_wet = event_slice.mean()

            # INTEGRATION STEP:
            # Physical area (m2) * fractional days (sum of water+wet)
            cum_area_days = r.area_days * feat_area

            # Mean physical area for this event
            mean_area_m2 = mean_wet * feat_area

            stats.append(
                {
                    pkey: feature_id,
                    "start_date": r.start_date,
                    "max_water+wet": max_wet,
                    "mean_water+wet": mean_wet,
                    "max_wet_area": max_wet * feat_area,
                    "mean_wet_area": mean_area_m2,
                    "cum_area_days": cum_area_days,
                }
            )

    if not events:
        return pd.DataFrame()

    event_df = pd.concat(events, ignore_index=True)
    if stats:
        st = pd.DataFrame(stats)
        # Merging on pkey and start_date (which includes NaT)
        event_df = event_df.merge(st, on=[pkey, "start_date"], how="left")

    # Date formatting (handling NaT gracefully)
    event_df["start_date"] = pd.to_datetime(event_df["start_date"]).dt.date
    event_df["end_date"] = pd.to_datetime(event_df["end_date"]).dt.date

    # Output management
    out = output_path / f"WIT_inundation_metrics{chunk}.parquet"
    write_batch_parquet(event_df, out)

    if debug_event_times and event_times:
        debug_df = pd.concat(event_times, ignore_index=True)
        debug_df["start_date"] = pd.to_datetime(debug_df["start_date"]).dt.date
        debug_df["end_date"] = pd.to_datetime(debug_df["end_date"]).dt.date
        write_batch_parquet(debug_df, output_path / f"WIT_event_times{chunk}.parquet")

    return event_df


def interpolate_daily(wit_data: pd.DataFrame, pkey: str = "feature_id") -> pd.DataFrame:
    """
    Interpolates WIT data to a daily frequency using a linear interpolation

    Args:
        wit_data (pd.DataFrame): Input WIT data.
        pkey (str, optional): Primary key column. Defaults to "feature_id".

    Returns:
        pd.DataFrame: Daily interpolated DataFrame.
    """

    df = wit_data.copy()
    df["date"] = pd.to_datetime(df["date"])

    out = []

    for gid, g in df.groupby(pkey, sort=False):
        g = g.set_index("date")

        # numeric columns
        numeric = (
            g.select_dtypes("number")
            .resample("D")
            .mean()  # collapse same-day duplicates
            .interpolate()
        )

        # object columns
        objects = g.select_dtypes("object").resample("D").first().ffill()

        daily = objects.join(numeric)
        daily[pkey] = gid
        out.append(daily.reset_index())

    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------------------
# Time since last inundation (correct for never-inundated)
# ------------------------------------------------------------------------------


def time_since_last_inundation(
    wit_data: pd.DataFrame, wit_im: pd.DataFrame | None, output_path: Path, pkey: str = "feature_id"
) -> pd.DataFrame:
    """
    Calculates time since last inundation using the 'Gap After' logic.
    """
    chunk = int(wit_data["chunk"].iat[0])

    # 1. Handle the 'Never Wet' case or missing metrics
    if wit_im is None or wit_im.empty:
        # Fallback: total record span per feature
        span = (
            wit_data.groupby(pkey, sort=False)["date"]
            .agg(lambda x: (x.max() - x.min()).days)
            .reset_index()
        )
        span.columns = [pkey, "timesincelast"]
    else:
        # 2. Extract the gap from the very last record of each feature
        # Since gaps now follow events, the 'timesincelast' is just the 'gap'
        # of the final entry in the event table.
        span = (
            wit_im.sort_values([pkey, "end_date"], na_position="first")
            .groupby(pkey, sort=False)
            .tail(1)[[pkey, "gap"]]
            .rename(columns={"gap": "timesincelast"})
        )

    # 3. Save and return
    # span["chunk"] = chunk
    out = output_path / f"WIT_time_since_last_inundation{chunk}.parquet"
    write_batch_parquet(span, out)

    return span


def adaptive_inundation_threshold(
    wit_data: pd.DataFrame,
    output_path: Path,
    members: list[list[str]] | None = None,
    pkey: str = "feature_id",
    threshold_percentile: float = 0.3,
    min_threshold: float = 0.05,
    max_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Computes an adaptive inundation threshold for each feature.

    Logic:
    1. Calculates the specified percentile (e.g. 0.3 = 30th) of 'water+wet'.
    2. Clips the result between `min_threshold` and `max_threshold`.

    Args:
        wit_data (pd.DataFrame): Input WIT data.
        members (list): List of columns to combine for the metric (default: water + wet).
        pkey (str): Primary key column.
        min_threshold (float): Minimum allowed threshold (floor).
        max_threshold (float): Maximum allowed threshold (ceiling).

    Returns:
        pd.DataFrame: DataFrame indexed by pkey with the calculated threshold.
    """
    if members is None:
        members = [["water", "wet"]]

    chunk = int(wit_data["chunk"].iat[0])

    # 1. Identify and combine members
    flat_members = list({item for sublist in members for item in sublist})
    wit_df = wit_data[[pkey] + flat_members].copy()
    wit_df = _add_combined_members(wit_df, members)

    # 2. Compute P30 (30th percentile)
    # We use 'water+wet' specifically as the benchmark for inundation
    # P30 is used to handle perennial water contamination by picking a lower baseline.
    thresholds = wit_df.groupby(pkey)["water+wet"].quantile(threshold_percentile)

    # 3. Adaptive Selection Logic
    thresholds = thresholds.clip(lower=min_threshold, upper=max_threshold)

    # Convert to DataFrame for consistent indexing in downstream joins
    threshold_df = thresholds.to_frame(name="water+wet")

    # 4. Safety Logging
    total = len(threshold_df)

    logger.debug(
        "Chunk %s has %s features. Adaptive inundation threshold %s with Floor (%s) "
        "and Ceiling (%s).",
        chunk,
        total,
        threshold_percentile,
        min_threshold,
        max_threshold,
    )

    # 5. Save and Return
    out_file = output_path / f"WIT_event_threshold{chunk}.parquet"
    write_batch_parquet(threshold_df.reset_index(), out_file)

    return threshold_df


def _batch_columns(
    out_files: list[Path],
    fname: str,
    monthly_subset: list[str] | None,
) -> tuple[dict[Path, list[str]], list[str]]:
    """Return per-file read columns and the combined output column order."""
    file_columns = {}
    output_columns = []
    seen_columns = set()

    for out_file in out_files:
        try:
            columns = pq.read_schema(out_file).names
        except Exception as ex:
            logger.error("Error reading file schema: %s - %s", out_file, ex)
            raise

        selected_columns = columns
        if monthly_subset and "monthly" in fname:
            missing_columns = set(monthly_subset) - set(columns)
            if missing_columns:
                logger.warning(
                    "Monthly subset columns missing from %s: %s",
                    out_file,
                    missing_columns,
                )
            selected_columns = [column for column in monthly_subset if column in columns]

        file_columns[out_file] = selected_columns
        for column in selected_columns:
            if column not in seen_columns:
                output_columns.append(column)
                seen_columns.add(column)

    return file_columns, output_columns


def _write_merged_batches(
    csv_output: TextIO,
    out_files: list[Path],
    file_columns: dict[Path, list[str]],
    output_columns: list[str],
    labels_df: pd.DataFrame | None,
    labels_join_field: str,
) -> None:
    """Convert Parquet batches to one CSV stream without retaining prior batches."""
    write_header = True
    for out_file in progress(out_files):
        try:
            batch = pd.read_parquet(out_file, columns=file_columns[out_file])
        except Exception as ex:
            logger.error("Error reading file: %s - %s", out_file, ex)
            raise

        if batch.columns.tolist() != output_columns:
            batch = batch.reindex(columns=output_columns)
        if labels_df is not None:
            if labels_join_field not in batch.columns:
                raise ValueError(
                    f"Label join field is missing from batch output: {labels_join_field}"
                )
            try:
                batch = batch.merge(labels_df, on=labels_join_field)
            except Exception as ex:
                logger.error("Error merging labels: %s", ex)
                raise

        batch.to_csv(
            csv_output,
            index=False,
            header=write_header,
            float_format="%.4f",
        )
        csv_output.flush()
        write_header = False
        del batch


def merge_batches(
    path: Path,
    output_filenames: list[str],
    *,
    tag: str = "RESULT",
    monthly_subset: list[str] | None = None,
    labels_df: pd.DataFrame | None = None,
    labels_join_field: str = "feature_id",
    zip_result: bool = False,
) -> None:
    """
    Merges batch output files into single result files.

    Args:
        path (str): Directory containing batch files.
        output_filenames (list): List of base filenames to merge.
        tag (str, optional): Prefix for the output file. Defaults to "RESULT".
        monthly_subset (list, optional): Columns to keep for monthly metrics to save space.
        labels_df (pd.DataFrame, optional): DataFrame to join with results (e.g., for labels).
        labels_join_field (str, optional): Field to join labels on. Defaults to "feature_id".
    """
    path = Path(path)
    tag = tag + "_" if tag else "RESULT_"
    for fname in output_filenames:
        out_files = list(path.glob(f"{fname}*.parquet"))
        if not out_files:
            continue

        csv_name = f"{tag}{fname}.csv"
        result_name = str(Path(csv_name).with_suffix(".zip")) if zip_result else csv_name
        result_path = path / result_name
        temporary_path = result_path.with_name(f".{result_path.name}.tmp")
        logger.info("Merging %s batch outputs into %s ...", len(out_files), result_name)

        file_columns, output_columns = _batch_columns(out_files, fname, monthly_subset)

        try:
            if zip_result:
                with (
                    zipfile.ZipFile(
                        temporary_path,
                        mode="w",
                        compression=zipfile.ZIP_DEFLATED,
                    ) as archive,
                    # The streamed CSV can exceed 2 GiB before compression.
                    archive.open(csv_name, mode="w", force_zip64=True) as binary_output,
                    io.TextIOWrapper(binary_output, encoding="utf-8", newline="") as csv_output,
                ):
                    _write_merged_batches(
                        csv_output,
                        out_files,
                        file_columns,
                        output_columns,
                        labels_df,
                        labels_join_field,
                    )
            else:
                with temporary_path.open("w", encoding="utf-8", newline="") as csv_output:
                    _write_merged_batches(
                        csv_output,
                        out_files,
                        file_columns,
                        output_columns,
                        labels_df,
                        labels_join_field,
                    )

            temporary_path.replace(result_path)
        except Exception as ex:
            logger.error("Error writing merged file: %s - %s", result_name, ex)
            temporary_path.unlink(missing_ok=True)
            raise


def delete_batch_outputs(path: Path, output_filenames: list[str]) -> None:
    """
    Deletes the individual batch results which are no longer required after they have been merged.

    Args:
        path (pathlib.Path): Directory path.
        output_filenames (list): List of filename patterns to delete.
    """
    deleted_count = 0
    error_count = 0
    logger.debug("Deleting batch outputs...")
    for fname in output_filenames:
        out_files = list(Path(path).glob(f"{fname}*.parquet"))
        for out_file in out_files:
            try:
                out_file.unlink()
                deleted_count += 1
            except Exception as ex:
                logger.error("Error deleting file: %s - %s", out_file, ex)
                error_count += 1
    if error_count > 0:
        logger.warning("Failed to delete %s batch outputs", error_count)
    if deleted_count > 0:
        logger.debug("Deleted %s batch outputs", deleted_count)


# ------------------------------------------------------------------------------
# Batch loader
# ------------------------------------------------------------------------------


def load_batch(
    csv_files: list[Path], chunk_id: int, config: WITMetricsConfig
) -> pd.DataFrame | None:
    """
    Loads, cleans, and combines a batch of CSV files.

    Performs the following:
    1. Filters data based on missing data threshold.
    2. Normalizes dates (removes time).
    3. Averages duplicate entries for the same feature and date.
    4. Sorts by feature and date.

    Args:
        csv_files (list): List of CSV file paths.
        chunk_id (int): ID for the current chunk/batch.

    Returns:
        pd.DataFrame: Processed DataFrame for the batch.
    """
    dfs = []
    pkey = config.wit_feature_id

    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            logger.warning("Skipping empty CSV file: %s", f)
            continue
        except Exception as e:
            logger.error("Error reading %s: %s", f, e)
            raise

        if df.empty:
            continue

        # 1. Filter missing data
        df = df[df["pc_missing"] < config.pc_missing_threshold]
        if df.empty:
            continue

        # 2. Robust Date Normalization
        # Use format='mixed' to handle both "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS"
        df["date"] = (
            pd.to_datetime(df["date"], utc=True, format="mixed").dt.tz_localize(None).dt.normalize()
        )

        dfs.append(df)

    if not dfs:
        logger.warning("Chunk %s: No valid data after loading %s files", chunk_id, len(csv_files))
        return None

    # 3. Combine all files
    wit_data = pd.concat(dfs, ignore_index=True)

    # 4. Collapse duplicates
    # Group by ID and Date to average observations from the same day
    wit_data = wit_data.groupby([pkey, "date"], as_index=False).mean(numeric_only=True)

    # 5. Final Sort
    wit_data = wit_data.sort_values([pkey, "date"]).reset_index(drop=True)
    wit_data["chunk"] = chunk_id

    return wit_data


def write_batch_parquet(batch_df: pd.DataFrame, batch_fname: Path) -> Path:
    """
    Write a batch DataFrame to a temp Parquet file.

    Args:
        batch_df (pd.DataFrame): DataFrame to write.
        batch_fname (str): Target filename.

    Returns:
        str: The filename written to.
    """
    tmp_out = Path(batch_fname).with_suffix(".tmp.parquet")

    try:
        # Write to temporary file
        batch_df.to_parquet(tmp_out, index=False)

        # Atomic rename
        tmp_out.replace(batch_fname)

        logger.info("  Written batch output:  %s", batch_fname)

    except Exception as e:
        logger.error("  Failed to write %s: %s", batch_fname, e)
        tmp_out.unlink(missing_ok=True)
        raise OSError(f"Parquet write failed for {batch_fname}") from e

    return batch_fname


# ------------------------------------------------------------------------------
# Batch processor (single process)
# ------------------------------------------------------------------------------


def process_batch(csv_files: list[Path], chunk_id: int, config: WITMetricsConfig) -> None:
    """
    Orchestrates the processing of a single batch of files.

    1. Loads data.
    2. Interpolates to daily (optional).
    3. Computes monthly and annual metrics.
    4. Computes inundation metrics.
    5. Computes time since last inundation.

    Args:
        csv_files (list): List of CSV files to process.
        chunk_id (int): ID for the batch.
    """
    # 1. Load, Normalize, and Sort
    # Handles daily averaging and strips time components immediately
    wit_data = load_batch(csv_files, chunk_id, config)

    if wit_data is None or wit_data.empty:
        return

    # 2. Optional daily interpolation
    # Fills gaps between observations using linear interpolation
    if config.interpolate_to_daily:
        wit_data = interpolate_daily(wit_data)

        if config.save_interpolated_csv:
            # Note: This can be slow for very large batches
            out_path = config.output_path / "csv_daily_interpolated"
            out_path.mkdir(parents=True, exist_ok=True)
            # Efficiently save grouped CSVs
            for fid, group in wit_data.groupby(config.wit_feature_id, sort=False):
                group.to_csv(out_path / f"{fid}.csv", index=False)

    # 3. Monthly & Annual Metrics
    # Uses the single-pass _resample_metrics for high speed
    monthly_metrics(wit_data, config.output_path)
    annual_metrics(wit_data, config.output_path)

    # 4. Inundation threshold (All-time median)
    # Calculates the threshold used for event detection
    median_df = adaptive_inundation_threshold(
        wit_data,
        config.output_path,
        threshold_percentile=config.threshold_percentile,
        min_threshold=config.min_threshold,
        max_threshold=config.max_threshold,
    )
    # Extract the specific column needed for _event_table
    threshold_df = pd.DataFrame(median_df["water+wet"])

    # 5. Inundation Metrics (The Core Analysis)
    # Vectorized event detection and area-weighted statistics
    wit_inundation_events_df = inundation_metrics(
        wit_data,
        threshold_df=threshold_df,
        shapefile_path=config.shapefile_path,
        shapefile_key=config.shapefile_key,
        output_path=config.output_path,
        debug_event_times=config.debug_event_times,
    )

    # 6. Time Since Last Inundation
    # Uses wit_data and the newly created wit_im to find the final dry gap
    time_since_last_inundation(wit_data, wit_inundation_events_df, config.output_path)


def main() -> None:
    config: WITMetricsConfig = load_config("wit_metrics")
    validate_config(config)
    setup_logging(log_level=config.log_level)

    output_filenames = [
        "WIT_yearly_metrics",
        "WIT_event_threshold",
        "WIT_inundation_metrics",
        "WIT_time_since_last_inundation",
        "WIT_event_times",
        "WIT_event_stats",
    ]
    if config.interpolate_to_daily:
        output_filenames.append("WIT_monthly_metrics")

    csv_list = list(config.wit_csv_path.glob("*.csv"))
    total_csv_files = len(csv_list)

    if total_csv_files == 0:
        RuntimeError(f"No CSV files found in {config.wit_csv_path}")
        return

    # Try to create output directories
    try:
        config.output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Cannot create OUTPUT_PATH {config.output_path}: {e}") from e

    # Cleanup any old batch outputs from prior runs
    delete_batch_outputs(config.output_path, output_filenames)

    # Conservative core usage (important for IO-bound workload reading many csv)
    cpu_count = max(1, multiprocessing.cpu_count() // 2)
    batch_size = config.batch_size
    chunk_size = batch_size * cpu_count

    logger.info("Processing with %s worker processes", cpu_count)
    logger.info("Found %s WIT CSV files", total_csv_files)
    logger.info("Batch size per process: %s", batch_size)

    start = time.process_time()

    # ------------------------------------------------------------------
    # Batch-level multiprocessing only
    # ------------------------------------------------------------------
    for j in progress(range(0, total_csv_files, chunk_size), desc="Processing chunks"):
        mpbatch = csv_list[j : j + chunk_size]

        work = []
        for i in range(0, len(mpbatch), batch_size):
            batch_files = mpbatch[i : i + batch_size]
            chunk_id = j + i
            work.append((batch_files, chunk_id, config))

        with multiprocessing.Pool(
            processes=min(len(work), cpu_count),
        ) as pool:
            pool.starmap(process_batch, work)

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.process_time() - start))
    logger.info("%s CSVs processed in %s", total_csv_files, elapsed)

    # ------------------------------------------------------------------
    # Merge batch outputs
    # ------------------------------------------------------------------
    merge_batches(
        config.output_path,
        output_filenames,
        monthly_subset=config.monthly_subset,
        tag=config.tag,
        zip_result=config.zip_result,
    )

    logger.info("All batches merged successfully.")

    # cleanup intermediate files
    delete_batch_outputs(config.output_path, output_filenames)


if __name__ == "__main__":
    main()
