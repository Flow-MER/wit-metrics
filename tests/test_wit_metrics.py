import zipfile
from datetime import date

import pandas as pd
import pytest

from config import WITMetricsConfig
from wit_metrics_worker import (
    adaptive_inundation_threshold,
    inundation_metrics,
    load_batch,
    merge_batches,
    progress,
    time_since_last_inundation,
)


@pytest.fixture
def wit_config(tmp_path):
    return WITMetricsConfig(
        wit_csv_path=tmp_path,
        output_path=tmp_path,
        log_path=tmp_path,
        shapefile_path=tmp_path / "unused.shp",
    )


@pytest.fixture
def make_wit_data():
    """Build daily observations with inundation split between water and wet vegetation."""

    def make(fractions):
        return pd.DataFrame(
            {
                "feature_id": ["A"] * len(fractions),
                "date": pd.date_range("2023-01-01", periods=len(fractions), freq="D"),
                "water": [fraction / 2 for fraction in fractions],
                "wet": [fraction / 2 for fraction in fractions],
                "chunk": [1] * len(fractions),
            }
        )

    return make


@pytest.fixture
def inundation_events(make_wit_data, wit_config):
    """Calculate events through the public workflow, without a polygon area lookup."""

    def calculate(fractions, threshold):
        return inundation_metrics(
            make_wit_data(fractions),
            threshold_df=threshold,
            shapefile_path=wit_config.shapefile_path,
            shapefile_key=wit_config.shapefile_key,
            output_path=wit_config.output_path,
        )

    return calculate


@pytest.fixture
def metric_batches(tmp_path):
    for number, feature_id in enumerate(["A", "B"]):
        pd.DataFrame({"feature_id": [feature_id], "value": [number]}).to_parquet(
            tmp_path / f"metric_{number}.parquet", index=False
        )
    return tmp_path


def test_load_batch_does_not_scroll_progress(tmp_path, wit_config, capsys):
    """Loading CSVs must not push terminal or notebook progress off screen."""
    csv_files = []
    for feature_id in ["A", "B", "C"]:
        csv_file = tmp_path / f"{feature_id}.csv"
        pd.DataFrame(
            {
                "feature_id": [feature_id],
                "date": ["2023-01-01"],
                "pc_missing": [0.0],
                "water+wet": [0.5],
            }
        ).to_csv(csv_file, index=False)
        csv_files.append(csv_file)

    result = load_batch(csv_files, 0, wit_config)

    assert result["feature_id"].tolist() == ["A", "B", "C"]
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


@pytest.mark.parametrize(
    "observation_date",
    ["2023-01-01T10:00:00Z", "2023-01-01", "2023-01-01 12:00:00"],
    ids=["iso-utc", "date-only", "space-separated"],
)
def test_load_batch_normalizes_dates(tmp_path, wit_config, observation_date):
    csv_path = tmp_path / "dates.csv"
    pd.DataFrame(
        {"feature_id": ["A"], "date": [observation_date], "water": [0.5], "pc_missing": [0]}
    ).to_csv(csv_path, index=False)

    result = load_batch([csv_path], chunk_id=1, config=wit_config)

    assert result["date"].tolist() == [pd.Timestamp("2023-01-01")]


def test_load_batch_filters_observations_at_or_above_missing_data_threshold(tmp_path, wit_config):
    csv_path = tmp_path / "missing.csv"
    pd.DataFrame(
        {
            "feature_id": ["A", "B", "C", "D"],
            "date": ["2023-01-01"] * 4,
            "water": [0.5] * 4,
            "pc_missing": [0.0, 0.099, 0.1, 0.95],
        }
    ).to_csv(csv_path, index=False)

    result = load_batch([csv_path], chunk_id=1, config=wit_config)

    assert result["feature_id"].tolist() == ["A", "B"]


def test_load_batch_averages_duplicates_per_feature_and_day(tmp_path, wit_config):
    csv_path = tmp_path / "duplicates.csv"
    pd.DataFrame(
        {
            "feature_id": ["A", "A", "A", "B"],
            "date": ["2023-01-01T10:00:00Z", "2023-01-01", "2023-01-02", "2023-01-01"],
            "water": [0.2, 0.4, 0.5, 0.6],
            "wet": [0.1, 0.1, 0.2, 0.3],
            "pc_missing": [0] * 4,
        }
    ).to_csv(csv_path, index=False)

    result = load_batch([csv_path], chunk_id=1, config=wit_config)

    expected = pd.DataFrame(
        {
            "feature_id": ["A", "A", "B"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-01"]),
            "water": [0.3, 0.5, 0.6],
            "wet": [0.1, 0.2, 0.3],
            "chunk": [1, 1, 1],
        }
    )
    pd.testing.assert_frame_equal(result[expected.columns], expected)


@pytest.mark.parametrize(
    "content",
    ["", "feature_id,date,water,wet,pc_missing\n"],
    ids=["empty-file", "header-only"],
)
def test_load_batch_returns_none_without_observations(tmp_path, wit_config, content):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(content, encoding="utf-8")

    result = load_batch([csv_path], chunk_id=1, config=wit_config)

    assert result is None


def test_inundation_metrics_attaches_gaps_to_preceding_events(inundation_events):
    # Both dry (0.04) and partially wet (0.20) observations are gaps at a 0.5 threshold.
    result = inundation_events([0.04, 0.20, 0.8, 0.8, 0.8, 0.20, 0.7, 0.20, 0.04, 0.04], 0.5)

    # The leading gap plus both events and their following gaps account for all ten days.
    expected = pd.DataFrame(
        {
            "start_date": [pd.NaT, date(2023, 1, 3), date(2023, 1, 7)],
            "end_date": [pd.NaT, date(2023, 1, 5), date(2023, 1, 7)],
            "duration": [0, 3, 1],
            "gap": [2, 1, 3],
        }
    )
    pd.testing.assert_frame_equal(result[expected.columns], expected)


def test_inundation_metrics_represents_never_wet_record_as_one_gap(inundation_events):
    result = inundation_events([0.1] * 100, 0.5)

    expected = pd.DataFrame(
        {"feature_id": ["A"], "duration": [0], "gap": [100], "area_days": [0.0]}
    )
    pd.testing.assert_frame_equal(result[expected.columns], expected)
    assert result[["start_date", "end_date"]].isna().all().all()


@pytest.mark.parametrize("threshold", [0.05, 0.3], ids=["floor", "variable"])
def test_inundation_metrics_requires_strictly_greater_than_threshold(inundation_events, threshold):
    result = inundation_events([threshold, threshold + 0.01], threshold)

    expected = pd.DataFrame(
        {
            "start_date": [pd.NaT, date(2023, 1, 2)],
            "end_date": [pd.NaT, date(2023, 1, 2)],
            "duration": [0, 1],
            "gap": [1, 0],
        }
    )
    pd.testing.assert_frame_equal(result[expected.columns], expected)


def test_inundation_metrics_sums_area_days_only_during_events(inundation_events):
    result = inundation_events([0.1, 0.8, 0.9, 0.4, 0.1], 0.3)

    # Only days 2-4 contribute: 0.8 + 0.9 + 0.4 = 2.1 fractional area-days.
    assert result["area_days"].tolist() == pytest.approx([0.0, 2.1])


def test_time_since_last_inundation_uses_final_event_gap(tmp_path):
    events = pd.DataFrame(
        {
            "feature_id": ["A", "A"],
            "start_date": pd.to_datetime(["1986-10-27", "1988-06-23"]),
            "end_date": pd.to_datetime(["1987-11-17", "1992-09-24"]),
            "duration": [387, 1555],
            "gap": [218, 91],
        }
    )
    wit_data = pd.DataFrame({"feature_id": ["A"], "chunk": [1]})

    result = time_since_last_inundation(wit_data, events, output_path=tmp_path)

    assert result.to_dict("records") == [{"feature_id": "A", "timesincelast": 91}]


@pytest.mark.parametrize(
    ("fractions", "expected"),
    [
        pytest.param([0.0] * 9 + [0.8], 0.05, id="floor"),
        # P30 interpolates 70% of the way from 0.11 to 0.15: 0.138.
        pytest.param([0.0, 0.0, 0.11, 0.15, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95], 0.138, id="p30"),
        pytest.param(
            [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98], 0.5, id="ceiling"
        ),
    ],
)
def test_adaptive_inundation_threshold_p30(tmp_path, make_wit_data, fractions, expected):
    result = adaptive_inundation_threshold(
        make_wit_data(fractions), output_path=tmp_path, min_threshold=0.05, max_threshold=0.5
    )

    assert result.loc["A", "water+wet"] == pytest.approx(expected)


def test_merge_batches_does_not_render_progress_when_output_is_captured(metric_batches, capsys):
    """Captured logs should not contain tqdm's carriage returns or clearing spaces."""
    merge_batches(metric_batches, ["metric"], zip_result=False)

    assert capsys.readouterr().err == ""


def test_merge_batches_writes_all_batches_to_one_csv(metric_batches):
    merge_batches(metric_batches, ["metric"])

    result = pd.read_csv(metric_batches / "RESULT_metric.csv")
    assert result.to_dict("records") == [
        {"feature_id": "A", "value": 0},
        {"feature_id": "B", "value": 1},
    ]


def test_merge_batches_streams_filtered_labeled_csv_into_zip(tmp_path):
    pd.DataFrame(
        {
            "feature_id": ["A"],
            "value": [1.23456],
            "optional": [9.87654],
            "discarded": [100],
        }
    ).to_parquet(tmp_path / "monthly_metric_1.parquet")
    pd.DataFrame({"feature_id": ["B"], "value": [2.34567], "discarded": [200]}).to_parquet(
        tmp_path / "monthly_metric_2.parquet"
    )

    merge_batches(
        tmp_path,
        ["monthly_metric"],
        monthly_subset=["feature_id", "value", "optional"],
        labels_df=pd.DataFrame({"feature_id": ["A", "B"], "label": ["alpha", "beta"]}),
        zip_result=True,
    )

    with zipfile.ZipFile(tmp_path / "RESULT_monthly_metric.zip") as archive:
        assert archive.namelist() == ["RESULT_monthly_metric.csv"]
        csv_text = archive.read("RESULT_monthly_metric.csv").decode("utf-8")

    assert csv_text.splitlines() == [
        "feature_id,value,optional,label",
        "A,1.2346,9.8765,alpha",
        "B,2.3457,,beta",
    ]


def test_merge_zip_supports_large_uncompressed_csv(tmp_path, monkeypatch):
    # Exercise the ZIP64 boundary with kilobytes instead of gigabytes.
    size_limit = 1024
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", size_limit)
    batches = [
        pd.DataFrame({"feature_id": [uid * 64] * 100, "year": year, "water": 0.25})
        for uid, year in [("A", 2020), ("B", 2021)]
    ]
    for number, batch in enumerate(batches):
        batch.to_parquet(tmp_path / f"WIT_yearly_metrics{number}.parquet", index=False)

    merge_batches(tmp_path, ["WIT_yearly_metrics"], tag="MDB", zip_result=True)

    with zipfile.ZipFile(tmp_path / "MDB_WIT_yearly_metrics.zip") as archive:
        assert archive.namelist() == ["MDB_WIT_yearly_metrics.csv"]
        member = archive.getinfo("MDB_WIT_yearly_metrics.csv")
        assert member.file_size > size_limit
        assert member.compress_size < size_limit
        with archive.open(member) as csv_input:
            result = pd.read_csv(csv_input)
    pd.testing.assert_frame_equal(result, pd.concat(batches, ignore_index=True))


def test_merge_batches_preserves_existing_result_when_a_later_batch_fails(
    metric_batches, monkeypatch
):
    result_path = metric_batches / "RESULT_metric.csv"
    result_path.write_text("existing result\n", encoding="utf-8")
    read_parquet = pd.read_parquet

    def fail_second_batch(path, *args, **kwargs):
        if path.name == "metric_1.parquet":
            raise OSError("simulated read failure")
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", fail_second_batch)

    with pytest.raises(OSError, match="simulated read failure"):
        merge_batches(metric_batches, ["metric"])

    assert result_path.read_text(encoding="utf-8") == "existing result\n"
    assert not (metric_batches / ".RESULT_metric.csv.tmp").exists()


def test_progress_renders_when_running_in_a_notebook(monkeypatch, capsys):
    """A Jupyter kernel should retain visible progress even though it is not a TTY."""
    import IPython

    shell_type = type("ZMQInteractiveShell", (), {})
    monkeypatch.setattr(IPython, "get_ipython", lambda: shell_type())

    list(progress([1], desc="Notebook progress"))

    assert "Notebook progress" in capsys.readouterr().err
