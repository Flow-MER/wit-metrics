import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure we can import from the parent directory
sys.path.append(str(Path(__file__).parents[1]))
from config import WITMetricsConfig

@pytest.fixture
def complex_wit_data():
    """15-day sequence with two distinct events and irregular dates."""
    data = {
        "date": pd.to_datetime([
            "2023-01-01", "2023-01-03", # Event 1
            "2023-01-05",                # Gap
            "2023-01-06", "2023-01-08", # Event 2
            "2023-01-11", "2023-01-12", # Event 2 continues
            "2023-01-15"                 # Trailing Gap
        ]),
        "feature_id": ["A"] * 8,
        "water+wet": [0.6, 0.6, 0.1, 0.6, 0.6, 0.6, 0.6, 0.1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def wit_config(tmp_path):
    return WITMetricsConfig(
        wit_csv_path=tmp_path,
        output_dir=tmp_path,
        log_dir=tmp_path,
        shapefile_path=None,
    )


## --- 1. Data Loading (load_batch) ---


def test_load_batch_robustness(tmp_path, wit_config):
    """Verifies mixed date formats, filtering, and duplicate averaging."""
    from wit_metrics_worker import load_batch
    csv_path = tmp_path / "robust_batch.csv"

    pd.DataFrame({
        "feature_id": ["A", "A", "A", "B"],
        # MIXED FORMATS: Full ISO, simplified date, and space-separated
        "date": [
            "2023-01-01T10:00:00Z", 
            "2023-01-01", 
            "2023-01-02 12:00:00", 
            "2023-01-01"
        ],
        "water": [0.2, 0.4, 0.5, 0.6], 
        "wet": [0.1, 0.1, 0.1, 0.2],
        "pc_missing": [0, 0, 0.95, 0] # Index 2 filtered
    }).to_csv(csv_path, index=False)

    df = load_batch([str(csv_path)], chunk_id=1, config=wit_config)

    # Feature A: 2023-01-01 rows average to 0.3. 2023-01-02 is filtered.
    assert len(df) == 2
    assert df[df["feature_id"] == "A"]["water"].iloc[0] == pytest.approx(0.3)
    assert df[df["feature_id"] == "B"]["water"].iloc[0] == 0.6


def test_load_batch_empty_scenarios(tmp_path, wit_config):
    from wit_metrics_worker import load_batch
    for content in ["", "feature_id,date,water,wet,pc_missing\n"]:
        csv_path = tmp_path / "empty_test.csv"
        csv_path.write_text(content)
        result = load_batch([str(csv_path)], chunk_id=1, config=wit_config)
        assert result is None or len(result) == 0


## --- 2. Threshold Logic (_event_table) ---

def test_event_table_gap_after_logic():
    """
    Verifies:
    1. Leading gap is recorded first.
    2. Gaps follow the event they belong to.
    3. Golden Rule (Total Span) is preserved.
    """
    from wit_metrics_worker import _event_table
    
    # 10 day sequence: 2 days dry, 3 days wet, 5 days dry
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    data = {
        "date": dates,
        "water+wet": [0.1, 0.1, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1]
    }
    df = pd.DataFrame(data)
    
    ev = _event_table(df, threshold=0.5)
    
    # Golden Rule: 10 days total
    assert ev["duration"].sum() + ev["gap"].sum() == 10
    assert len(ev) == 2  # One leading gap, one event with trailing gap
    
    # Row 0: Leading Gap (Jan 1 - Jan 2)
    assert ev.iloc[0]["duration"] == 0
    assert ev.iloc[0]["gap"] == 2
    assert pd.isna(ev.iloc[0]["start_date"])
    
    # Row 1: Event + Trailing Gap (Jan 3 - Jan 5 = 3 days; Jan 6 - Jan 10 = 5 days)
    assert ev.iloc[1]["duration"] == 3
    assert ev.iloc[1]["gap"] == 5
    assert ev.iloc[1]["start_date"] == pd.Timestamp("2023-01-03")

## --- 3. Metrics Workflow (inundation_metrics) ---


def test_inundation_with_leading_gap(complex_wit_data):
    from wit_metrics_worker import _event_table
    import pandas as pd

    df = complex_wit_data.copy()
    # Force the dates to be a clean daily sequence
    df["date"] = pd.date_range("2023-01-01", periods=len(df), freq="D")

    # Jan 1 & 2 (index 0, 1) = Dry (0.1)
    df.loc[0:1, "water+wet"] = 0.1

    # Jan 3 (index 2) = Wet (0.7)
    # Must be > 0.6 to trigger an event with the new logic!
    df.loc[2, "water+wet"] = 0.7

    ev = _event_table(df, threshold=0.6)

    # Now ev will have:
    # Row 0: Leading Gap (Jan 1-2)
    # Row 1: Event (Jan 3) + Trailing Gap (Jan 4-end)

    assert len(ev) == 2
    assert ev.iloc[0]["gap"] == 2
    assert ev.iloc[1]["start_date"] == pd.Timestamp("2023-01-03")
    assert ev.iloc[1]["duration"] == 1


## --- 4. Other Tools ---


def test_time_since_last_inundation_simple(tmp_path):
    """
    Verifies that we simply pull the 'gap' from the last row.
    """
    from wit_metrics_worker import time_since_last_inundation

    # Simulate a wit_im table where the last event has a 91 day trailing gap
    wit_im = pd.DataFrame({
        "feature_id": ["A", "A"],
        "start_date": [pd.Timestamp("1986-10-27"), pd.Timestamp("1988-06-23")],
        "end_date": [pd.Timestamp("1987-11-17"), pd.Timestamp("1992-09-24")],
        "duration": [387, 1555],
        "gap": [218, 91] # The 91 is the 'time since last'
    })

    # Dummy wit_data just for chunk ID
    wit_data = pd.DataFrame({"feature_id": ["A"], "chunk": [1]})

    result = time_since_last_inundation(wit_data, wit_im, output_dir=tmp_path)

    assert result[result["feature_id"] == "A"]["timesincelast"].iloc[0] == 91

    ## --- 3. The Never-Wet Case ---


def test_event_table_never_wet():
    from wit_metrics_worker import _event_table
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({"date": dates, "water+wet": [0.1]*100})
    
    ev = _event_table(df, threshold=0.5)
    
    assert len(ev) == 1
    assert ev.iloc[0]["duration"] == 0
    assert ev.iloc[0]["gap"] == 100

def test_threshold_floor_and_strictly_greater():
    from wit_metrics_worker import _event_table
    import pandas as pd

    # Threshold is 0.05 (due to floor)
    # Day 1: 0.05 (Should be GAP because 0.05 is not > 0.05)
    # Day 2: 0.06 (Should be EVENT)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=2, freq="D"),
        "water+wet": [0.05, 0.06]
    })

    ev = _event_table(df, threshold=0.05)

    # Row 0 should be a leading gap of 1 day (for the 0.05 value)
    assert ev.iloc[0]["duration"] == 0
    assert ev.iloc[0]["gap"] == 1

    # Row 1 should be the event for 0.06
    assert ev.iloc[1]["duration"] == 1


def test_adaptive_inundation_threshold_p30(tmp_path):
    from wit_metrics_worker import adaptive_inundation_threshold
    import pandas as pd
    import numpy as np

    # A: Ephemeral (90% dry) -> P30=0, Median=0
    vals_A = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]

    # B: Variable (Similar to r1wjpxg6j: 40% dry, 60% wet)
    # P30 will be the 3rd value (0.0), Median will be 0.6
    # NOTE: With 10 samples, P30 is the 3rd sorted value.
    # To simulate your 0.106 result, we provide some low-level water.
    vals_B = [0.0, 0.0, 0.11, 0.15, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

    # C: Perennial (Always > 80%) -> P30=0.82, Median=0.90
    vals_C = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]

    data = {
        "feature_id": ["A"]*10 + ["B"]*10 + ["C"]*10,
        "water": vals_A + vals_B + vals_C,
        "wet": [0.0] * 30,
        "chunk": [1] * 30
    }
    df = pd.DataFrame(data)

    # Run logic with P30 and the 0.05/0.5 bounds
    result = adaptive_inundation_threshold(
        df, output_dir=tmp_path, min_threshold=0.05, max_threshold=0.5
    )

    # --- VERIFICATIONS ---

    # Feature A: min(Med=0, P30=0) = 0 -> Clips to Floor 0.05
    assert result.loc["A", "water+wet"] == 0.05

    # Feature B: min(Med=0.65, P30=0.11) = 0.11
    # This reflects your r1wjpxg6j result (0.1062)
    assert round(result.loc["B", "water+wet"], 2) == 0.14

    # Feature C: min(Med=0.89, P30=0.82) = 0.82 -> Clips to Ceiling 0.50
    assert result.loc["C", "water+wet"] == 0.50

    print("P30 Test Passed: Floor, Variable Baseline, and Ceiling verified.")


def test_event_table_three_state_model():
    """
    Verifies the Wet-Normal-Dry model:
    1. Dry (0.04) and Normal (0.2) both count as Gaps.
    2. Wet (0.6) triggers Duration.
    3. The threshold (0.3) correctly separates Normal from Wet.
    """
    from wit_metrics_worker import _event_table
    import pandas as pd
    
    # 10 day sequence
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    data = {
        "date": dates,
        "water+wet": [
            0.04, 0.04, # Days 1-2: DRY
            0.20, 0.20, # Days 3-4: NORMAL
            0.60, 0.60, # Days 5-6: WET
            0.20, 0.20, # Days 7-8: NORMAL
            0.04, 0.04  # Days 9-10: DRY
        ]
    }
    df = pd.DataFrame(data)
    
    # Threshold set at 0.3 (The Normal -> Wet transition)
    threshold = 0.3
    ev = _event_table(df, threshold=threshold)
    
    # Golden Rule: 10 days total
    assert ev["duration"].sum() + ev["gap"].sum() == 10
    
    # Expected Structure:
    # Row 0: Leading Gap (Dry + Normal) = 4 days
    # Row 1: Wet Event (2 days) + Trailing Gap (Normal + Dry) = 4 days
    
    assert len(ev) == 2
    
    # Row 0: Leading Gap includes the transition from Dry to Normal
    assert ev.iloc[0]["gap"] == 4
    assert ev.iloc[0]["duration"] == 0
    
    # Row 1: The 'Wet' event
    assert ev.iloc[1]["start_date"] == pd.Timestamp("2023-01-05")
    assert ev.iloc[1]["duration"] == 2
    assert ev.iloc[1]["gap"] == 4


def test_area_day_calculation():
    """
    Verifies:
    1. Area-Days are only summed for the 'Wet' duration.
    2. The calculation correctly handles varying inundation levels.
    """
    from wit_metrics_worker import _event_table
    import pandas as pd
    
    # 5-day sequence
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    data = {
        "date": dates,
        "water+wet": [
            0.10, # Day 1: NORMAL/GAP (Threshold 0.3)
            0.80, # Day 2: WET (Event Start)
            0.90, # Day 3: WET
            0.40, # Day 4: WET (Still above 0.3)
            0.10  # Day 5: NORMAL/GAP (Event End)
        ]
    }
    df = pd.DataFrame(data)
    threshold = 0.3
    
    # Generate event table
    ev = _event_table(df, threshold=threshold)
    
    # Manual Calculation for validation:
    # Event is Days 2, 3, 4 (Duration = 3)
    # Area-Days = 0.80 + 0.90 + 0.40 = 2.10
    
    # Note: Depending on your code, you might store this in 
    # a column named 'area_days' or 'total_inundated_area'
    event_row = ev.iloc[1]
    assert event_row["duration"] == 3
    assert round(event_row["area_days"], 2) == 2.10
