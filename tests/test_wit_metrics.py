import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import your functions here
# from wit_metrics_worker import _event_table, load_batch, _resample_metrics

## --- Fixtures ---

@pytest.fixture
def sample_wit_data():
    """Consistent 10-day sequence."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    data = {
        "feature_id": ["A"] * 10,
        "date": dates,
        # Event 1: Indices 0,1,2 (0.6) | Event 2: Indices 5,6 (0.6)
        "water": [0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        "wet":   [0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0],
    }
    df = pd.DataFrame(data)
    df["water+wet"] = df["water"] + df["wet"]
    return df

def test_event_table_detection(sample_wit_data):
    """Test that vectorized edge detection finds the correct start/end dates."""
    from wit_metrics_worker import _event_table
    threshold = 0.3
    
    ev = _event_table(sample_wit_data, threshold)
    
    # Check number of events
    assert len(ev) == 2
    
    # Check first event (Jan 1 to Jan 3)
    assert ev.iloc[0]["duration"] == 3
    assert ev.iloc[0]["start_date"] == pd.Timestamp("2023-01-01")
    assert ev.iloc[0]["end_date"] == pd.Timestamp("2023-01-03")
    
    # Check second event (Jan 6 to Jan 7)
    assert ev.iloc[1]["duration"] == 2
    
    # Check Gap (End of Event 1 is Jan 3, Start of Event 2 is Jan 6)
    # Gap days are Jan 4 and Jan 5. 
    # Calculation: (Jan 6 - Jan 3 - 1 day) = 2 days
    assert ev.iloc[1]["gap"] == 2

def test_load_batch_sanitization(tmp_path):
    from wit_metrics_worker import load_batch
    csv_path = tmp_path / "test_data.csv"
    pd.DataFrame({
        "feature_id": ["A", "A"],
        "date": ["2023-01-01 10:00:00", "2023-01-01 14:00:00"],
        "water": [0.2, 0.4],
        "wet": [0.1, 0.1],
        "pc_missing": [0, 0]
    }).to_csv(csv_path, index=False)
    
    df = load_batch([str(csv_path)], chunk_id=1)
    
    assert len(df) == 1
    # FIX: Use approx to handle float precision
    assert df["water"].iloc[0] == pytest.approx(0.3)

def test_inundation_stats_slicing(sample_wit_data):
    g_series = sample_wit_data.set_index("date")["water+wet"]
    start = pd.Timestamp("2023-01-05")
    end = pd.Timestamp("2023-01-07")
    
    sl = g_series.loc[start:end]
    
    # Values at index 5 and 6 are 0.6. Index 4 is 0.0.
    # Slicing Jan 5 to Jan 7 picks up indices 4, 5, 6
    assert len(sl) == 3 
    # FIX: Match the actual data (0.6)
    assert sl.max() == pytest.approx(0.6) 

def test_resample_metrics_logic(sample_wit_data):
    from wit_metrics_worker import _resample_metrics
    df = sample_wit_data.set_index("date")
    
    res = _resample_metrics(df, "10D", "feature_id")
    
    # Verify the flattened named aggregation columns
    assert "water_max" in res.columns
    assert "water_mean" in res.columns
    assert "water+wet_max" in res.columns
    assert res["count"].dtype == "int32"
    
    # Check for the Double Row Index cleanup (should be columns now)
    assert "feature_id" in res.columns
    assert "date" in res.columns
    


# Assuming _event_table is imported

def test_event_table_edge_cases():
    from wit_metrics_worker import _event_table
    # Create test dates
    start = datetime(2023, 1, 1)
    dates = pd.date_range(start, periods=10, freq="D")

    # Test cases
    cases = [
        # No inundation
        {
            "name": "no_inundation",
            "wet": np.zeros(10),
            "threshold": 0.5,
            "expected": 0
        },
        # All inundated
        {
            "name": "all_inundated",
            "wet": np.ones(10),
            "threshold": 0.5,
            "expected": 1,
            "duration": [10]
        },
        # Single-day events at start and end
        {
            "name": "start_end_events",
            "wet": np.array([1,0,0,0,0,0,0,0,0,1]),
            "threshold": 0.5,
            "expected": 2,
            "duration": [1,1],
            "gap": [0,8]
        },
        # Consecutive multi-day events
        {
            "name": "two_events",
            "wet": np.array([0,1,1,0,0,1,1,1,0,0]),
            "threshold": 0.5,
            "expected": 2,
            "duration": [2,3],
            "gap": [0,2]
        },
        # Single-day event in middle
        {
            "name": "single_middle",
            "wet": np.array([0,0,1,0,0,0,0,0,0,0]),
            "threshold": 0.5,
            "expected": 1,
            "duration": [1],
            "gap": [0]
        }
    ]

    for case in cases:
        df = pd.DataFrame({"date": dates, "water+wet": case["wet"]})
        res = _event_table(df, case["threshold"])
        
        assert len(res) == case["expected"], f"{case['name']} failed: wrong number of events"
        
        if "duration" in case:
            assert res["duration"].tolist() == case["duration"], f"{case['name']} failed: wrong durations"
        if "gap" in case:
            assert res["gap"].tolist() == case["gap"], f"{case['name']} failed: wrong gaps"
