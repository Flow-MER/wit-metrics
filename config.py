"""config.py-----------------------------------------------------------------
Config parameters used by wit_metric_worker.py
-----------------------------------------------------------------
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import logging


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class WITMetricsConfig:

    output_path: Path = Path("output")
    log_path: Path = Path("log")

    # shapefile: the shape file mentioned above to find the  and get their area
    # set to None to disable area lookup
    shapefile_path: Optional[Path] = Path("input/shp/ANAEv3_test.shp")

    # shapefile field name that identifies each polygon -  the ANAEv3 UID geohash was used here.
    # The ANAE UID is also used in the naming convention for the CSV files
    shapefile_key: str = "UID"

    # Path to folder that contains the WIT csv files to process.  When debugging providing a single file might be prudent.
    wit_csv_path: Path = Path("input/csv")

    # Only use WIT data where the pc_missing is less than the threshold (default is 0.1) i.e. >90% of the polygon was visible to satellites
    pc_missing_threshold: float = 0.1

    # csv feature_id - the WIT csv output files include a column 'feature_id' that in this case is the ANAE UID
    wit_feature_id: str = "feature_id"

    # whether to interpolate the WIT observation dates (typically 10-50+ per year) to daily data (365 per year)
    # This is computationally expensive but improves estimates of inundation duration and time since last inundation.
    # Monthly WIT stats require interpolated daily data to infill missing records and will not be generated if interpolate_to_daily = False
    interpolate_to_daily: bool = True

    # set to True to save the interpolated daily WIT csv in a subfolder under the csv_files (unnecessary and take up a lot of space but good for debugging)
    save_interpolated_csv: bool = False

    # monthly metrics files are too big for most computers when all metrics are used (e.g. just 4 metrics x 270,000 polygons x 450 months is a 5GB csv file)
    # specify a subset that will be joined together into the monthly result - must include ["feature_id","date", at-least-one-metric]
    # set to None or comment out all fields to save all metrics

    monthly_subset: Optional[List[str]] = field(
        default_factory=lambda: [
            "feature_id",
            "date",
            "water_median",
            "wet_median",
            "pv_median",
            "npv_median",
            "bs_median",
            "count",
        ]
    )

    # set to true to save intermediate data frames containing the event times and stats
    # these are saved in the working directory
    debug_event_times: bool = False

    # batchsize is the number of WIT csv files to include in each 'batch' that is processed by each single CPU core.
    # The code was designed to process several 100,000 polygons in small batches that fit into the computer memory
    # then glue all the batch results together at the end.
    # On a workstation with 16 cpu cores and 64MB RAM a batchsize of 100-200 worked well. With smaller number of CSV a batch size < total number of CSV allows the
    # calculations to be spread across multiple processors.
    # during processing the code will generate outputs for each batch then glue them together at the end.

    batch_size: int = 100

    # tag prepended to final result files (zipped csv)
    tag: str = "RESULT"

    # Whether to zip the final result csv to save space (python/pandas can read the csv from the zips)
    zip_result: bool = True

    # define threshold and bounds to define inundation events
    threshold_percentile: float = (
        0.3  # 0.3 = 30th percentile of 'water+wet' area (lower than 0.5 keeps the median in the inundated state)
    )
    min_threshold: float = (
        0.05  # Floor for dry sites. (0.05=5% area) stops noise and minimal water detection inclusion as inundation event
    )
    max_threshold: float = (
        0.50  # Cap for very wet sites. permanent lake is still considered inundated until falls below 50% water by area
    )
    
    # -----------------------------------------------------------------
    
    project_root: Path = field(
        default=Path(__file__).resolve().parent, init=False
    )
    def __post_init__(self):
        object.__setattr__(self, "output_path", self.project_root / self.output_path)
        object.__setattr__(self, "shapefile_path", self.project_root / self.shapefile_path)
        object.__setattr__(self, "log_path", self.project_root / self.log_path)


# named config simplifies reuse in other projects
CONFIGS = {"wit_metrics": WITMetricsConfig}

def load_config(name: str):
    if name not in CONFIGS:
        valid_configs = ", ".join(CONFIGS.keys())
        raise ValueError(f"Unknown config: {name}. Valid configs are: {valid_configs}")
    else:
        return CONFIGS[name]()
