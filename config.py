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

    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    log_dir: Path = Path("log")

    # shapefile: the shape file mentioned above to find the  and get their area
    # set to None to disable area lookup
    shapefile_path: Optional[Path] = Path("input/shp/ANAEv3_test.shp")

    # shapefile field name that identifies each polygon -  the ANAEv3 UID geohash was used here.
    # The ANAE UID is also used in the naming convention for the CSV files
    shapefile_key: str = "UID"

    # Path to folder that contains the WIT csv files to process.  When debugging providing a single file might be prudent.
    wit_csv_path: Path = Path("input/csv")
    wit_csv_path: Path = Path("m:\\ANAE_MDB_WIT_Feb_2026")

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
    # specify a subset that will be joined together into the monthly result - must include ["feature_id","date", at-least-one-metric]\
    monthly_subset: Optional[List[str]] = None  # do not prune
    # monthly_subset=[
    #         "feature_id",
    #         "date",
    #         "water_median",
    #         "wet_median",
    #         "pv_median",
    #         "npv_median",
    #         "bs_median",
    #         "count",
    #     ]

    # set to true to save intermediate data frames containing the event times and stats
    # these are saved in the working directory
    debug_event_times: bool = True

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
    zip_result: bool = False

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
    
   
    def __post_init__(self):
        """
        Validates configuration before processing starts.
        Catches issues early rather than failing hours into a batch job.
        """
        errors = []
        warnings = []

        # 1. Check critical paths exist
        # Ensure paths are Path objects
        object.__setattr__(self, "wit_csv_path", Path(self.wit_csv_path))
        object.__setattr__(self, "shapefile_path", 
                   Path(self.shapefile_path) if self.shapefile_path else None)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "log_dir", Path(self.log_dir)) 
        

        if not self.wit_csv_path.is_dir():
            errors.append(f"wit_csv_path is not a directory: {self.wit_csv_path}")

        # Check if shapefile exists (only if area lookup is enabled)
        if self.shapefile_path is not None:
            if not self.shapefile_path.exists():
                errors.append(f"shapefile_path does not exist: {self.shapefile_path}")
            if not self.shapefile_path.suffix == ".shp":
                warnings.append(
                    f"shapefile_path doesn't have .shp extension: {self.shapefile_path}"
                )

        # Check value ranges
        if not 0 <= self.pc_missing_threshold <= 1:
            errors.append(
                f"pc_missing_threshold must be between 0 and 1, got: {self.pc_missing_threshold}"
            )

        if self.batch_size < 1:
            errors.append(f"BATCH_SIZE must be >= 1, got: {self.batch_size}")

        if self.threshold_percentile < 0 or self.threshold_percentile > 1:
            errors.append(
                f"threshold_percentile must be between 0 and 1 (0.3 recommended), got: {self.threshold_percentile}"
            )

        if self.min_threshold < 0 or self.min_threshold > 1:
            errors.append(
                f"min_threshold must be between 0 and 1 (0.05 recommended), got: {self.min_threshold}"
            )

        if self.max_threshold < 0 or self.max_threshold > 1:
            errors.append(
                f"max_threshold must be between 0 and 1 (0.5 recommended), got: {self.max_threshold}"
            )

        # Check MONTHLY_SUBSET validity
        if self.monthly_subset is not None:
            required_cols = ["feature_id", "date"]
            missing = [col for col in required_cols if col not in self.monthly_subset]
            if missing:
                errors.append(
                    f"monthly_subset must include {required_cols}, missing: {missing}"
                )

            # Should have at least one metric column
            if len(self.monthly_subset) <= 2:
                warnings.append(
                    "monthly_subset only has feature_id and date - no metrics selected"
                )

        
        # Report results
        if warnings:
            for w in warnings:
                logger.warning(f"Config warning: {w}")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            raise ValueError(error_msg)

        logger.info("Configuration validated successfully")

        # Log key settings
        logger.debug(f"Processing settings:")
        logger.debug(f"  Input: {self.wit_csv_path}")
        logger.debug(f"  Output: {self.output_dir}")
        logger.debug(f"  Interpolate to daily: {self.interpolate_to_daily}")
        logger.debug(f"  Batch size: {self.batch_size}")
        logger.debug(f"  Missing threshold: {self.pc_missing_threshold}")
        logger.debug(f"  Threshold percentile: {self.threshold_percentile}")
        logger.debug(f"  Min threshold: {self.min_threshold}")
        logger.debug(f"  Max threshold: {self.max_threshold}")
        logger.debug(f"  Monthly subset: {self.monthly_subset}")
        logger.debug(f"  Debug event times: {self.debug_event_times}")
        logger.debug(f"  Shapefile path: {self.shapefile_path}")
        logger.debug(f"  Shapefile key: {self.shapefile_key}")
        logger.debug(f"  Zip result: {self.zip_result}")


# ###########################################################
    @staticmethod
    def load_config() -> "WITMetricsConfig":
        return WITMetricsConfig()
