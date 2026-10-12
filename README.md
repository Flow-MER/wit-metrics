# WIT-metrics

Calculate inundation summary metrics from Geoscience Australia Wetland Insights Tool (WIT) data (csv files)

Geoscience Australia's [Wetland Insights Tool (WIT)](https://knowledge.dea.ga.gov.au/data/product/dea-wetlands-insight-tool-ramsar-wetlands/?tab=details) csv timeseries outputs for all 270,653 polygons in the [ANAEv3 mapping of the MDB](https://data.flow-mer.org.au/dataset/anae-wetlands-and-floodplains) larger than 1 hectare.  Small polygons <1 hectare (approx 9 Landsat pixels) were excluded as cover estimates from Landsat are likely to be unreliable when the majority of incident 30m x 30m pixels overlap the polygon boundary.  The outputs of the ANAE WIT are 270,653 csv formatted time series estimating the proportion of wetland surface area on each satellite pass date covered by:

1. bare ground {bs)
2. dry non-photosynthetic vegetation (npv)
3. green photosynthetic vegetation (pv)
4. water under vegetation (wet)
5. open water (water)
6. missing data - obscured by clouds (pc_missing)

Each WIT csv file is named using the ANAE UID (a 9 character geohash of the polygon centroid).

**these data were last updated Feb 2026 to include the period 1986-2025 calendar years to the end of the previous water year (June to July) to support Flow-MER evaluation.  [Contact the data manager](mailto:data@flow-mer.org.au) for more information.  We are expecting the next update in July 2026

The raw time series are then used to generate a suite of hydrological metrics for each wetland polygon that includes various combinations of the WIT metrics e.g:

* water+wet = the total inundated area of the wetland
* npv+pv+wet =  the total vegetation cover of the wetland

The summary metrics are generated using only the observation dates when 90% of more of the wetland was clearly visible to the satellite.  The [WIT-metrics python notebook](https://github.com/Flow-MER/wit-metrics) on the Flow-MER Github Code Repository was used to calculate:

* annual summary (max, min, mean, median inundation and vegetation fractional cover per year)
* monthly summary (limited to inundation metrics because is a very large data set)
* threshold proportion of inundation (water+wet) used to identify the start and end of inundation events - this is currently the long term median but work is underway to improve this. (see note below)
* inundation event statistics maximum proportion inundated and area of inundation
* start and end dates of inundation events, duration of inundation and duration of "gaps" between inundation "events"

Note:  The thresholds used to identify inundation events is based on the individual per-wetland **long-term median** inundation ( **water+wet** ) over the period of satellite record.  *__ New for 2026: __* a floor of 5% area and ceiling if 50% area inundation are applied .  This means an inundation event for a wetlands is not considered to have begun until at least 5% of its area is covered by water.  This prevents sensor noise and tiny single pixels inundation being classified as an inundation event.   At the wetter end, a ceiling of 50% means that permanently wet lakes are considered inundated until they drop to 50% area of water.  This prevents a tiny dip below the median fill amount (e.g. 100% water dropping to 90% full in a permanent lake) being called "the end of an inundation event or the start of a "dry period".  These changes influence the event statistics to make estimates of duration of inundation and duration of inter-flood periods more realistic. These floor and ceiling thresholds can be adjusted in config of the [WIT-metrics code library](https://github.com/Flow-MER/wit-metrics) to generate new event statistics form the raw csv.

[https://knowledge.dea.ga.gov.au/data/product/dea-wetlands-insight-tool-ramsar-wetlands/](https://knowledge.dea.ga.gov.au/data/product/dea-wetlands-insight-tool-ramsar-wetlands/)

[Dunn B, Ai E, Alger MJ, Fanson B, Fickas KC, Krause CE, Lymburner L, Nanson R, Papas P, Ronan M, Thomas RF (2023)  Wetlands Insight Tool: Characterising the Surface Water and Vegetation Cover Dynamics of Individual Wetlands Using Multidecadal Landsat Satellite Data. Wetlands 43, 37. https://doi.org/10.1007/s13157-023-01682-7](https://link.springer.com/article/10.1007/s13157-023-01682-7)

## Lineage

This code was derived from some initial code Geoscience Australia contributed to an MDBA project "BWS Vulnerabilities"
Changes since include:

* modernised code for Pandas 2.0+, Python 3.11
* batch input of multiple WIT CVS in a folder (currently the ANAEv3 WIT output includes 270,653 polygons, each with its own csv file)
* multiple processor pool support to speed execution when running on a PC workstation
* linear interpolation of the observations dates to daily data to improve estimates of inundation duration, the interpolated data permits estimation of monthly stats
* some bug fixes in the inundation event metrics that were required when using the interpolated data
* output formatting

## Dependencies

* A folder containing WIT csv (obtained for the BWS Priorities Project from Geoscience Australia for each ANAE polygon > 1Ha)
* The `wit_metrics_worker.py` module in the same folder as the notebook contains the program code.  This notebook is just the orchestrator to configure and run it.

## Optional

* A shapefile (or equivalent) that contains the area that the WIT result was run over.

## Background

The WIT data are generated by DEA with given wetland polygons and stored in a database on NCI. The data can be dumped into a csv when required. This notebook provides a way in computing temporal statistics (metrics) from the WIT csv.

## WIT Data definition

* WIT csv data files provide the following metrics for each polygon unit

      feature_id: The unique identifier for the polygon
      date: time of observation
      bs: percentage of bare soil
      npv: percentage of non photosynthetic vegetation
      pv: percentage of green/photosynthetic vegetation
      wet: percentage of wetness
      water: percentage of water
      pc_missing: the proportion of missing pixels in the polygon (cloud cover, satellite sensor issues)

## Description

This notebook uses existing WIT data to compute metrics.

* First we load the existing WIT csv data from a saved csv location
* Then we compute the metrics for all polygons and output the results to CSV files.  The input CSV files are processed in "batches" that are spread across multiple CPU cores.  When execution is complete the various batch outputs are merged together into single result files that contain metrics for every CSV feature ID (e.g. ANAE polygons)

The following files are created:

* **RESULT_WIT_yearly_metrics**: min, max, mean, median of each WIT metric per calendar year
* **RESULT_WIT_event_threshold**: Uses an adaptive approach (30th percentile capped by floor of 5% area of polygon to prevent trivial inundation and noise being an inundation event in very dry sites, and ceiling of 50% so that very wet sites are still considered inundated until they dry to < 50% area).
* **RESULT_WIT_time_since_last_inundation**: number of days since the inundation event threshold was exceeded
* **RESULT_WIT_inundation_metrics**: this is a join of the RESULT_WIT_ANAE_event_time and RESULT_WIT_ANAE_event_stats

If the optional debug_event_stats is set True then two intermediate files that are joined to make "RESULT_WIT_ANAE_inundation_metrics" are also saved:

* **RESULT_WIT_event_times**: start and end time, duration, duration of preceding gap (the dry period)
* **RESULT_WIT_event_stats**: for each event calculates the area of the polygon that was wet using the combination water+wet

### Processing Environment

Python 3.11.11

install requirements

```pip install -r requirements.txt```

### Execution in Jupyter environment

* NOTE: all important code has now been migrated into a dedicated python module `wit_metrics_worker.py`.  This notebook provides a visual front-end and is mostly a vehicle to display this introduction and the configuration.  You can run the worker code from this notebook or independently

Load ```wit-metrics.ipynb``` into a Jupyter environment and step through the notebook. It writes the user-defined configuration to ```config.py```, then calls the same standalone entry point provided by ```wit_metrics_worker.py```. Processing is distributed across available CPU cores.

### Alternative method of execution in pure Python

The ```wit_metrics_worker.py``` module is maintained independently from the notebook and can be run directly. First edit ```config.py```, then run it from the command line.

```python wit_metrics_worker.py```

## Contact

Dr Shane Brooks
<https://brooks.eco>

![Brooks.eco logo](brooks-logo.png "Brooks Ecology & Technology")
