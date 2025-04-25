

#  set the working directory - needs to have enough free space for the generated outputs
working_directory = "d:/wit-metrics/output"
working_directory = r"T:\ANAE_WIT_Apr_2025_RESULTS"


# shapefile: the shape file mentioned above to find the  and get their area
# set to '' to disable area lookup
shapefile = "d:/wit-metrics/input/shp/ANAEv3_WIT.shp"
shapefile = r"D:\BWSVulnerability\WIT\ANAEv3_WIT_clean19042022\ANAEv3_WIT.shp"
# shapefile field name that identifies each polygon -  the ANAEv3 UID geohash was used here.
# The ANAE UID is also used in the naming convention for the CSV files
shape_uid = "UID"

# Path to folder that contains the WIT csv files to process.  When debugging providing a single file might be prudent.
csv_files = "d:/wit-metrics/input/csv"
csv_files = r"T:\test_data"

#Only use WIT data where the pc_missing is less than the threshold (default is 0.1) i.e. 90% of polygon was visible to satellites
pc_missing_threshold = 0.1

# csv feature_id - the WIT csv output files include a column 'feature_id' that in this case is the ANAE UID
pkey = "feature_id"

# whether to interpolate the WIT observation dates (typically 10-50+ per year) to daily data (365 per year)
# This is computationally expensive but improves estimates of inundation duration and time since last inundation.
# Monthly and Annual statistics are summarised from the daily data.
interpolate_to_daily = True

# set to True to save the interpolated daily WIT csv in a subfolder under the csvfiles (unnecessary and take up a lot of space but good for debugging)
save_interpolated_csv = False

# set to true to save intermediate data frames containing the event times and stats
# these are saved in the working directory
debug_event_times = False

# batchsize is the number of WIT csv files to include in each 'batch' that is processed by each single CPU core.
# The code was designed to process several 100,000 polygons in small batches that fit into the computer memory
# then glue all the batch results together at the end.
# On a workstation with 16 cpu cores and 64MB RAM a batchsize of 100-200 worked well. With smaller number of CSV a batch size < total number of CSV allows the
# calculations to be spread across multiple processors.
# during processing the code will generate outputs for each batch then glue them together at the end.

batch_size = 100
