# config/routing script for las processing and analysis

# ----- Pre-processing -----
# run init_config.bat to conduct lastools preprocessing and classification

# ----- DEM analysis -----
# calculate dems and differential dem products
import hs_dhs_data_products

# ----- Canopy analysis -----
# hemigen
# tree_top_kho
import tree_top_kho
# dce
import distance_to_canopy_edge
# lpm calculation
import lpm_map_gen
# ray_sampling_method

# merged data products
import merged_data_products