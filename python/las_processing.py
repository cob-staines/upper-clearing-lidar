# config/routing script for las processing and analysis

# ----- Pre-processing -----
# run init_config.bat to conduct lastools preprocessing and classification

# ----- DEM analysis -----
# calculate dems and differential dem products
import hs_swe_data_products

# ----- Canopy analysis -----
# # hemi'photo analysis
# hemi_optimization.py (optimizing synthetics)
# hemisfer (photos/optimizing synthetics)
# hemisfer_parse.py (photos/optimizing synthetics)
# hemi_optimization_2.r (select optimization parameters)
# hemi_grid_points.py (establish grid for hemispheres and
# hemi_from_pts.py (synthetics)
# hemisfer (synthetics)
# hemisfer_parse.py (synthetics)

# tree_top_kho
import tree_top_kho
# dce
import distance_to_canopy_edge
# mCH
import mean_canopy_height
# lpm calculation
import lpm_map_gen
# ray_sampling_method

# ----- Combined analysis -----
# merged data products
import merged_data_products

# snow transmission modeling
import snow_transmission_modeling

# frequency distributions
import frequency_distributions
# swe_distribution_fitting.r

# semivariogram analysis
import semivar_analysis

# scatter plots
import scatter_plots