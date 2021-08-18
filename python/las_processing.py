# config/routing script for las processing and analysis
# outline of essential python workflow for cob's masters

# ----- Pre-processing -----
# run init_config.bat to conduct lastools preprocessing and classification

# ----- DEM analysis -----
# calculate dems and differential dem products
import hs_swe_data_products

# ----- Canopy analysis -----
# tree_top_kho, distance to nearest tree (DNT) -- Khosravipour et al. 2016
import tree_top_kho
# distance to canopy edge (DCE) -- Mazzotti et al 2019
import distance_to_canopy_edge
# mean distance to canopy(MDC), total gap area (TGA) -- Moeser et al. 2015
import moeser_vector_search
# laser penetation metrics (lpm) -- Alonzo et al 2015
import lpm_map_gen

# lidar point reprojection (eg. Moeser et al. 2014)
# ---------------------
# # hemiphoto analysis
import hemi_optimization  # (optimizing synthetics)
# hemisfer (photos/optimizing synthetics)
import hemisfer_parse  # (photos/optimizing synthetics)
# hemi_optimization_2.r (select optimization parameters)
import grid_points  # (establish grid for hemispheres)
import hemi_from_pts  # (synthetic hemispherical images over upper forest grid)
# hemisfer (synthetics)
import hemisfer_parse  # (synthetics)
# ---------------------

# voxel ray sampling of lidar
# ---------------------
import las_ray_sample_hemi_optimization
import las_ray_sample_from_grid
# lrs_hemi_optimization.r
import lrs_footprint_products
import light_penetration_with_angle
# ---------------------

# mCH
import mean_canopy_height

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