# config/routing script for las processing and analysis
# outline of essential python workflow for cob's masters

# ----- Pre-processing -----
# run init_config.bat to conduct lastools preprocessing and classification

# ----- DEM analysis -----
# calculate dems and differential dem products
import analysis.hs_swe_data_products

# ----- Canopy analysis -----
# tree_top_kho, distance to nearest tree (DNT) -- Khosravipour et al. 2016
import canopy_metrics.tree_top_kho
# distance to canopy edge (DCE) -- Mazzotti et al 2019
import canopy_metrics.distance_to_canopy_edge
# mean distance to canopy(MDC), total gap area (TGA) -- Moeser et al. 2015
import canopy_metrics.moeser_vector_search
# laser penetation metrics (lpm) -- Alonzo et al 2015
import canopy_metrics.lpm_map_gen

# lidar point reprojection (eg. Moeser et al. 2014)
# ---------------------
# # hemiphoto analysis
import lidar_point_cloud_reprojection. hemi_optimization  # (optimizing synthetics)
# hemisfer (photos/optimizing synthetics)
import analysis.hemisfer_parse  # (photos/optimizing synthetics)
# hemi_optimization_2.r (select optimization parameters)
import analysis.grid_points  # (establish grid for hemispheres)
import lidar_point_cloud_reprojection.hemi_from_pts  # (synthetic hemispherical images over upper forest grid)
# hemisfer (synthetics)
import analysis.hemisfer_parse  # (synthetics)
# ---------------------

# voxel ray sampling of lidar
# ---------------------
import voxel_ray_sampling_of_lidar.las_ray_sample_hemi_optimization
import voxel_ray_sampling_of_lidar.las_ray_sample_from_grid
# lrs_hemi_optimization.r
import voxel_ray_sampling_of_lidar.lrs_footprint_products
import voxel_ray_sampling_of_lidar.light_penetration_with_angle
# ---------------------

# mCH
import canopy_metrics.mean_canopy_height

# ----- Combined analysis -----
# merged data products
import analysis.merged_data_products

# snow transmission modeling
import analysis.snow_transmission_modeling

# frequency distributions
import analysis.frequency_distributions
# swe_distribution_fitting.r

# semivariogram analysis
import analysis.semivar_analysis

# scatter plots
import analysis.scatter_plots