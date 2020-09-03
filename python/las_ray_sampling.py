import numpy as np
import pandas as pd
import laslib
import rastools
import time
import cProfile
import h5py


class VoxelObj(object):
    def __init__(self):
        # voxel object metadata
        self.desc = None
        self.origin = None
        self.max = None
        self.step = None
        self.ncells = None
        self.sample_length = None
        self.sample_set = None
        self.sample_data = None
        self.return_set = None
        self.return_data = None

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def save(self, hdf5_path):
        vox_save(self, hdf5_path)


def vox_save(vox, hdf5_path):
    h5f = h5py.File(hdf5_path, 'r+')
    h5f.create_dataset('vox_desc', data=vox.desc)
    h5f.create_dataset('vox_origin', data=vox.origin)
    h5f.create_dataset('vox_max', data=vox.max)
    h5f.create_dataset('vox_step', data=vox.step)
    h5f.create_dataset('vox_ncells', data=vox.ncells)
    h5f.create_dataset('vox_sample_length', data=vox.sample_length)
    h5f.create_dataset('vox_sample_set', data=vox.sample_set)
    h5f.create_dataset('vox_sample_data', data=vox.sample_data)
    h5f.create_dataset('vox_return_set', data=vox.return_set)
    h5f.create_dataset('vox_return_data', data=vox.return_data)


def vox_load(hdf5_path):
    vox = VoxelObj()
    h5f = h5py.File(hdf5_path, 'r')
    vox.desc = h5f.get('vox_desc')[()]
    vox.origin = h5f.get('vox_origin')[()]
    vox.max = h5f.get('vox_max')[()]
    vox.step = h5f.get('vox_step')[()]
    vox.ncells = h5f.get('vox_ncells')[()]
    vox.sample_length = h5f.get('vox_sample_length')[()]
    vox.sample_set = h5f.get('vox_sample_set')[()]
    vox.sample_data = h5f.get('vox_sample_data')[()]
    vox.return_set = h5f.get('vox_return_set')[()]
    vox.return_data = h5f.get('vox_return_data')[()]
    return vox


def utm_to_vox(voxel_object, utm_points):
    return (utm_points - voxel_object.origin) / voxel_object.step


def vox_to_utm(voxel_object, vox_points):
    return vox_points * voxel_object.step + voxel_object.origin


def add_points_to_voxels(voxel_object, dataset, points):
    # convert to voxel coordinate system
    vox_coords = utm_to_vox(voxel_object, points).astype(int)

    # select voxels within range (not needed if successfully interpolated to walls.
    in_range = np.all(([0, 0, 0] <= vox_coords) & (vox_coords <= voxel_object.ncells), axis=1)
    if np.sum(~in_range):
        raise Exception('You thought that points would not show up out of bounds... you thought wrong.')
    vox_in_range = vox_coords[in_range]

    # format
    vox_address = (vox_in_range[:, 0], vox_in_range[:, 1], vox_in_range[:, 2])

    if dataset == "samples":
        np.add.at(voxel_object.sample_data, vox_address, 1)
    elif dataset == "returns":
        np.add.at(voxel_object.return_data, vox_address, 1)
    else:
        raise Exception("Expected 'samples' or 'returns' for dataset, encountered:" + str(dataset))

    return voxel_object


def interpolate_to_bounding_box(fixed_points, flex_points, bb=None):
    if fixed_points.shape != flex_points.shape:
        raise Exception('fixed_points and flex_points have different shapes!')

    bb_points = flex_points.copy()

    if bb:
        lb = bb[0]
        ub = bb[1]
    else:
        # define bounding box (lower bounds and upper bounds)
        lb = np.min(fixed_points, axis=0)
        ub = np.max(fixed_points, axis=0)

    # for each dimension
    for ii in range(0, 3):
        # for flex_points points outside bounding box of fixed_points
        lows = (bb_points[:, ii] < lb[ii])
        highs = (bb_points[:, ii] > ub[ii])
        for jj in range(0, 3):
            # interpolate flex_points to bounding box of fixed_points
            bb_points[lows, jj] = (lb[ii] - flex_points[lows, ii]) * (flex_points[lows, jj] - fixed_points[lows, jj]) / (
                    flex_points[lows, ii] - fixed_points[lows, ii]) + flex_points[lows, jj]
            bb_points[highs, jj] = (ub[ii] - flex_points[highs, ii]) * (flex_points[highs, jj] - fixed_points[highs, jj]) / (
                        flex_points[highs, ii] - fixed_points[highs, ii]) + flex_points[highs, jj]
    return bb_points


def las_ray_sample(hdf5_path, sample_length, voxel_length, return_set='all'):
    start = time.time()

    # define voxel object
    vox = VoxelObj()
    vox.desc = "ray sampling of " + hdf5_path
    vox.sample_length = sample_length
    vox.return_set = return_set


    print('Loading data...')
    # load returns
    returns_df = pd.read_hdf(hdf5_path, key='las_data', columns=['gps_time', 'x', 'y', 'z', 'classification', 'return_num', 'num_returns'])
    # load trajectory
    traj_df = pd.read_hdf(hdf5_path, key='las_traj', columns=['traj_x', 'traj_y', 'traj_z'])

    # concatenate rays
    rays = pd.concat([returns_df, traj_df], axis=1)
    # filter by class (should be moved to input...)
    rays = rays[(rays.classification == 2) | (rays.classification == 5)]

    # filter rays according to return_set
    if vox.return_set == "all":
        # use last returns for ray sampling
        vox.sample_set = "last"
        ray_set = rays[rays.return_num == rays.num_returns]

        # use all returns for return sampling
        return_points = np.array(rays.loc[:, ['x', 'y', 'z']])
    elif vox.return_set == "first":
        # use first returns for ray sampling
        vox.sample_set = "first"
        ray_set = rays[rays.return_num == 1]

        # use first returns for return sampling
        return_points = np.array(ray_set.loc[:, ['x', 'y', 'z']])
    else:
        raise Exception("Expected 'first' or 'all' for return_set, encountered:" + str(return_set))

    # trajectory/sensor
    ray_0 = np.array(ray_set.loc[:, ['traj_x', 'traj_y', 'traj_z']])
    # return points
    ray_1 = np.array(ray_set.loc[:, ['x', 'y', 'z']])

    # calculate voxel space secifications
    vox.step = np.full(3, voxel_length)  # allow to be specified independently
    vox.origin = np.min(ray_1, axis=0)
    vox.max = np.max(ray_1, axis=0)
    vox.ncells = ((np.max(ray_1, axis=0) - vox.origin) / vox.step).astype(int) + 1
    # preallocate voxels
    vox.sample_data = np.zeros(vox.ncells)
    vox.return_data = np.zeros(vox.ncells)

    print('Interpolating to bounding box...')
    # interpolate rays to bounding box
    ray_bb = interpolate_to_bounding_box(ray_1, ray_0)

    print('Voxel sampling of ' + vox.sample_set + ' return rays ...')
    # calculate distance between ray start (p0) and end (p1)
    dist = np.sqrt(np.sum((ray_1 - ray_bb) ** 2, axis=1))

    # calc unit step along ray in x, y, z dims (avoid edge cases where dist == 0)
    xyz_step = np.full([len(dist), 3], np.nan)
    xyz_step[dist > 0, :] = (ray_1[dist > 0] - ray_bb[dist > 0]) / dist[dist > 0, np.newaxis]

    # random offset for each ray sample series
    offset = np.random.random(len(ray_1))

    # initiate while loop
    ii = 0
    max_dist = np.max(dist)

    # iterate until longest ray length is surpassed
    while (ii * vox.sample_length) < max_dist:
        print(str(ii + 1) + ' of ' + str(int(max_dist / vox.sample_length) + 1))
        # distance from p0 along ray
        sample_dist = (ii + offset) * vox.sample_length

        # select rays where t_dist is in range
        in_range = (dist > sample_dist)

        # calculate tracer point coords for step
        sample_points = xyz_step[in_range, :] * sample_dist[in_range, np.newaxis] + ray_bb[in_range]

        if np.size(sample_points) != 0:
            # add counts to voxel_samples
            vox = add_points_to_voxels(vox, "samples", sample_points)
        # advance step
        ii = ii + 1

    # correct sample_data to unit length (ex. meters)
    vox.sample_data = vox.sample_data * vox.sample_length

    # voxel sample returns
    print('sampling ' + vox.return_set + ' returns...')
    vox = add_points_to_voxels(vox, "returns", return_points)

    end = time.time()
    print('done in ' + str(end - start) + ' seconds.')

    return vox


def aggregate_voxels_over_rays(vox, rays, agg_sample_length, iterations=100):
    print('Aggregating voxels over rays:')

    # pull points
    p0 = rays.values[:, 0:3]
    p1 = rays.values[:, 3:6]


    # calculate distance between ray start (ground) and end (sky)
    dist = np.sqrt(np.sum((p1 - p0) ** 2, axis=1))  # DO WE WANT TO ADD THIS VALUE TO THE DF?

    # calc unit step along ray in x, y, z dims
    xyz_step = (p1 - p0) / dist[:, np.newaxis]

    # random offset for each ray sample series
    offset = np.random.random(len(p0))

    # calculate number of samples
    n_samples = ((dist - offset) / agg_sample_length).astype(int)
    max_steps = np.max(n_samples)

    # preallocate aggregate lists
    path_samples = np.full([len(p0), max_steps], np.nan)
    path_returns = np.full([len(p0), max_steps], np.nan)

    # for each sample step
    print('Sampling voxels...')
    for ii in range(0, max_steps):
        # distance from p0 along ray
        sample_dist = (ii + offset) * agg_sample_length

        # select rays where t_dist is in range
        in_range = (dist > sample_dist)

        # calculate tracer point coords for step
        sample_points = xyz_step[in_range, :] * sample_dist[in_range, np.newaxis] + p0[in_range]

        if np.size(sample_points) != 0:
            # add voxel value to list
            sample_vox = utm_to_vox(vox, sample_points).astype(int)
            sample_address = (sample_vox[:, 0], sample_vox[:, 1], sample_vox[:, 2])

            path_samples[in_range, ii] = vox.sample_data[sample_address]
            path_returns[in_range, ii] = vox.return_data[sample_address]

        print(str(ii + 1) + ' of ' + str(max_steps) + ' steps')

    # calculate expected points and varience
    # MOVE PARAMETERS TO PASSED VARIABLE
    ratio = .05  # ratio of voxel area weight of prior
    F = .16 * 0.05  # expected footprint area
    V = np.prod(vox.step)  # volume of each voxel
    K = np.sum(vox.return_data)  # total number of returns in set
    N = np.sum(vox.sample_data)  # total number of meters sampled in set

    # gamma prior hyperparameters
    prior_b = ratio * V / F  # path length required to scan "ratio" of one voxel volume
    prior_a = prior_b * 1  # prior_b * K / N

    permutations = int(iterations * 0.1)

    returns_mean = np.full(len(path_samples), np.nan)
    returns_med = np.full(len(path_samples), np.nan)
    returns_std = np.full(len(path_samples), np.nan)
    print('Aggregating samples over each ray...')
    for ii in range(0, len(path_samples)):
        kk = path_returns[ii, 0:n_samples[ii]]
        nn = path_samples[ii, 0:n_samples[ii]]

        # posterior hyperparameters
        post_a = kk + prior_a
        post_b = 1 - 1 / (1 + prior_b + nn)

        nb_samples = np.full([iterations, n_samples[ii]], 0)
        for jj in range(0, n_samples[ii] - 1):
            nb_samples[:, jj] = np.random.negative_binomial(post_a[jj], post_b[jj], iterations)

        # correct for agg sample length
        nb_samples = nb_samples * agg_sample_length

        # permutate sums for resultant distribution
        return_sums = np.full([iterations, permutations], np.nan)
        for jj in range(0, permutations):
            return_sums[:, jj] = np.sum(np.random.permutation(nb_samples), axis=1)
        return_sums = np.reshape(return_sums, return_sums.size)

        returns_mean[ii] = np.mean(return_sums)
        returns_med[ii] = np.median(return_sums)
        returns_std[ii] = np.std(return_sums)

        print(str(ii + 1) + ' of ' + str(len(path_samples)) + ' rays')

    rays = rays.assign(path_length=dist)
    rays = rays.assign(returns_mean=returns_mean)
    rays = rays.assign(returns_median=returns_med)
    rays = rays.assign(returns_std=returns_std)

    return rays


def dem_to_vox_rays(dem_in, vec, vox):
    # why pass vox here? consider workaround/more general approach for ray bounding

    # load raster dem
    dem = rastools.raster_load(dem_in)

    # use cell center and elevation as ray source (where data exists)
    xy = np.swapaxes(np.array(dem.T1 * np.where(dem.data != dem.no_data)), 0, 1)
    z = dem.data[np.where(dem.data != dem.no_data)]
    ground_all = np.concatenate([xy, z[:, np.newaxis]], axis=1)

    # specify integration rays
    phi = vec[0]  # angle from nadir in degrees
    theta = vec[1]  # angle cw from N in degrees

    # calculate endpoint at z_max
    dz = vox.max[2] - ground_all[:, 2]
    dx = dz * np.sin(theta * np.pi / 180) * np.tan(phi * np.pi / 180)
    dy = dz * np.cos(theta * np.pi / 180) * np.tan(phi * np.pi / 180)
    sky_all = ground_all + np.swapaxes(np.array([dx, dy, dz]), 0, 1)

    # select rays with both points within voxel bounding box
    ground_in_range = np.all((vox.origin <= ground_all) & (ground_all <= vox.max), axis=1)
    sky_in_range = np.all((vox.origin <= sky_all) & (sky_all <= vox.max), axis=1)
    ground = ground_all[ground_in_range & sky_in_range]
    sky = sky_all[ground_in_range & sky_in_range]

    df = pd.DataFrame({'x0': ground[:, 0],
                       'y0': ground[:, 1],
                       'z0': ground[:, 2],
                       'x1': sky[:, 0],
                       'y1': sky[:, 1],
                       'z1': sky[:, 2]})
    return df


def ray_stats_to_dem(rays, dem_in):
    # load raster dem
    dem = rastools.raster_load(dem_in)

    # pull points
    p0 = rays.values[:, 0:3]
    p1 = rays.values[:, 3:6]

    ground_dem = np.rint(~dem.T1 * (p0[:, 0], p0[:, 1])).astype(int)
    ground_dem = (ground_dem[1], ground_dem[0])  # check index, make sure correct

    ras = dem
    shape = ras.data.shape
    ras.data = []
    ras.data.append(np.full(shape, np.nan))
    ras.data.append(np.full(shape, np.nan))
    ras.data.append(np.full(shape, np.nan))

    ras.data[0][ground_dem] = rays.returns_mean
    ras.data[0][np.isnan(ras.data[0])] = ras.no_data

    ras.data[1][ground_dem] = rays.returns_median
    ras.data[1][np.isnan(ras.data[1])] = ras.no_data

    ras.data[2][ground_dem] = rays.returns_std
    ras.data[2][np.isnan(ras.data[2])] = ras.no_data

    ras.band_count = 3

    return ras



# las file
# las_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_UF.las'
las_in = 'C:\\Users\\jas600\\workzone\\data\\las\\19_149_UF.las'
# trajectory file
# traj_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_traj.txt'
traj_in = 'C:\\Users\\jas600\\workzone\\data\\las\\19_149_all_traj.txt'
# working hdf5 file
hdf5_path = las_in.replace('.las', '_ray_sampling_0.1.hdf5')

# # write las to hdf5
laslib.las_to_hdf5(las_in, hdf5_path)
# # interpolate trajectory
laslib.las_traj(hdf5_path, traj_in)

voxel_length = 0.1
vox_sample_length = voxel_length/np.pi
vox = las_ray_sample(hdf5_path, vox_sample_length, voxel_length, return_set='first')
vox_save(vox, hdf5_path)


vox = vox_load(hdf5_path)


# sample voxel space
# dem_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\DEM\\19_149_dem_res_.10m.bil"
dem_in = "C:\\Users\\jas600\\workzone\\data\\dem\\19_149_dem_res_.10m.bil"
# ras_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\DEM\\19_149_expected_returns_res_.10m_0-0_t_1.tif"
ras_out = "C:\\Users\\jas600\\workzone\\data\\dem\\19_149_expected_returns_res_.10m.tif"
phi = 10
theta = 0
agg_sample_length = vox.sample_length
vec = [phi, theta]
rays_in = dem_to_vox_rays(dem_in, vec, vox)
rays_out = aggregate_voxels_over_rays(vox, rays_in, agg_sample_length)
ras = ray_stats_to_dem(rays_out, dem_in)
rastools.raster_save(ras, ras_out)


### SANDBOX

cProfile.run('rays_out = aggregate_voxels_over_dem(vox, rays_in, agg_sample_length)')

peace = rastools.raster_load(ras_out)


# convert voxel counts to path length units [m]
voxS.data = voxS.data * voxS.sample_length
# turn 0 samples to nans
voxS.data[voxS.data == 0] = np.nan
# calculate transmission
transmission = voxR.data / voxS.data

# play

vox.step = np.array([2, 2, 2])
vox.sample_length = 1
ii = 1

# original points
p0 = source_utm[in_range]
p1 = p0 + np.array([1, 1, 1])

# calculate a sample
norm = np.sqrt(np.sum((p0 - p1) ** 2, axis=1))
xyz_step = (p0 - p1)/norm[:, np.newaxis]
ps = p0 + xyz_step * ii * vox.sample_length
# convert to voxel coordinates
vs_1 = utm_to_vox(vox, ps)

# convert to voxel coordinates
us = 1/vox.step
v0 = utm_to_vox(vox, p0)
v1 = utm_to_vox(vox, p1)
# calculate a sample
norm = np.sqrt(np.sum(((v0 - v1) * us) ** 2, axis=1))
xyz_step = (v0 - v1)/norm[:, np.newaxis]

vs_2 = v0 + xyz_step * ii * vox.sample_length

np.max(np.abs(vs_1 - vs_2))

# i don't understand why these do not equal one another. might need to be worked out in mathematica...

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


peace_1 = peace.data[0]
peace_1[peace_1 == peace.no_data] = -1
plt.imshow(peace_1, interpolation='nearest')

peace_2 = peace.data[1]
peace_2[peace_2 == peace.no_data] = 1
plt.imshow(peace_2, interpolation='nearest')

plt.imshow(ras.data[0], interpolation='nearest')



plt.scatter(rays_out.x0, rays_out.y0)
plt.scatter(ground_dem[0], ground_dem[1])
plt.scatter(p0[:, 0], p0[:, 1])