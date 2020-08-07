import numpy as np
import pandas as pd
import laslib
import rastools
import time
import cProfile

class VoxelObj(object):
    def __init__(self):
        # voxel object metadata
        self.desc = None
        self.origin = None
        self.max = None
        self.step = None
        self.ncells = None
        self.sample_step = None
        self.sample_set = None
        self.sample_data = None
        self.return_set = None
        self.return_data = None

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)


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


def interpolate_to_bounding_box(returns, traj):
    if returns.shape != traj.shape:
        raise Exception('returns and traj have different shapes!')

    traj_bb = traj.copy()

    # define bounding box (lower bounds and upper bounds)
    lb = np.min(returns, axis=0)
    ub = np.max(returns, axis=0)

    # for each dimension
    for ii in range(0, 3):
        # for traj points outside bounding box of returns
        lows = (traj_bb[:, ii] < lb[ii])
        highs = (traj_bb[:, ii] > ub[ii])
        for jj in range(0, 3):
            # interpolate traj to bounding box
            traj_bb[lows, jj] = (lb[ii] - traj[lows, ii]) * (traj[lows, jj] - returns[lows, jj]) / (
                    traj[lows, ii] - returns[lows, ii]) + traj[lows, jj]
            traj_bb[highs, jj] = (ub[ii] - traj[highs, ii]) * (traj[highs, jj] - returns[highs, jj]) / (
                        traj[highs, ii] - returns[highs, ii]) + traj[highs, jj]
    return traj_bb


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

    # calculate distance between ray start and end
    dist = np.sqrt(np.sum((ray_1 - ray_bb) ** 2, axis=1))

    # calc unit step along ray in x, y, z dims (avoid edge cases where dist == 0)
    xyz_step = np.full([len(dist), 3], np.nan)
    xyz_step[dist > 0, :] = (ray_1[dist > 0] - ray_bb[dist > 0])/dist[dist > 0, np.newaxis]

    # random offset for each ray sample series
    offset = np.random.random(len(ray_1))

    # initiate while loop
    ii = 0
    max_dist = np.max(dist)

    # iterate until longest ray length is surpassed
    while (ii * vox.sample_length) < max_dist:
        print(str(ii + 1) + ' of ' + str(int(max_dist/vox.sample_length) + 1))
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


    # voxel sample returns
    print('sampling ' + vox.return_set + ' returns...')
    vox = add_points_to_voxels(vox, "returns", return_points)

    end = time.time()
    print('done in ' + str(end - start) + ' seconds.')

    return vox


# las file
las_in = 'C:\\Users\\jas600\\workzone\\data\\las\\19_149_UF.las'
# trajectory file
traj_in = 'C:\\Users\\jas600\\workzone\\data\\las\\19_149_all_traj.txt'
# working hdf5 file
hdf5_path = las_in.replace('.las', '_ray_sampling.hdf5')

# # write las to hdf5
# laslib.las_to_hdf5(las_in, hdf5_path)
# # interpolate trajectory
# laslib.las_traj(hdf5_path, traj_in)

sample_length = 1
voxel_length = 1
vox = las_ray_sample(hdf5_path, sample_length, voxel_length, return_set='all')


# input points
# pull in raster DEM, use cell center and elevation
dem_in = "C:\\Users\\jas600\\workzone\\data\\dem\\19_149_dem_res_1.00m.bil"
dem = rastools.raster_load(dem_in)

# specify starting points
xy = np.swapaxes(np.array(dem.T1 * np.where(dem.data != dem.no_data)), 0, 1)
z = dem.data[np.where(dem.data != dem.no_data)]
source_utm = np.concatenate([xy, z[:, np.newaxis]], axis=1)


source_vox = utm_to_vox(vox, source_utm)
# select voxels within range (not needed if successfully interpolated to walls.
in_range = np.all(([0, 0, 0] <= source_vox) & (source_vox <= vox.ncells), axis=1)
source_utm = source_utm[in_range]
# want to sample vox_trans along an arbitrary path

# specify integration vectors (phi from nadir, theta from North)
phi = 0  # angle from nadir in degrees
theta = 0  # angle cw from N in degrees

# signs need to be checked to make sure this all adds up to reference frame definition
dz = vox.max[2] - source_utm[in_range, 2]
dx = dz * np.sin(theta * np.pi / 180) * np.tan(phi * np.pi / 180)
dy = dz * np.cos(theta * np.pi / 180) * np.tan(phi * np.pi / 180)

# calculate end point at z_ceiling
sink_utm = source_utm[in_range] + np.swapaxes(np.array([dx, dy, dz]), 0, 1)

# sample_step = sample step
# same algorithm as for ray sampling above to generate points
# in this case, sum(?) voxel transmission values (normalized to sample path length)

# return likely number of returns



# convert voxel counts to path length units [m]
voxS.data = voxS.data * voxS.sample_step
# turn 0 samples to nans
voxS.data[voxS.data == 0] = np.nan
# calculate transmission
transmission = voxR.data / voxS.data

# play

vox.step = np.array([2, 2, 2])
vox.sample_step = 1
ii = 1

# original points
p0 = source_utm[in_range]
p1 = p0 + np.array([1, 1, 1])

# calculate a sample
norm = np.sqrt(np.sum((p0 - p1) ** 2, axis=1))
xyz_step = (p0 - p1)/norm[:, np.newaxis]
ps = p0 + xyz_step * ii * vox.sample_step
# convert to voxel coordinates
vs_1 = utm_to_vox(vox, ps)

# convert to voxel coordinates
us = 1/vox.step
v0 = utm_to_vox(vox, p0)
v1 = utm_to_vox(vox, p1)
# calculate a sample
norm = np.sqrt(np.sum(((v0 - v1) * us) ** 2, axis=1))
xyz_step = (v0 - v1)/norm[:, np.newaxis]

vs_2 = v0 + xyz_step * ii * vox.sample_step

np.max(np.abs(vs_1 - vs_2))

# i don't understand why these do not equal one another. might need to be worked out in mathematica...