import numpy as np
import pandas as pd
import laslib
import time
import cProfile

class VoxelObj(object):
    def __init__(self):
        # voxel object metadata
        self.desc = None
        self.origin = None
        self.step = None
        self.ncells = None
        self.data = None
        self.sample_step = None

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

def add_points_to_voxels(voxel_object, points):
    # convert to voxel coordinate system
    vox_coords = ((points - voxel_object.origin) / voxel_object.step).astype(int)

    # select voxels within range (not needed if successfully interpolated to walls.
    x_filter = (vox_coords[:, 0] >= 0) & (vox_coords[:, 0] < voxel_object.ncells[0])
    y_filter = (vox_coords[:, 1] >= 0) & (vox_coords[:, 1] < voxel_object.ncells[1])
    z_filter = (vox_coords[:, 2] >= 0) & (vox_coords[:, 2] < voxel_object.ncells[2])
    in_range = x_filter & y_filter & z_filter
    vox_in_range = vox_coords[in_range]

    # format
    vox_address = (vox_in_range[:, 0], vox_in_range[:, 1], vox_in_range[:, 2])

    np.add.at(voxel_object.data, vox_address, 1)

    return voxel_object


def las_ray_sample(hdf5_path, sample_step, voxel_length, return_set='all'):
    start = time.time()
    print('Loading data...')
    # load returns
    returns = pd.read_hdf(hdf5_path, key='las_data', columns=['gps_time', 'x', 'y', 'z', 'classification', 'return_num', 'num_returns'])
    # load trajectory
    traj = pd.read_hdf(hdf5_path, key='las_traj', columns=['traj_x', 'traj_y', 'traj_z', 'distance_from_sensor_m'])

    # concatenate rays
    rays = pd.concat([returns, traj], axis=1)
    # filter by class
    rays = rays[(rays.classification == 2) | (rays.classification == 5)]

    if return_set == "all":
        # filter to last returns for sampling
        p_all = np.array(rays.loc[:, ['x', 'y', 'z']])
        rays = rays[rays.return_num == rays.num_returns]
    elif return_set == "first":
        # filter to first returns for sampling
        rays = rays[rays.return_num == 1]
    else:
        raise Exception("Expected 'first' or 'all' for return_set, encountered:" + str(return_set))

    # drop unnecessary columns
    rays = rays[['gps_time', 'x', 'y', 'z', 'traj_x', 'traj_y', 'traj_z', 'distance_from_sensor_m']]

    # interpolate rays to z-max ceiling (could do same thing for x-y bounds...
    z_ceiling = np.max(rays.z).astype(int) + 1
    rays = rays.assign(x_c=(z_ceiling - rays.traj_z) * (rays.traj_x - rays.x) / (rays.traj_z - rays.z) + rays.traj_x)
    rays = rays.assign(y_c=(z_ceiling - rays.traj_z) * (rays.traj_y - rays.y) / (rays.traj_z - rays.z) + rays.traj_y)
    rays = rays.assign(z_c=z_ceiling)

    # p0 -> trajectory/sensor,
    p0 = np.array(rays.loc[:, ['x_c', 'y_c', 'z_c']])
    # p1 -> returns
    p1 = np.array(rays.loc[:, ['x', 'y', 'z']])

    # calculate distance between source and last return
    dist = np.sqrt(np.sum((p1 - p0) ** 2, axis=1))

    # define voxel object
    voxS = VoxelObj()
    voxS.desc = "ray sampling of " + hdf5_path
    voxS.step = np.array([voxel_length, voxel_length, -voxel_length])
    voxS.origin = np.array([np.min(rays.x), np.min(rays.y), z_ceiling])
    voxS.ncells = np.array([int((np.max(rays.x) - voxS.origin[0]) / voxS.step[0]) + 1,
                           int((np.max(rays.y) - voxS.origin[1]) / voxS.step[1]) + 1,
                           int((np.min(rays.z) - voxS.origin[2]) / voxS.step[2]) + 1])
    voxS.data = np.zeros(voxS.ncells)
    voxS.sample_step = sample_step

    # calc sample step in x, y, z dims
    xyz_step = (p1 - p0)/dist[:, np.newaxis]

    # random offset seed for each ray sample series
    offset = np.random.random(len(p0))

    # initiate while loop
    ii = 0

    max_dist = np.max(dist)
    # iterate until longest ray length is surpassed

    end = time.time()
    print('done in ' + str(end - start) + ' seconds.')

    print('Voxel ray sampling...')
    start = time.time()

    while (ii * sample_step) < max_dist:
        print(str(ii + 1) + ' of ' + str(int(max_dist/sample_step) + 1))
        # distance from p0 along ray
        sample_dist = (ii + offset) * sample_step

        # select rays where t_dist is in range
        in_range = (dist > sample_dist)

        # calculate tracer point coords for step
        sample_points = xyz_step[in_range, :] * sample_dist[in_range, np.newaxis] + p0[in_range]

        if np.size(sample_points) != 0:
            # add counts to voxel_samples
            voxS = add_points_to_voxels(voxS, sample_points)

        # advance step
        ii = ii + 1

    end = time.time()
    print('done in ' + str(end - start) + ' seconds.')

    # voxel sample returns
    # ONLY 1st or last returns used here... needs more thorough consideration
    print('sampling returns...')
    start = time.time()
    voxR = voxS.copy()
    voxR.data = np.zeros(voxR.ncells)
    if return_set == 'all':
        voxR = add_points_to_voxels(voxR, p_all)
    elif return_set == 'first':
        voxR = add_points_to_voxels(voxR, p1)
    end = time.time()
    print('done in ' + str(end - start) + ' seconds.')

    return voxR, voxS


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

sample_step = 1
voxel_length = 1
voxR, voxS = las_ray_sample(hdf5_path, sample_step, voxel_length, return_set='all')


# want to sample vox_trans along an arbitrary path

# convert voxel counts to path length units [m]
voxS.data = voxS.data * voxS.sample_step
# turn 0 samples to nans
voxS.data[voxS.data == 0] = np.nan
# calculate transmission
transmission = voxR.data / voxS.data

# input points
# pull in raster DEM, use cell center and elevation

# integration vectors
# theta = blah
# phi = blah
# calculate end point at z_ceiling

# sample_step = sample step
# same algorithm as for ray sampling above to generate points
# in this case, sum(?) voxel transmission values (normalized to sample path length)

# return likely number of returns



