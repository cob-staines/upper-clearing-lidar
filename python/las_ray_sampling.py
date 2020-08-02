import numpy as np
import pandas as pd
import laslib
import time
import cProfile

class VoxelObj(object):
    def __init__(self, name):
        # preload metadata
        self.name = name
        self.origin = None
        self.step = None
        self.count = None
        self.sample_step = None

def add_points_to_voxels(voxel_space, vox_origin, vox_step, vox_count, points):
    # convert to voxel coordinate system
    vox_coords = ((points - vox_origin) / vox_step).astype(int)

    # find counts of unique voxels
    vox_unique, sample_count = np.unique(vox_coords, axis=0, return_counts=True)

    # select voxels within range
    x_filter = (vox_unique[:, 0] >= 0) & (vox_unique[:, 0] < vox_count[0])
    y_filter = (vox_unique[:, 1] >= 0) & (vox_unique[:, 1] < vox_count[1])
    z_filter = (vox_unique[:, 2] >= 0) & (vox_unique[:, 2] < vox_count[2])
    in_range = x_filter & y_filter & z_filter
    vox_in_range = vox_unique[in_range]

    # format
    vox_address = (vox_in_range[:, 0], vox_in_range[:, 1], vox_in_range[:, 2])

    # add counts to voxel_samples
    voxel_space[vox_address] = voxel_space[vox_address] + sample_count[in_range]

    return voxel_space


def las_ray_sample(hdf5_path, sample_step, voxel_length):
    start = time.time()
    print('Loading data...')
    # load returns
    returns = pd.read_hdf(hdf5_path, key='las_data', columns=['gps_time', 'x', 'y', 'z', 'classification', 'return_num', 'num_returns'])
    # load trajectory
    traj = pd.read_hdf(hdf5_path, key='las_traj', columns=['traj_x', 'traj_y', 'traj_z', 'distance_from_sensor_m'])

    # concatinate rays
    rays = pd.concat([returns, traj], axis=1)
    # filter by class
    rays = rays[(rays.classification == 2) | (rays.classification == 5)]
    # filter to last returns
    rays = rays[rays.return_num == rays.num_returns]
    rays = rays[['gps_time', 'x', 'y', 'z', 'traj_x', 'traj_y', 'traj_z', 'distance_from_sensor_m']]

    # interpolate rays to ceiling
    z_ceiling = np.max(rays.z).astype(int) + 1
    rays = rays.assign(x_c=(z_ceiling - rays.traj_z) * (rays.traj_x - rays.x) / (rays.traj_z - rays.z) + rays.traj_x)
    rays = rays.assign(y_c=(z_ceiling - rays.traj_z) * (rays.traj_y - rays.y) / (rays.traj_z - rays.z) + rays.traj_y)
    rays = rays.assign(z_c=z_ceiling)

    # p0 = source, p1 = last return
    p0 = np.array(rays.loc[:, ['x_c', 'y_c', 'z_c']])
    p1 = np.array(rays.loc[:, ['x', 'y', 'z']])

    # calculate distance between source and last return
    dist = np.sqrt(np.sum((p1 - p0) ** 2, axis=1))

    # define voxel dimensions
    vox_x_step = voxel_length
    vox_y_step = voxel_length
    vox_z_step = -voxel_length
    vox_step = np.array([vox_x_step, vox_y_step, vox_z_step])

    vox_x0 = np.min(rays.x)
    vox_y0 = np.min(rays.y)
    vox_z0 = z_ceiling
    vox_origin = np.array([vox_x0, vox_y0, vox_z0])

    vox_x_count = int((np.max(rays.x) - vox_x0) / vox_x_step) + 1
    vox_y_count = int((np.max(rays.y) - vox_y0) / vox_y_step) + 1
    vox_z_count = int((np.min(rays.z) - vox_z0) / vox_z_step) + 1
    vox_count = np.array([vox_x_count, vox_y_count, vox_z_count])

    # preallocate voxels
    voxel_samples = np.zeros(vox_count)

    # calc unit-wise xyz step
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
        print(str(ii) + ' of ' + str(int(max_dist/sample_step) + 1))
        # distance from p0 along ray
        t_dist = (ii + offset) * sample_step

        # select rays where t_dist is in range
        in_range = (dist > t_dist)

        # calculate tracer point coords for step
        sample_points = xyz_step[in_range, :] * t_dist[in_range, np.newaxis] + p0[in_range]

        if np.size(sample_points) != 0:
            # add counts to voxel_samples
            voxel_samples = add_points_to_voxels(voxel_samples, vox_origin, vox_step, vox_count, sample_points)

        # advance step
        ii = ii + 1

    # convert voxel counts to path length units [m]
    voxel_samples = voxel_samples * sample_step

    end = time.time()
    print('done in ' + str(end - start) + ' seconds.')

    return voxel_samples


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

sample_length = 10
voxel_length = 20
voxel_samples = las_ray_sample(hdf5_path, sample_length, voxel_length)

voxel_samples[voxel_samples == 0] = np.nan

# voxel_returns
voxel_returns = np.zeros(vox_count)
voxel_returns = add_points_to_voxels(voxel_returns, vox_origin, vox_step, vox_count, p1)

transmission = voxel_returns / voxel_samples


las_ray_sample(las_in, traj_in, hdf5_path)

# voxel grid summarizing of point cloud

grid_l = 1

ray_samples = pd.read_hdf(hdf5_path, key='ray_samples', columns=['x', 'y', 'z'])
x_min = ray_samples

# scrap
# # subsample for preliminary testing
# print("dataset is being subsampled, review code if not desired")
# window = 5  # square window in meters x and y
# x_filter = (rays.x > (np.mean(rays.x) - window/2)) & (rays.x < (np.mean(rays.x) + window/2))
# y_filter = (rays.y > (np.mean(rays.y) - window/2)) & (rays.y < (np.mean(rays.y) + window/2))
# rays = rays[x_filter & y_filter]