import numpy as np
import pandas as pd
import laslib
import time

# las file
las_in = 'C:\\Users\\jas600\\workzone\\data\\las\\19_149_UF.las'
# trajectory file
traj_in = 'C:\\Users\\jas600\\workzone\\data\\las\\19_149_all_traj.txt'
# working hdf5 file
hdf5_path = las_in.replace('.las', '_ray_sampling.hdf5')

def las_ray_sample(las_in, traj_in, hdf5_path):
    # write las to hdf5
    laslib.las_to_hdf5(las_in, hdf5_path)
    # interpolate trajectory
    laslib.las_traj(hdf5_path, traj_in)


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

    # subsample for preliminary testing
    print("dataset is being subsampled, review code if not desired")
    window = 5  # square window in meters x and y
    x_filter = (rays.x > (np.mean(rays.x) - window/2)) & (rays.x < (np.mean(rays.x) + window/2))
    y_filter = (rays.y > (np.mean(rays.y) - window/2)) & (rays.y < (np.mean(rays.y) + window/2))
    rays = rays[x_filter & y_filter]

    # begin algorithm
    # p0 = source, p1 = last return
    p0 = np.array(rays.loc[:, ['x_c', 'y_c', 'z_c']])
    p1 = np.array(rays.loc[:, ['x', 'y', 'z']])

    # calculate distance between source and last return
    dist = np.sqrt(np.sum((p1 - p0) ** 2, axis=1))

    sample_step = .1

    # calc unit-wise xyz step
    xyz_step = (p1 - p0)/dist[:, np.newaxis]

    # random offset seed for each ray sample series
    offset = np.random.random(len(p0))

    # initiate while loop
    ii = 0

    max_dist = np.max(dist)
    # iterate until longest ray length is surpassed

    start = time.time()

    while (ii * sample_step) < max_dist:
        # distance from p0 along ray
        t_dist = (ii + offset) * sample_step

        # select rays where t_dist is in range
        in_range = (dist > t_dist)

        # calculate tracer point coords for step
        sample_points = xyz_step[in_range, :] * t_dist[in_range, np.newaxis] + p0[in_range]

        if np.size(sample_points) != 0:
            # write to file
            tp = pd.DataFrame({'gps_time': rays.gps_time[in_range],
                               'x': sample_points[:, 0],
                               'y': sample_points[:, 1],
                               'z': sample_points[:, 2]})
            tp.to_hdf(hdf5_path, key='ray_samples', mode='r+', format='table', append=bool(ii))

        # advance step
        ii = ii + 1

las_ray_sample(las_in, traj_in, hdf5_path)

# voxel grid summarizing of point cloud

grid_l = 1

ray_samples = pd.read_hdf(hdf5_path, key='ray_samples', columns=['x', 'y', 'z'])
x_min = ray_samples
