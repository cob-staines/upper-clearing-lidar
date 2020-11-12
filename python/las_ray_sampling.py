import numpy as np
import pandas as pd
import laslib
import rastools
import time
import h5py


class VoxelObj(object):
    def __init__(self):
        # voxel object metadata
        self.vox_hdf5 = None
        self.las_in = None
        self.traj_in = None
        self.las_traj_hdf5 = None
        self.las_traj_chunksize = None
        self.return_set = None
        self.drop_class = None
        self.cw_rotation = None
        self.origin = None
        self.max = None
        self.step = None
        self.ncells = None
        self.sample_length = None
        self.sample_dtype = None
        self.return_dtype = None
        self.sample_data = None
        self.return_data = None

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def save(self):
        save_vox_meta(self)


def save_vox_meta(vox):
    with h5py.File(vox.vox_hdf5, 'a') as h5f:
        meta = h5f.create_group("meta")
        meta.create_dataset('las_in', data=vox.las_in)
        meta.create_dataset('traj_in', data=vox.traj_in)
        meta.create_dataset('las_traj_hdf5', data=vox.las_traj_hdf5)
        meta.create_dataset('return_set', data=vox.return_set)
        meta.create_dataset('drop_class', data=vox.drop_class)
        meta.create_dataset('las_traj_chunksize', data=vox.las_traj_chunksize)
        meta.create_dataset('cw_rotation', data=vox.cw_rotation)
        meta.create_dataset('vox_origin', data=vox.origin)
        meta.create_dataset('vox_max', data=vox.max)
        meta.create_dataset('vox_step', data=vox.step)
        meta.create_dataset('vox_ncells', data=vox.ncells)
        meta.create_dataset('vox_sample_length', data=vox.sample_length)


def load_vox_meta(hdf5_path, load_data=False):
    vox = VoxelObj()
    vox.vox_hdf5 = hdf5_path

    with h5py.File(hdf5_path, 'r') as h5f:
        meta = h5f.get('meta')
        vox.las_in = meta.get('las_in')[()]
        vox.traj_in = meta.get('traj_in')[()]
        vox.las_traj_hdf5 = meta.get('las_traj_hdf5')[()]
        vox.return_set = meta.get('return_set')[()]
        vox.drop_class = meta.get('drop_class')[()]
        vox.las_traj_chunksize = meta.get('las_traj_chunksize')[()]
        vox.cw_rotation = meta.get('cw_rotation')[()]
        vox.origin = meta.get('vox_origin')[()]
        vox.max = meta.get('vox_max')[()]
        vox.step = meta.get('vox_step')[()]
        vox.ncells = meta.get('vox_ncells')[()]
        vox.sample_length = meta.get('vox_sample_length')[()]
        vox.sample_dtype = h5f.get('sample_data').dtype
        vox.return_dtype = h5f.get('return_data').dtype
        if load_data:
            vox.sample_data = h5f.get('sample_data')[()]
            vox.return_data = h5f.get('return_data')[()]
    return vox


def z_rotmat(theta):
    if theta is None:
        theta = 0
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])


def utm_to_vox(vox, utm_points, rotation=True):
    if rotation:
        theta = vox.cw_rotation
    else:
        theta = 0
    return (np.matmul(utm_points, z_rotmat(theta)) - vox.origin) / vox.step


def vox_to_utm(vox, vox_points, rotation=True):
    if rotation:
        theta = vox.cw_rotation
    else:
        theta = 0
    return np.matmul(vox_points * vox.step + vox.origin, z_rotmat(-theta))


def add_points_to_voxels(vox, dataset, points):
    # convert to voxel coordinate system
    vox_coords = utm_to_vox(vox, points).astype(int)

    # format
    vox_address = (vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2])

    if dataset == "samples":
        np.add.at(vox.sample_data, vox_address, 1)
    elif dataset == "returns":
        np.add.at(vox.return_data, vox_address, 1)
    else:
        raise Exception("Expected 'samples' or 'returns' for dataset, encountered:" + str(dataset))

    return vox


def interpolate_to_bounding_box(fixed_points, flex_points, bb=None, cw_rotation=0):
    # print('Interpolating rays to bounding box... ', end='')

    if cw_rotation != 0:
        fixed_points = np.matmul(fixed_points, z_rotmat(cw_rotation))
        flex_points = np.matmul(flex_points, z_rotmat(cw_rotation))

    if fixed_points.shape != flex_points.shape:
        raise Exception('fixed_points and flex_points have different shapes!')

    bb_points = flex_points.copy()

    if bb:
        lb = bb[0]
        ub = bb[1]
        if np.any((lb > fixed_points) | (fixed_points > ub)):
            raise Exception('fixed points do not lie within provided bounding box. Process aborted.')
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

    if cw_rotation != 0:
        bb_points = np.matmul(bb_points, z_rotmat(-cw_rotation))

    # print('done.')

    return bb_points


def interpolate_to_z_slice(p1, p0, z_low_utm, z_high_utm):

    def interp_to_z(s1, s0, z_new):
        z_prop = (z_new - s0[:, 2]) / (s1[:, 2] - s0[:, 2])
        x_new = z_prop * (s1[:, 0] - s0[:, 0]) + s0[:, 0]
        y_new = z_prop * (s1[:, 1] - s0[:, 1]) + s0[:, 1]
        return np.array([x_new, y_new, np.full(len(s0), z_new)]).swapaxes(0, 1)


    q_p0 = np.array([(p0[:, 2] >= z_low_utm), (p0[:, 2] <= z_high_utm)]).swapaxes(0, 1)
    q_p1 = np.array([(p1[:, 2] >= z_low_utm), (p1[:, 2] <= z_high_utm)]).swapaxes(0, 1)

    z0 = p0.copy()
    z1 = p1.copy()

    # handle all cases:

    # both points below (neither point meets low criteria)
    set = ~q_p0[:, 0] & ~q_p1[:, 0]
    # set to nan
    z0[set, :] = z1[set, :] = np.nan

    # both points above (neither point meets high criteria, throw out)
    set = ~q_p0[:, 1] & ~q_p1[:, 1]
    # set to nan
    z0[set, :] = z1[set, :] = np.nan

    # one above, one below (interpolate to high and low)
    # p0 below, p1 above
    set = ~q_p0[:, 0] & ~q_p1[:, 1]
    z0[set, :] = interp_to_z(p1[set, :], p0[set, :], z_low_utm)
    z1[set, :] = interp_to_z(p1[set, :], p0[set, :], z_high_utm)
    # p0 above, p1 below
    set = ~q_p0[:, 1] & ~q_p1[:, 0]
    z1[set, :] = interp_to_z(p1[set, :], p0[set, :], z_low_utm)
    z0[set, :] = interp_to_z(p1[set, :], p0[set, :], z_high_utm)

    # both in bounds (pass)
    # set = q_p0[:, 0] & q_p0[:, 1] & q_p1[:, 0] & q_p1[:, 1]

    # one in, one below (interpolate below point to low)
    # p0 in, p1 below
    set = q_p0[:, 0] & q_p0[:, 1] & ~q_p1[:, 0]
    z1[set, :] = interp_to_z(p1[set, :], p0[set, :], z_low_utm)
    # p0 below, p1 in
    set = ~q_p0[:, 0] & q_p1[:, 0] & q_p1[:, 1]
    z0[set, :] = interp_to_z(p1[set, :], p0[set, :], z_low_utm)

    # one in, one above (interpolate above point to high)
    # p0 in, p1 above
    set = q_p0[:, 0] & q_p0[:, 1] & ~q_p1[:, 1]
    z1[set, :] = interp_to_z(p1[set, :], p0[set, :], z_high_utm)
    # p0 above, p1 in
    set = ~q_p0[:, 1] & q_p1[:, 0] & q_p1[:, 1]
    z0[set, :] = interp_to_z(p1[set, :], p0[set, :], z_high_utm)

    # drop nan rows
    z0 = z0[~np.any(np.isnan(z1), axis=1), :]
    z1 = z1[~np.any(np.isnan(z1), axis=1), :]

    return z1, z0


def las_ray_sample_by_z_slice(vox, z_slices, samp_floor_as_returns=True, fail_overflow=False):

    print('----- LAS Ray Sampling -----')

    start = time.time()

    if vox.sample_length > vox.step[0]:
        import warnings
        warnings.warn("vox.sample_length is greater than vox.step, some voxels will be stepped over in sampling. Consider smaller sample_length", UserWarning)

    if fail_overflow:
        # define parameters for overflow testing of sample
        current_max = 0
        current_max_args = (np.array([0]), np.array([0]), np.array([0]))
        if vox.sample_dtype is np.uint8:
            valmax = (2 ** 8 - 1)
        elif vox.sample_dtype is np.uint16:
            valmax = (2 ** 16 - 1)
        elif vox.sample_dtype is np.uint32:
            valmax = (2 ** 32 - 1)
        elif vox.sample_dtype is np.uint64:
            valmax = (2 ** 64 - 1)
        else:
            raise Exception('vox.sample_dtype not recognized, please do the honors of adding to the code')

    print('Loading data descriptors... ', end='')

    # load simple params
    with h5py.File(vox.las_traj_hdf5, 'r') as hf:
        las_time = hf['lasData'][:, 0]
        traj_time = hf['trajData'][:, 0]
        n_rows = len(las_time)
        z_min = np.min(hf['lasData'][:, 3])
        z_max = np.max(hf['lasData'][:, 3])

    # check that gps_times align
    if np.all(las_time != traj_time):
        raise Exception('gps_times do not align between las and traj dfs, process aborted.')
    las_time = None
    traj_time = None

    # calculate n_chunks
    if vox.las_traj_chunksize is None:
        vox.las_traj_chunksize = n_rows
    n_chunks = np.ceil(n_rows / vox.las_traj_chunksize).astype(int)

    if vox.cw_rotation == 0:
        with h5py.File(vox.las_traj_hdf5, 'r') as hf:
            x_min = np.min(hf['lasData'][:, 1])
            x_max = np.max(hf['lasData'][:, 1])
            y_min = np.min(hf['lasData'][:, 2])
            y_max = np.max(hf['lasData'][:, 2])
    else:
        # determine x/y min & max in rotated reference frame (slow, must load and convert all data...)
        x_min = y_min = x_max = y_max = np.nan
        with h5py.File(vox.las_traj_hdf5, 'r') as hf:
            for ii in range(0, n_chunks):
                print('Chunk ' + str(ii) + '... ', end='')

                # chunk start and end
                idx_start = ii * vox.las_traj_chunksize
                if ii != (n_chunks - 1):
                    idx_end = (ii + 1) * vox.las_traj_chunksize
                else:
                    idx_end = n_rows

                pts_rot = np.matmul(hf['lasData'][idx_start:idx_end, 1:3], z_rotmat(vox.cw_rotation)[0:2, 0:2])

                xmin_chunk, ymin_chunk = np.min(pts_rot, axis=0)
                xmax_chunk, ymax_chunk = np.max(pts_rot, axis=0)

                x_min = np.nanmin((x_min, xmin_chunk))
                y_min = np.nanmin((y_min, ymin_chunk))
                x_max = np.nanmax((x_max, xmax_chunk))
                y_max = np.nanmax((y_max, ymax_chunk))

    print('done')

    # define voxel parameters
    vox.origin = np.array([x_min, y_min, z_min])
    vox.max = np.array([x_max, y_max, z_max])
    vox.ncells = np.ceil((vox.max - vox.origin) / vox.step).astype(int)

    z_step = np.ceil(vox.ncells[2] / z_slices).astype(int)

    # memory test of slice size
    m_test = np.zeros((vox.ncells[0], vox.ncells[1], z_step), dtype=vox.sample_dtype)
    m_test = np.zeros((vox.ncells[0], vox.ncells[1], z_step), dtype=vox.return_dtype)
    m_test = None

    # preallocate voxel hdf5
    with h5py.File(vox.vox_hdf5, mode='w') as hf:
        hf.create_dataset('sample_data', dtype=vox.sample_dtype, shape=vox.ncells, chunks=True, compression='gzip')
        hf.create_dataset('return_data', dtype=vox.return_dtype, shape=vox.ncells, chunks=True, compression='gzip')

    # loop over las_traj chunks
    for ii in range(0, n_chunks):

        print('Chunk ' + str(ii + 1) + ' of ' + str(n_chunks))

        # chunk start and end
        idx_start = ii * vox.las_traj_chunksize
        if ii != (n_chunks - 1):
            idx_end = (ii + 1) * vox.las_traj_chunksize
        else:
            idx_end = n_rows

        print('Loading data chunk... ', end='')
        with h5py.File(vox.las_traj_hdf5, 'r') as hf:
            ray_1 = hf['lasData'][idx_start:idx_end, 1:4]
            ray_0 = hf['trajData'][idx_start:idx_end, 1:4]
        print('done')

        # transform returns to vox coords
        rtns_vox = utm_to_vox(vox, ray_1).astype(int)

        # interpolate sensor to bounding box
        ray_bb = interpolate_to_bounding_box(ray_1, ray_0)

        # loop over z_slices
        for zz in range(0, z_slices):
            print('slice ' + str(zz + 1) + ' of ' + str(z_slices))
            z_low = zz * z_step
            if zz != (z_slices - 1):
                z_high = (zz + 1) * z_step
            else:
                z_high = vox.ncells[2]

            z_cells = z_high - z_low
            z_low_utm = z_low * vox.step[2] + vox.origin[2]
            z_high_utm = z_high * vox.step[2] + vox.origin[2]

            # cycle through all las_traj data chunks
            z1, z0 = interpolate_to_z_slice(ray_1, ray_bb, z_low_utm, z_high_utm)

            # load slices from vox
            with h5py.File(vox.vox_hdf5, mode='r') as hf:
                sample_slice = hf['sample_data'][:, :, z_low:z_high]
                return_slice = hf['return_data'][:, :, z_low:z_high]


            # add valid retruns to return slice
            rtns_valid = rtns_vox[(rtns_vox[:, 2] >= z_low) & (rtns_vox[:, 2] < z_high), :]
            # correct for z_slice offset
            rtns_valid[:, 2] = rtns_valid[:, 2] - z_low
            # format
            rtns_address = (rtns_valid[:, 0], rtns_valid[:, 1], rtns_valid[:, 2])
            # add points
            np.add.at(return_slice, rtns_address, 1)


            # calculate length of ray
            dist = np.sqrt(np.sum((z1 - z0) ** 2, axis=1))

            # calc unit step along ray in x, y, z dims (avoid edge cases where dist == 0)
            xyz_step = np.full([len(dist), 3], np.nan)
            xyz_step[dist > 0, :] = (z1[dist > 0] - z0[dist > 0]) / dist[dist > 0, np.newaxis]

            # random offset for each ray sample series
            offset = np.random.random(len(z1))

            # iterate until longest ray length is surpassed
            max_step = np.ceil(np.max(dist) / vox.sample_length).astype(int)
            for jj in range(0, max_step):
                print(str(jj + 1) + ' of ' + str(max_step))
                # distance from p0 along ray
                sample_dist = (jj + offset) * vox.sample_length

                # select rays where t_dist is in range
                in_range = (dist > sample_dist)

                # calculate tracer point coords for step
                sample_points = xyz_step[in_range, :] * sample_dist[in_range, np.newaxis] + z0[in_range]

                if np.size(sample_points) != 0:
                    # transform samples to vox coords
                    samps_vox = utm_to_vox(vox, sample_points).astype(int)

                    # correct for z_slice offset
                    samps_vox[:, 2] = samps_vox[:, 2] - z_low

                    # format
                    samps_address = (samps_vox[:, 0], samps_vox[:, 1], samps_vox[:, 2])

                    # add points
                    np.add.at(sample_slice, samps_address, 1)

                if fail_overflow:
                    past_max = current_max
                    past_max_args = current_max_args

                    current_max = np.max(sample_slice)
                    current_max_args = np.where(sample_slice == current_max)

                    if np.any(sample_slice[past_max_args] < past_max):
                        raise Exception('Overflow observed in past_max_args decrease, process aborted.\npast_max: ' + str(past_max) + '\ncurrent_max: ' + str(current_max))
                    if current_max == valmax:
                        raise Exception('Overflow expected based on reaching maximum value, process aborted')

            # save slice to file
            with h5py.File(vox.vox_hdf5, mode='r+') as hf:
                hf['sample_data'][:, :, z_low:z_high] = sample_slice
                hf['return_data'][:, :, z_low:z_high] = return_slice

    if samp_floor_as_returns:
        # set minimum sample count equal to return count
        with h5py.File(vox.vox_hdf5, mode='r+') as hf:
            fix = hf['sample_data'][()] < hf['return_data'][()]
            hf['sample_data'][fix] = hf['return_data'][fix]

    end = time.time()
    print('Ray sampling done in ' + str(end - start) + ' seconds.')

    return vox


def nb_sample_sum_explicit(rays, path_samples, path_returns, n_samples, agg_sample_length, prior, ray_iterations, commentation=False):
    if commentation:
        print('Aggregating samples over each ray...')

    returns_mean = np.full(len(path_samples), np.nan)
    returns_med = np.full(len(path_samples), np.nan)
    returns_var = np.full(len(path_samples), np.nan)

    for ii in range(0, len(path_samples)):
        kk = path_returns[ii, 0:n_samples[ii]]
        nn = path_samples[ii, 0:n_samples[ii]]

        # posterior hyperparameters
        post_a = kk + prior[0]
        post_b = 1 - 1 / (1 + prior[1] + nn)

        nb_samples = np.full([ray_iterations, n_samples[ii]], 0)
        for jj in range(0, n_samples[ii] - 1):
            nb_samples[:, jj] = np.random.negative_binomial(post_a[jj], post_b[jj], ray_iterations)

        # correct for agg sample length
        nb_samples = nb_samples * agg_sample_length

        # sum modeled values along ray
        return_sums = np.sum(nb_samples, axis=1)

        #calculate stats
        returns_mean[ii] = np.mean(return_sums)
        returns_med[ii] = np.median(return_sums)
        returns_var[ii] = np.var(return_sums)

        if commentation:
            print(str(ii + 1) + ' of ' + str(len(path_samples)) + ' rays')

    rays = rays.assign(returns_mean=returns_mean)
    rays = rays.assign(returns_median=returns_med)
    rays = rays.assign(returns_var=returns_var)

    return rays


def nb_sample_sum_combined(rays, path_samples, path_returns, n_samples, agg_sample_length, prior, ray_iterations, commentation=False):
    if commentation:
        print('Aggregating samples over each ray')

    # preallocate
    returns_mean = np.full(len(path_samples), np.nan)
    returns_med = np.full(len(path_samples), np.nan)
    returns_std = np.full(len(path_samples), np.nan)

    for ii in range(0, len(path_samples)):
        kk = path_returns[ii, 0:n_samples[ii]]
        nn = path_samples[ii, 0:n_samples[ii]]

        # posterior hyperparameters
        post_a = kk + prior[0]
        post_b = 1 - 1 / (1 + prior[1] + nn)

        unique_p = np.unique(post_b)
        nb_samples = np.full([ray_iterations, len(unique_p)], 0)
        for pp in range(0, len(unique_p)):
            # sum alphas with corresponding probabilities (p)
            alpha = np.sum(post_a[post_b == unique_p[pp]])
            nb_samples[:, pp] = np.random.negative_binomial(alpha, unique_p[pp], ray_iterations)

        # correct for agg sample length
        nb_samples = nb_samples * agg_sample_length

        # sum modeled values along ray
        return_sums = np.sum(nb_samples, axis=1)

        # calculate stats
        returns_mean[ii] = np.mean(return_sums)
        returns_med[ii] = np.median(return_sums)
        returns_std[ii] = np.std(return_sums)

        if commentation:
            print(str(ii + 1) + ' of ' + str(len(path_samples)) + ' rays')

    rays = rays.assign(returns_mean=returns_mean)
    rays = rays.assign(returns_median=returns_med)
    rays = rays.assign(returns_std=returns_std)

    return rays


def nb_sample_sum_lookup(rays, path_samples, path_returns, n_samples, agg_sample_length, prior, ray_iterations, commentation=False):
    if commentation:
        print('Aggregating samples over each ray')

    # preallocate
    returns_mean = np.full(len(path_samples), np.nan)
    returns_med = np.full(len(path_samples), np.nan)
    returns_std = np.full(len(path_samples), np.nan)

    # lookup for unique pairs of post_a and post_b
    post_a = prior[0] + path_returns
    post_b = 1 - 1 / (1 + prior[1] + path_samples)

    if commentation:
        print('Building dictionary...', end='')

    all_par = np.array((post_a.reshape(post_a.size), post_b.reshape(post_b.size))).swapaxes(0, 1)
    unique_par = np.unique(all_par, axis=0)
    unique_par = unique_par[~np.any(np.isnan(unique_par), axis=1)]
    unique_par = list(zip(unique_par[:, 0], unique_par[:, 1]))

    # calculate all possible values
    lookup = {}
    for dd in range(0, len(unique_par)):
        lookup[unique_par[dd]] = np.random.negative_binomial(unique_par[dd][0], unique_par[dd][1], ray_iterations)
    if commentation:
        print('done')

    # for each ray
    for ii in range(0, len(path_samples)):
        aa = post_a[ii, 0:n_samples[ii]]
        bb = post_b[ii, 0:n_samples[ii]]

        keys = list(zip(aa, bb))

        nb_samples = np.full([ray_iterations, n_samples[ii]], 0)
        for kk in range(0, n_samples[ii]):
            nb_samples[:, kk] = lookup[keys[kk]]

        # correct for agg sample length
        nb_samples = nb_samples * agg_sample_length

        # sum samples along ray
        return_sums = np.sum(nb_samples, axis=1)

        returns_mean[ii] = np.mean(return_sums)
        returns_med[ii] = np.median(return_sums)
        returns_std[ii] = np.std(return_sums)

        if commentation:
            print(str(ii + 1) + ' of ' + str(len(path_samples)) + ' rays')

    rays = rays.assign(returns_mean=returns_mean)
    rays = rays.assign(returns_median=returns_med)
    rays = rays.assign(returns_std=returns_std)

    return rays


def linear_return_model(rays, path_samples, path_returns, agg_sample_length, prior, commentation=False):
    if commentation:
        print('Aggregating samples over each ray')

    # handle case for zero-samples...
    bref = path_returns / (path_samples + agg_sample_length)  # should be vox.sample_length, fix later
    returns_mean = np.nansum(bref, axis=1) * agg_sample_length
    returns_std = np.sqrt(prior / np.nansum(path_samples + agg_sample_length, axis=1))

    rays = rays.assign(returns_mean=returns_mean)
    rays = rays.assign(returns_median=np.nan)
    rays = rays.assign(returns_std=returns_std)

    return rays


def beta_clt(rays, path_samples, path_returns, vox_sample_length, agg_sample_length, prior):

    kk = path_returns
    nn = path_samples * vox_sample_length / agg_sample_length

    post_a = kk + prior[0]
    post_b = nn + kk + prior[1]

    # normal approximation of sum
    returns_mean = np.nansum(post_a/(post_a + post_b), axis=1)
    returns_std = np.sqrt(np.nansum(post_a * post_b / ((post_a + post_b) ** 2 * (post_a + post_b + 1)), axis=1))

    rays = rays.assign(returns_mean=returns_mean)
    rays = rays.assign(returns_median=np.nan)
    rays = rays.assign(returns_std=returns_std)

    return rays


def aggregate_voxels_over_rays(vox, rays, agg_sample_length, prior, ray_iterations, method, commentation=False):

    # pull points
    p0 = rays.loc[:, ['x0', 'y0', 'z0']].values
    p1 = rays.loc[:, ['x1', 'y1', 'z1']].values


    # calculate distance between ray start (ground) and end (sky)
    dist = np.sqrt(np.sum((p1 - p0) ** 2, axis=1))
    rays = rays.assign(path_length=dist)

    # calc unit step along ray in x, y, z dims
    xyz_step = (p1 - p0) / dist[:, np.newaxis]

    # random offset for each ray sample series
    offset = np.random.random(len(p0))

    # calculate number of samples
    n_samples = ((dist - offset) / agg_sample_length).astype(int)
    max_steps = np.max(n_samples)

    # preallocate aggregate lists
    path_samples = np.full([len(p0), max_steps], np.nan)  # precision could be lower if used -1 or...
    path_returns = np.full([len(p0), max_steps], np.nan)

    # for each sample step
    if commentation:
        print('Sampling voxels', end='')
        cur_cent = 0
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

        if commentation:
            past_cent = cur_cent
            cur_cent = int(100 * (ii + 1) / max_steps)
            if cur_cent > past_cent:
                for jj in range(0, cur_cent - past_cent):
                    if (past_cent + jj + 1) % 10 == 0:
                        print(str(past_cent + jj + 1), end='')
                    else:
                        print('.', end='')

    if commentation:
        print(' -- Calculating returns... ', end='')

    rays_out = beta_clt(rays, path_samples, path_returns, vox.sample_length, agg_sample_length, prior)

    # if method == 'nb_lookup':
    #     # correct voxel sample distance with vox sample_length
    #     path_samples = path_samples * vox.sample_length
    #     rays_out = nb_sample_sum_lookup(rays, path_samples, path_returns, n_samples, agg_sample_length, prior, ray_iterations, commentation)
    # elif method == 'nb_combined':
    #     # correct voxel sample distance with vox sample_length
    #     path_samples = path_samples * vox.sample_length
    #     rays_out = nb_sample_sum_combined(rays, path_samples, path_returns, n_samples, agg_sample_length, prior, ray_iterations, commentation)
    # elif method == 'nb_explicit':
    #     # correct voxel sample distance with vox sample_length
    #     path_samples = path_samples * vox.sample_length
    #     rays_out = nb_sample_sum_explicit(rays, path_samples, path_returns, n_samples, agg_sample_length, prior, ray_iterations, commentation)
    # elif method == 'linear':
    #     # correct voxel sample distance with vox sample_length
    #     path_samples = path_samples * vox.sample_length
    #     rays_out = linear_return_model(rays, path_samples, path_returns, agg_sample_length, prior, commentation)
    # elif method == 'beta':
    #     if commentation:
    #         print(' -- Calculating returns... ', end='')
    #     rays_out = beta_clt(rays, path_samples, path_returns, vox.sample_length, agg_sample_length, prior)
    # else:
    #     raise Exception('Aggregation method "' + method + '" not found, process aborted.')

    return rays_out


def agg_chunk(chunksize, vox, rays, agg_sample_length, prior, ray_iterations, method, commentation=False):

    n_rays = len(rays)

    # allow null case
    if chunksize is None:
        chunksize = n_rays

    n_chunks = np.ceil(n_rays / chunksize).astype(int)

    if n_chunks > 1:
        print("\n\t", end='')

    # chunk las ray_sample
    for ii in range(0, n_chunks):
        if n_chunks > 1:
            print('Chunk ' + str(ii + 1) + ' of ' + str(n_chunks) + ': ', end='')

        # chunk start and end
        idx_start = ii * chunksize
        if ii != (n_chunks - 1):
            idx_end = (ii + 1) * chunksize
        else:
            idx_end = n_rays

        chunk_rays_in = rays.iloc[idx_start:idx_end, ]

        set_pts = np.concatenate((chunk_rays_in.loc[:, ['x0', 'y0', 'z0']].values, chunk_rays_in.loc[:, ['x1', 'y1', 'z1']].values), axis=0)
        vox_sub = subset_vox(set_pts, vox, 0)

        chunk_rays_out = aggregate_voxels_over_rays(vox_sub, chunk_rays_in, agg_sample_length, prior, ray_iterations, method, commentation)

        if ii == 0:
            all_rays_out = chunk_rays_out
        else:
            all_rays_out = pd.concat([all_rays_out, chunk_rays_out])

        print("done\n\t", end='')

    return all_rays_out


def dem_to_points(dem_in, mask_in=None):

    if mask_in is None:
        ddict = {'z_m': dem_in}
    else:
        ddict = {'z_m': dem_in,
                 'dmask': mask_in}

    # load data
    pts = rastools.pd_sample_raster_gdal(ddict)

    if mask_in is not None:
        # drop pts where mask is nan
        pts = pts.loc[~np.isnan(pts.dmask), :].drop('dmask', axis='columns')

    # clean up
    pts = pts.reset_index().rename(columns={'index': 'id'})

    # drop z_m nan values
    pts = pts.loc[~np.isnan(pts.z_m), :]

    return pts


def points_to_angle_rays(pts, vox, phi, theta, min_dist=0, max_dist=100):
    """
    :param points: pd dataframe of points with columns [id, x_coord, y_coord, and z_m]
    :param phi: ray angle from nadir in radians
    :param theta: angle cw from N in radians
    :param vox: voxel object (used for bounding box only)
    :return: rays to be sampled
    """
    rays = pts.copy()
    origin = pts.loc[:, ['x_coord', 'y_coord', 'z_m']].values

    # displacement vector of start ray
    rr0 = min_dist
    x0 = rr0 * np.sin(theta) * np.sin(phi)
    y0 = rr0 * np.cos(theta) * np.sin(phi)
    z0 = rr0 * np.cos(phi)

    # displacement vector of end ray
    rr1 = max_dist
    x1 = rr1 * np.sin(theta) * np.sin(phi)
    y1 = rr1 * np.cos(theta) * np.sin(phi)
    z1 = rr1 * np.cos(phi)

    p0 = np.array([x0, y0, z0]) + origin
    p1 = np.array([x1, y1, z1]) + origin

    # interpolate p1 to bounding box
    p1_bb = interpolate_to_bounding_box(p0, p1, bb=[vox.origin, vox.max], cw_rotation=vox.cw_rotation)
    rays = rays.assign(x0=p0[:, 0], y0=p0[:, 1], z0=p0[:, 2])
    rays = rays.assign(x1=p1_bb[:, 0], y1=p1_bb[:, 1], z1=p1_bb[:, 2])

    return rays


def ray_stats_to_dem(rays, dem_in, file_out):
    # load raster dem
    ras = rastools.raster_load(dem_in)

    # pull points
    xy_index = (rays.y_index.values, rays.x_index.values)
    shape = ras.data.shape

    # preallocate data
    ras.data = []
    ras.data.append(np.full(shape, np.nan))
    ras.data.append(np.full(shape, np.nan))
    ras.data.append(np.full(shape, np.nan))

    # store data
    ras.data[0][xy_index] = rays.returns_mean
    ras.data[0][np.isnan(ras.data[0])] = ras.no_data

    ras.data[1][xy_index] = rays.returns_median
    ras.data[1][np.isnan(ras.data[1])] = ras.no_data

    ras.data[2][xy_index] = rays.returns_std
    ras.data[2][np.isnan(ras.data[2])] = ras.no_data

    ras.band_count = 3

    rastools.raster_save(ras, file_out)

    return ras


def hemi_vectors(img_size, max_phi):

    # convert img index to phi/theta
    img_origin = (img_size - 1) / 2
    # cell index
    template = np.full([img_size, img_size], True)
    index = np.where(template)

    rays = pd.DataFrame({'x_index': index[0],
                         'y_index': index[1]})
    # calculate phi and theta
    rays = rays.assign(phi=np.sqrt((rays.x_index - img_origin) ** 2 + (rays.y_index - img_origin) ** 2) * max_phi / img_origin)
    rays = rays.assign(theta=np.arctan2((rays.x_index - img_origin), (rays.y_index - img_origin)) + np.pi)

    # remove rays which exceed max_phi (beyond image horizon)
    rays = rays[rays.phi <= max_phi]

    return rays


def point_to_hemi_rays(origin, vox, img_size, max_phi=np.pi/2, max_dist=50, min_dist=0):

    rays = hemi_vectors(img_size, max_phi)

    # calculate utm coords of point at r = max_dist along ray
    rr0 = min_dist
    x0 = rr0 * np.sin(rays.theta) * np.sin(rays.phi)
    y0 = rr0 * np.cos(rays.theta) * np.sin(rays.phi)
    z0 = rr0 * np.cos(rays.phi)

    rr1 = max_dist
    x1 = rr1 * np.sin(rays.theta) * np.sin(rays.phi)
    y1 = rr1 * np.cos(rays.theta) * np.sin(rays.phi)
    z1 = rr1 * np.cos(rays.phi)

    p0 = np.swapaxes(np.array([x0, y0, z0]), 0, 1) + origin
    p1 = np.swapaxes(np.array([x1, y1, z1]), 0, 1) + origin

    # interpolate p1 to bounding box of vox
    p1_bb = interpolate_to_bounding_box(p0, p1, bb=[vox.origin, vox.max], cw_rotation=vox.cw_rotation)
    rays = rays.assign(x0=p0[:, 0], y0=p0[:, 1], z0=p0[:, 2])
    rays = rays.assign(x1=p1_bb[:, 0], y1=p1_bb[:, 1], z1=p1_bb[:, 2])

    return rays


def hemi_rays_to_img(rays_out, img_path, img_size, area_factor):
    import imageio

    rays_out = rays_out.assign(transmittance=np.exp(-1 * area_factor * rays_out.returns_median))
    template = np.full([img_size, img_size], 1.0)
    template[(rays_out.y_index.values, rays_out.x_index.values)] = rays_out.transmittance

    img = np.rint(template * 255).astype(np.uint8)
    imageio.imsave(img_path, img)


def las_to_vox(vox, z_slices, run_las_traj=True, fail_overflow=False):
    if run_las_traj:
        # interpolate trajectory
        laslib.las_traj(vox.las_in, vox.traj_in, vox.las_traj_hdf5, vox.las_traj_chunksize, vox.return_set, vox.drop_class)

    # sample voxel space from las_traj hdf5
    vox = las_ray_sample_by_z_slice(vox, z_slices, fail_overflow)
    vox.save()

    return vox


def subset_vox(pts, vox, buffer):

    # inherit non-spatial attributes
    vox_sub = VoxelObj()
    vox_sub.las_in = vox.las_in
    vox_sub.traj_in = vox.traj_in
    vox_sub.las_traj_hdf5 = vox.las_traj_hdf5
    vox_sub.las_traj_chunksize = vox.las_traj_chunksize
    vox_sub.vox_hdf5 = vox.vox_hdf5
    vox_sub.sample_dtype = vox.sample_dtype
    vox_sub.return_dtype = vox.return_dtype
    vox_sub.return_set = vox.return_set
    vox_sub.drop_class = vox.drop_class
    vox_sub.cw_rotation = vox.cw_rotation
    vox_sub.sample_length = vox.sample_length
    vox_sub.step = vox.step

    # convert pts to rotated utm if needed
    if vox.cw_rotation != 0:
        pts = np.matmul(pts, z_rotmat(vox.cw_rotation))

    # extent of data in rotated utm
    pnt_min = np.array([np.min(pts[:, 0]),  np.min(pts[:, 1]), np.min(pts[:, 2])])
    pnt_max = np.array([np.max(pts[:, 0]), np.max(pts[:, 1]), np.max(pts[:, 2])])

    # extent of data with buffer in rotated utm
    buff_min = pnt_min - buffer
    buff_max = pnt_max + buffer

    # convert to voxel space
    buff_min_vox = np.floor(utm_to_vox(vox, buff_min, rotation=False)).astype(int)
    buff_max_vox = np.ceil(utm_to_vox(vox, buff_max, rotation=False)).astype(int)

    # calculate sub_vox min/max
    vox_sub_min = np.max(np.array([buff_min_vox, [0, 0, 0]]), axis=0)
    vox_sub_max = np.min(np.array([buff_max_vox, vox.ncells]), axis=0)

    vox_sub.origin = vox_to_utm(vox, vox_sub_min, rotation=False)
    vox_sub.max = vox_to_utm(vox, vox_sub_max, rotation=False)
    vox_sub.ncells = vox_sub_max - vox_sub_min



    with h5py.File(vox_sub.vox_hdf5, mode='r') as hf:
        vox_sub.sample_data = hf['sample_data'][vox_sub_min[0]:vox_sub_max[0], vox_sub_min[1]:vox_sub_max[1], vox_sub_min[2]:vox_sub_max[2]]
        vox_sub.return_data = hf['return_data'][vox_sub_min[0]:vox_sub_max[0], vox_sub_min[1]:vox_sub_max[1], vox_sub_min[2]:vox_sub_max[2]]

    return vox_sub


class RaySampleHemiMetaObj(object):
    def __init__(self):
        # preload metadata
        self.id = None
        self.file_name = None
        self.file_dir = None
        self.origin = None
        self.ray_sample_length = None
        self.ray_iterations = None
        self.max_phi_rad = None
        self.max_distance = None
        self.min_distance = None
        self.img_size = None
        self.prior = None
        self.agg_method = None


class RaySampleGridMetaObj(object):
    def __init__(self):
        # preload metadata
        self.id = None
        self.file_name = None
        self.file_dir = None
        self.src_ras_file = None
        self.mask_file = None
        self.ray_sample_length = None
        self.ray_iterations = None
        self.phi = None
        self.theta = None
        self.max_distance = None
        self.min_distance = None
        self.prior = None
        self.agg_method = None


def rs_hemigen(rshmeta, vox, initial_index=0):
    import os
    import tifffile as tiff

    tot_time = time.time()

    # handle case with only one output
    if rshmeta.origin.shape.__len__() == 1:
        rshmeta.origin = np.array([rshmeta.origin])
    if type(rshmeta.file_name) == str:
        rshmeta.file_dir = [rshmeta.file_dir]

    # QC: ensure origins and file_names have same length
    if rshmeta.origin.shape[0] != rshmeta.file_name.__len__():
        raise Exception('origin_coords and img_out_path have different lengths, execution halted.')

    rshm = pd.DataFrame({"id": rshmeta.id,
                        "file_name": rshmeta.file_name,
                        "file_dir": rshmeta.file_dir,
                        "x_utm11n": rshmeta.origin[:, 0],
                        "y_utm11n": rshmeta.origin[:, 1],
                        "elevation_m": rshmeta.origin[:, 2],
                        "src_las_file": vox.las_in,
                        "vox_step": vox.step[0],
                        "vox_sample_length": vox.sample_length,
                        "src_return_set": vox.return_set,
                        "src_drop_class": vox.drop_class,
                        "ray_sample_length": rshmeta.ray_sample_length,
                        "ray_iterations": rshmeta.ray_iterations,
                        "img_size_px": rshmeta.img_size,
                        "max_phi_rad": rshmeta.max_phi_rad,
                        "min_distance_m": rshmeta.min_distance,
                        "max_distance_m": rshmeta.max_distance,
                        "agg_method": rshmeta.agg_method,
                        "prior": str(rshmeta.prior),
                        "created_datetime": None,
                        "computation_time_s": None})

    # resent index in case of rollover indexing
    rshm = rshm.reset_index(drop=True)

    # preallocate log file
    log_path = rshmeta.file_dir + "rshmetalog.csv"
    if not os.path.exists(log_path):
        with open(log_path, mode='w', encoding='utf-8') as log:
            log.write(",".join(rshm.columns) + '\n')
        log.close()

    # export phi_theta_lookup of vectors in grid
    vector_set = hemi_vectors(rshmeta.img_size, rshmeta.max_phi_rad)
    vector_set.to_csv(rshm.file_dir[0] + "phi_theta_lookup.csv", index=False)

    # subset voxel space
    vox_sub = subset_vox(rshm.loc[:, ['x_utm11n', 'y_utm11n', 'elevation_m']].values, vox, rshmeta.max_distance)

    for ii in range(initial_index, len(rshm)):
        print(str(ii + 1) + " of " + str(rshmeta.origin.shape[0]) + ': ', end='')
        it_time = time.time()

        origin = (rshm.x_utm11n[ii], rshm.y_utm11n[ii], rshm.elevation_m[ii])
        # calculate rays
        rays_in = point_to_hemi_rays(origin, vox_sub, rshmeta.img_size, rshmeta.max_phi_rad, max_dist=rshmeta.max_distance, min_dist=rshmeta.min_distance)

        # sample rays
        # rays = rays_in
        # agg_sample_length = rshmeta.ray_sample_length
        # prior = rshmeta.prior
        # ray_iterations = rshmeta.ray_iterations
        # commentation = True
        # method = rshmeta.agg_method
        rays_out = aggregate_voxels_over_rays(vox_sub, rays_in, rshmeta.ray_sample_length, rshmeta.prior, rshmeta.ray_iterations, rshmeta.agg_method, commentation=True)

        # format to image
        template = np.full((rshmeta.img_size, rshmeta.img_size, 2), np.nan)
        template[(rays_out.y_index.values, rays_out.x_index.values, 0)] = rays_out.returns_mean
        template[(rays_out.y_index.values, rays_out.x_index.values, 1)] = rays_out.returns_std
        # write image
        tiff.imsave(rshm.file_dir.iloc[ii] + rshm.file_name.iloc[ii], template)

        # log meta
        rshm.loc[ii, "created_datetime"] = time.strftime('%Y-%m-%d %H:%M:%S')
        rshm.loc[ii, "computation_time_s"] = int(time.time() - it_time)

        # write to log file
        rshm.iloc[ii:ii + 1].to_csv(log_path, encoding='utf-8', mode='a', header=False, index=False)

        print("done in " + str(rshm.computation_time_s[ii]) + " seconds")
    print("-------- Ray Sample Hemigen completed--------")
    print(str(rshmeta.origin.shape[0] - initial_index) + " images generated in " + str(int(time.time() - tot_time)) + " seconds")
    return rshm

def rs_gridgen(rsgmeta, vox, initial_index=0):
    import os

    tot_time = time.time()

    # handle case with only one output
    if isinstance(rsgmeta.phi, (float, int)):
        rsgmeta.phi = np.array([rsgmeta.phi])
    if isinstance(rsgmeta.theta, (float, int)):
        rsgmeta.theta = np.array([rsgmeta.theta])
    if isinstance(rsgmeta.file_name, str):
        rsgmeta.file_dir = [rsgmeta.file_dir]

    # QC: ensure origins and file_names have same length
    if len(rsgmeta.phi) != len(rsgmeta.theta):
        raise Exception('phi and theta have different lengths, execution halted.')
    if len(rsgmeta.phi) != len(rsgmeta.file_name):
        raise Exception('phi and file_name have different lengths, execution halted.')


    rsgm = pd.DataFrame({"id": rsgmeta.id,
                        "file_name": rsgmeta.file_name,
                        "file_dir": rsgmeta.file_dir,
                        "phi": rsgmeta.phi,
                        "theta": rsgmeta.theta,
                        "src_las_file": vox.las_in,
                        "src_ras_file": rsgmeta.src_ras_file,
                        "mask_file": rsgmeta.mask_file,
                        "vox_step": vox.step[0],
                        "vox_sample_length": vox.sample_length,
                        "src_return_set": vox.return_set,
                        "src_drop_class": vox.drop_class,
                        "ray_sample_length": rsgmeta.ray_sample_length,
                        "ray_iterations": rsgmeta.ray_iterations,
                        "min_distance_m": rsgmeta.min_distance,
                        "max_distance_m": rsgmeta.max_distance,
                        "agg_method": rsgmeta.agg_method,
                        "prior": str(rsgmeta.prior),
                        "created_datetime": None,
                        "computation_time_s": None})

    # resent indexing in case of rollover indexing
    rsgm = rsgm.reset_index(drop=True)

    # preallocate log file
    log_path = rsgmeta.file_dir + "rsgmetalog.csv"
    if not os.path.exists(log_path):
        with open(log_path, mode='w', encoding='utf-8') as log:
            log.write(",".join(rsgm.columns) + '\n')
        log.close()

    # load dem as points
    points_in = dem_to_points(rsgmeta.src_ras_file, rsgmeta.mask_file)

    for ii in range(initial_index, len(rsgm)):
        print(str(ii + 1) + " of " + str(len(rsgm)) + ': ', end='')

        it_time = time.time()

        # calculate rays
        # dem_in = rsgmeta.src_ras_file
        # phi = rsgm.phi[ii]
        # theta = rsgm.theta[ii]
        # mask_in = rsgmeta.mask_file
        # min_dist = rsgmeta.min_distance
        # max_dist = rsgmeta.max_distance
        # calculate rays
        rays_in = points_to_angle_rays(points_in, vox, rsgm.phi[ii], rsgm.theta[ii], rsgmeta.min_distance, rsgmeta.max_distance)

        # sample rays
        # chunksize = 100000
        # rays = rays_in
        # agg_sample_length = rsgmeta.ray_sample_length
        # prior = rsgmeta.prior
        # ray_iterations = rsgmeta.ray_iterations
        # commentation = True
        # method = rsgmeta.agg_method
        rays_out = agg_chunk(100000, vox, rays_in, rsgmeta.ray_sample_length, rsgmeta.prior, rsgmeta.ray_iterations, rsgmeta.agg_method, commentation=True)

        # format to image
        ras = rastools.raster_load(rsgmeta.src_ras_file)
        ras.band_count = 2
        mean_data = np.full((ras.rows, ras.cols), ras.no_data)
        mean_data[(rays_out.y_index.values, rays_out.x_index.values)] = rays_out.returns_mean
        std_data = np.full((ras.rows, ras.cols), ras.no_data)
        std_data[(rays_out.y_index.values, rays_out.x_index.values)] = rays_out.returns_std
        ras.data = [mean_data, std_data]
        ras_clipped = rastools.clip_raster_to_valid_extent(ras)
        # write image
        rastools.raster_save(ras_clipped, rsgm.file_dir.iloc[ii] + rsgm.file_name.iloc[ii], data_format="float32")

        # log meta
        rsgm.loc[ii, "created_datetime"] = time.strftime('%Y-%m-%d %H:%M:%S')
        rsgm.loc[ii, "computation_time_s"] = int(time.time() - it_time)

        # write to log file
        rsgm.iloc[ii:ii + 1].to_csv(log_path, encoding='utf-8', mode='a', header=False, index=False)

        print("done in " + str(rsgm.computation_time_s[ii]) + " seconds")
    print("-------- Ray Sample Hemigen completed--------")
    print(str(len(rsgm) - initial_index) + " images generated in " + str(int(time.time() - tot_time)) + " seconds")
    return rsgm


# def rsm_chunk(args):
#     import tifffile as tiff
#     rshm = args[0]
#     log_path = args[1]
#     vox = args[2]
#
#     # initialize empty log file
#     with open(log_path, 'w') as input_file:
#         pass
#
#     for ii in range(np.min(rshm.id), np.max(rshm.id) + 1):
#         it_time = time.time()
#
#         origin = (rshm.x_utm11n[ii], rshm.y_utm11n[ii], rshm.elevation_m[ii])
#         # calculate rays
#         rays_in = point_to_hemi_rays(origin, rshm.img_size_px[ii], vox, max_phi=rshm.max_phi_rad[ii], max_dist=rshm.max_distance_m[ii], min_dist=rshm.min_distance_m[ii])
#
#         # sample rays
#         rays_out = aggregate_voxels_over_rays(vox, rays_in, rshm.ray_sample_length[ii], [rshm.prior_a[ii], rshm.prior_b[ii]], rshm.ray_iterations[ii])
#
#         output = rays_out.loc[:, ['x_index', 'y_index', 'phi', 'theta', 'returns_mean', 'returns_median', 'returns_std']]
#
#         # format to image
#         template = np.full((rshm.img_size_px, rshm.img_size_px, 3), np.nan)
#         template[(rays_out.y_index.values, rays_out.x_index.values, 0)] = rays_out.returns_mean
#         template[(rays_out.y_index.values, rays_out.x_index.values, 1)] = rays_out.returns_median
#         template[(rays_out.y_index.values, rays_out.x_index.values, 2)] = rays_out.returns_std
#
#
#         tiff.imsave(rshm.file_dir.iloc[ii] + rshm.file_name.iloc[ii], template)
#
#         # log meta
#         rshm.loc[ii, "created_datetime"] = time.strftime('%Y-%m-%d %H:%M:%S')
#         rshm.loc[ii, "computation_time_s"] = int(time.time() - it_time)
#
#         # write to log file
#         rshm.iloc[ii:ii + 1].to_csv(log_path, encoding='utf-8', mode='a', header=False, index=False)
#
#         print(str(ii + 1) + " of " + str(len(rshm)) + " complete: " + str(rshm.computation_time_s[ii]) + " seconds")
#
# def rs_hemigen_multiproc(rshmeta, vox, initial_index=0, n_cores=4):
#     import os
#     import multiprocessing as mp
#
#     tot_time = time.time()
#
#     # handle case with only one output
#     if rshmeta.origin.shape.__len__() == 1:
#         rshmeta.origin = np.array([rshmeta.origin])
#     if type(rshmeta.file_name) == str:
#         rshmeta.file_dir = [rshmeta.file_dir]
#
#     # QC: ensure origins and file_names have same length
#     if rshmeta.origin.shape[0] != rshmeta.file_name.__len__():
#         raise Exception('origin_coords and img_out_path have different lengths, execution halted.')
#
#     rshm = pd.DataFrame({"id": rshmeta.id,
#                         "file_name": rshmeta.file_name,
#                         "file_dir": rshmeta.file_dir,
#                         "x_utm11n": rshmeta.origin[:, 0],
#                         "y_utm11n": rshmeta.origin[:, 1],
#                         "elevation_m": rshmeta.origin[:, 2],
#                         "src_las_file": vox.las_in,
#                         "vox_step": vox.step[0],
#                         "vox_sample_length": vox.sample_length,
#                         "src_return_set": vox.return_set,
#                         "src_drop_class": vox.drop_class,
#                         "ray_sample_length": rshmeta.ray_sample_length,
#                         "ray_iterations": rshmeta.ray_iterations,
#                         "img_size_px": rshmeta.img_size,
#                         "max_phi_rad": rshmeta.max_phi_rad,
#                         "min_distance_m": rshmeta.min_distance,
#                         "max_distance_m": rshmeta.max_distance,
#                         "prior_a": rshmeta.prior[0],
#                         "prior_b": rshmeta.prior[1],
#                         "created_datetime": None,
#                         "computation_time_s": None})
#
#     # resent index in case of rollover indexing
#     rshm = rshm.reset_index(drop=True)
#
#
#     # export table of rays in grid
#     ii = 0
#     origin = (rshm.x_utm11n[ii], rshm.y_utm11n[ii], rshm.elevation_m[ii])
#     # calculate rays
#     rays_in = point_to_hemi_rays(origin, rshmeta.img_size, vox, max_phi=rshmeta.max_phi_rad,
#                                  max_dist=rshmeta.max_distance, min_dist=rshmeta.min_distance)
#     phi_theta_lookup = rays_in.loc[:, ['x_index', 'y_index', 'phi', 'theta']]
#     phi_theta_lookup.to_csv(rshm.file_dir[ii] + "phi_theta_lookup.csv", index=False)
#
#
#     # initial_index = 0
#
#
#
#
#     rshm_split = np.array_split(rshm, n_cores)
#
#     if not os.path.exists(rshmeta.file_dir + "templog\\"):
#         os.makedirs(rshmeta.file_dir + "templog\\")
#
#     x = range(1, n_cores)
#     names = [rshmeta.file_dir + "templog\\rshmetalog_temp_" + str(y) + ".csv" for y in x]
#     voxlist = []
#     for ii in range(0, n_cores):
#         voxlist.append(vox)
#     args = list(zip(rshm_split, names, voxlist))
#     try:
#         pool = mp.Pool(n_cores)
#         pool.map(rsm_chunk, args)
#     except Exception as e:
#         pool.close()
#         raise
#     pool.close()
#
#     # compile temporary files
#     log_path = rshmeta.file_dir + "rshmetalog.csv"
#
#     with open(log_path, mode='w', encoding='utf-8') as log:
#         log.write(",".join(rshm.columns) + '\n')
#         for o in rshmeta.file_dir + "templog\\":
#             with open(o, 'r') as fin:
#                 for line in fin:
#                     log.write(line)
#
#     print("-------- Ray Sample Hemigen completed--------")
#     print(str(rshmeta.origin.shape[0] - initial_index) + " images generated in " + str(int(time.time() - tot_time)) + " seconds")
#     return rshm

#
#
# # sample voxel space from dem
# dem_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DEM\\interpolated\\19_149_dem_r1.00m_q0.25_interpolated_min1.tif"
# #dem_in = "C:\\Users\\jas600\\workzone\\data\\dem\\19_149_dem_res_.10m.bil"
# ras_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DEM\\19_149_expected_first_returns_res_.25m_0-0_t_1.tif"
# # ras_out = "C:\\Users\\jas600\\workzone\\data\\dem\\19_149_expected_returns_res_.10m.tif"
# phi = 0
# theta = 0
# agg_sample_length = vox.sample_length
# vec = [phi, theta]
# rays_in = dem_to_vox_rays(dem_in, vec, vox)
# rays_out = aggregate_voxels_over_rays(vox, rays_in, agg_sample_length)
# ras = ray_stats_to_dem(rays_out, dem_in)
# rastools.raster_save(ras, ras_out)
#
#
# # sample voxel space as hemisphere
# # import from hemi_grid_points
# hemi_pts = pd.read_csv('C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\synthetic_hemis\\uf_1m_pr_.15_os_10\\1m_dem_points.csv')
# hemi_pts = hemi_pts[hemi_pts.uf == 1]
# hemi_pts = hemi_pts[hemi_pts.id == 20426]
# pts = pd.DataFrame({'x0': hemi_pts.x_utm11n,
#                     'y0': hemi_pts.y_utm11n,
#                     'z0': hemi_pts.z_m})
#
# # # import from hemi-photo lookup
# #
# img_lookup_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\hemi_lookup_cleaned.csv"
# # img_lookup_in = 'C:\\Users\\jas600\\workzone\\data\\las\\hemi_lookup_cleaned.csv'
# max_quality = 4
# las_day = "19_149"
# # import hemi_lookup
# img_lookup = pd.read_csv(img_lookup_in)
# # filter lookup by quality
# img_lookup = img_lookup[img_lookup.quality_code <= max_quality]
# # filter lookup by las_day
# img_lookup = img_lookup[img_lookup.folder == las_day]
#
# pts = pd.DataFrame({'x0': img_lookup.xcoordUTM1,
#                     'y0': img_lookup.ycoordUTM1,
#                     'z0': img_lookup.elevation})
#
#
# # for each point
# ii = 0
# origin = (pts.iloc[ii].x0, pts.iloc[ii].y0, pts.iloc[ii].z0)
# img_size = 200
# agg_sample_length = vox.sample_length
# rays_in = point_to_hemi_rays(origin, img_size, vox, max_phi=np.pi/2, max_dist=50)
# start = time.time()
# rays_out, nb_lookup = aggregate_voxels_over_rays(vox, rays_in, agg_sample_length, prior)
# end = time.time()
# print(end - start)

#
# area_factor = .005
# img_path = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\RSM\\ray_sampling_transmittance_' + str(img_lookup.index[ii]) + '_af' + str(area_factor) + '.png'
# #img_path = 'C:\\Users\\jas600\\workzone\\data\\las\\' + str(img_lookup.index[ii]) + '_af' + str(area_factor) + '.png'
# hemi_rays_to_img(rays_out, img_path, img_size, area_factor)
#
# #
# #
# #
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
# plt.scatter(rays1.returns_mean, rays2.returns_mean)
# plt.scatter(rays_trad.returns_median, rays_squared.returns_median)
#
#
# #
# #
# # peace_1 = peace.data[0]
# peace_1[peace_1 == peace.no_data] = -1
# plt.imshow(peace_1, interpolation='nearest')
#
# peace_2 = peace.data[1]
# peace_2[peace_2 == peace.no_data] = 1
# plt.imshow(peace_2, interpolation='nearest')
#
# plt.imshow(ras.data[0], interpolation='nearest')
#
# ### VISUALIZATION
# import rastools
# import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
# ras_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DEM\\19_149_expected_returns_res_.25m_0-0_t_1.tif"
# ras = rastools.raster_load(ras_in)
#
#
# plot_data = ras.data[0]
# plot_data[plot_data == ras.no_data] = -10
# fig = plt.imshow(plot_data, interpolation='nearest', cmap='binary_r')
# plt.colorbar()
# plt.title('Upper Forest expected returns from nadir scans with no occlusion\n(ray-sampling method)')
# # plt.show(fig)
#
# plot_data = ras.data[2] / ras.data[0]
# plot_data[ras.data[2] == ras.no_data] = 0
# fig = plt.imshow(plot_data, interpolation='nearest', cmap='binary_r')
# plt.colorbar()
# plt.title('Upper Forest relative standard deviation of returns\n(ray-sampling method)')
# # plt.show(fig)
#
#
# plt.scatter(rays_out.x0, rays_out.y0)
# plt.scatter(ground_dem[0], ground_dem[1])
# plt.scatter(p0[:, 0], p0[:, 1])
#
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.imshow(template[:, :, 0], interpolation='nearest')

#
# from scipy import misc
# import imageio
# img = np.rint(template * 255).astype(np.uint8)
# img_out = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DEM\\ray_sampling_transmittance.png'
# imageio.imsave(img_out, img)
#
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.scatter(uni, coo)
# plt.scatter(runi, roo)
# plt.scatter(uni, cumsoo)
# plt.scatter(runi, rumsoo)
#
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
# plt.imshow(ras.data[1], interpolation='nearest')
#
# plt.plot(range(0, len(cs)), cs)
#
# result = lookup.items()
# peace = list(result)
# train = [[peace[ii][0][0], peace[ii][0][1], peace[ii][1]] for ii in range(0, len(peace))]
# arr = np.array(train)
# dft = pd.DataFrame(data=arr, columns={'alpha', 'p', 'cnt'})
# lala = dft.sort_values('cnt')
# cs = np.cumsum(np.flip(lala.cnt.values))