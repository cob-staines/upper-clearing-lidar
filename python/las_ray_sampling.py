import numpy as np
import pandas as pd
import laslib
import rastools
import time
import h5py
import os
import shutil
import tifffile as tiff
from tqdm import tqdm


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

        # voxel object data
        self.sample_data = None
        self.return_data = None

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def save_meta(self):
        save_vox_meta(self)

    def save_post(self):
        save_vox_post(self)

    def update_meta(self):
        update_vox_meta(self)


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


def save_vox_post(vox):
    with h5py.File(vox.vox_hdf5, 'a') as h5f:

        try:
            post = h5f.get('post')
        except ValueError:
            post = h5f.create_group("post")

        post.create_dataset('under_count', data=vox.under_count)
        post.create_dataset('unsamp_count', data=vox.unsamp_count)
        post.create_dataset('valid_count', data=vox.valid_count)
        post.create_dataset('prior_alpha', data=vox.prior_alpha)
        post.create_dataset('prior_beta', data=vox.prior_beta)
        post.create_dataset('agg_sample_length', data=vox.agg_sample_length)


def update_vox_meta(vox):
    with h5py.File(vox.vox_hdf5, 'a') as h5f:
        meta = h5f.get('meta')
        meta['las_in'][()] = vox.las_in
        meta['traj_in'][()] = vox.traj_in
        meta['las_traj_hdf5'][()] = str(vox.las_traj_hdf5)
        meta['return_set'][()] = vox.return_set
        meta['drop_class'][()] = vox.drop_class
        meta['las_traj_chunksize'][()] = vox.las_traj_chunksize
        meta['cw_rotation'][()] = vox.cw_rotation
        meta['vox_origin'][()] = vox.origin
        meta['vox_max'][()] = vox.max
        meta['vox_step'][()] = vox.step
        meta['vox_ncells'][()] = vox.ncells
        meta['vox_sample_length'][()] = vox.sample_length


def load_vox(hdf5_path, load_data=False, load_post=False, load_post_data=False):
    vox = VoxelObj()
    vox.vox_hdf5 = hdf5_path

    with h5py.File(hdf5_path, 'r') as h5f:
        # load voxel meta
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

        # load posterior meta
        if load_post:
            post = h5f.get('post')
            vox.under_count = post.get('under_count')[()]
            vox.unsamp_count = post.get('unsamp_count')[()]
            vox.valid_count = post.get('valid_count')[()]
            vox.prior_alpha = post.get('prior_alpha')[()]
            vox.prior_beta = post.get('prior_beta')[()]
            vox.agg_sample_length = post.get('agg_sample_length')[()]

        if load_post_data:
            post = h5f.get('post')
            vox.posterior_alpha = post.get('posterior_alpha')[()]
            vox.posterior_beta = post.get('posterior_beta')[()]
    return vox


def duplicate_vox(vox, hdf5_out, duplicate_post=False, z_slices=1):

    if isinstance(vox, str):
        vox_in = vox
        vox = load_vox(vox_in, load_data=False, load_post=duplicate_post, load_post_data=False)

    # define z_step
    z_step = np.ceil(vox.ncells[2] / z_slices).astype(int)

    # memory test of slice size (attempt to trigger memory error early)
    try:
        m_test = np.zeros((vox.ncells[0], vox.ncells[1], z_step), dtype=vox.sample_dtype)
        m_test = np.zeros((vox.ncells[0], vox.ncells[1], z_step), dtype=vox.return_dtype)
    except MemoryError:
        print("Size of z_slices is too large for memory. Try more z_slices.")
        raise
    finally:
        m_test = None

    # preallocate datasets
    with h5py.File(hdf5_out, mode='w') as hf_out:
        hf_out.create_dataset('sample_data', dtype=vox.sample_dtype, shape=vox.ncells, chunks=True, compression='gzip')
        hf_out.create_dataset('return_data', dtype=vox.return_dtype, shape=vox.ncells, chunks=True, compression='gzip')

        if duplicate_post:
            post_dtype = np.float32  # explicitly define posterior data type
            post = hf_out.create_group("post")
            post.create_dataset('posterior_alpha', dtype=post_dtype, shape=vox.ncells, chunks=True, compression='gzip')
            post.create_dataset('posterior_beta', dtype=post_dtype, shape=vox.ncells, chunks=True, compression='gzip')

    # copy datasets by z_slice
    for zz in tqdm(range(0, z_slices), leave=True, ncols=100, desc="Write data"):
        z_low = zz * z_step
        if zz != (z_slices - 1):
            z_high = (zz + 1) * z_step
        else:
            z_high = vox.ncells[2]

        with h5py.File(vox.vox_hdf5, mode='r') as hf_in:
            with h5py.File(hdf5_out, mode='r+') as hf_out:
                hf_out['sample_data'][:, :, z_low:z_high] = hf_in['sample_data'][:, :, z_low:z_high]
                hf_out['return_data'][:, :, z_low:z_high] = hf_in['return_data'][:, :, z_low:z_high]

                if duplicate_post:
                    hf_out['/post/posterior_alpha'][:, :, z_low:z_high] = hf_in['/post/posterior_alpha'][:, :, z_low:z_high]
                    hf_out['/post/posterior_beta'][:, :, z_low:z_high] = hf_in['/post/posterior_beta'][:, :, z_low:z_high]

    # update vox.vox_hdf5
    vox.vox_hdf5 = hdf5_out

    # save metadata
    vox.save_meta()
    if duplicate_post:
        vox.save_post()


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

    # both points below (neither point meets low criteria, set to nan)
    set = ~q_p0[:, 0] & ~q_p1[:, 0]
    # set to nan
    z0[set, :] = z1[set, :] = np.nan

    # both points above (neither point meets high criteria, set to nan)
    set = ~q_p0[:, 1] & ~q_p1[:, 1]
    # set to nan
    z0[set, :] = z1[set, :] = np.nan

    # one above, one below (interpolate above to high and below to low)
    # p0 below, p1 above
    set = ~q_p0[:, 0] & ~q_p1[:, 1]
    z0[set, :] = interp_to_z(p1[set, :], p0[set, :], z_low_utm)
    z1[set, :] = interp_to_z(p1[set, :], p0[set, :], z_high_utm)
    # p0 above, p1 below
    set = ~q_p0[:, 1] & ~q_p1[:, 0]
    z1[set, :] = interp_to_z(p1[set, :], p0[set, :], z_low_utm)
    z0[set, :] = interp_to_z(p1[set, :], p0[set, :], z_high_utm)

    # both in bounds (pass as is)
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


def las_ray_sample_by_z_slice(vox, z_slices=1, fail_overflow=False):

    print('----- LAS Ray Sampling -----')

    start_time = time.time()

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

    # check that gps_times align
    if np.all(las_time != traj_time):
        raise Exception('gps_times do not align between las and traj dfs, process aborted.')
    las_time = None
    traj_time = None

    # calculate n_chunks
    if vox.las_traj_chunksize is None:
        vox.las_traj_chunksize = n_rows
    n_chunks = np.ceil(n_rows / vox.las_traj_chunksize).astype(int)


    if (vox.origin is None) | (vox.max is None) :
        print("Setting voxel space origin and max from data extent.")

        with h5py.File(vox.las_traj_hdf5, 'r') as hf:
            z_min = np.min(hf['lasData'][:, 3])
            z_max = np.max(hf['lasData'][:, 3])

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

        if vox.origin is None:
            vox.origin = np.array([x_min, y_min, z_min])
        if vox.max is None:
            vox.max = np.array([x_max, y_max, z_max])
    else:
        # this is the case if vox.origin and vox.max are specified prior
        print("Voxel space origin and max provided by user.")

    vox.ncells = np.ceil((vox.max - vox.origin) / vox.step).astype(int)

    z_step = np.ceil(vox.ncells[2] / z_slices).astype(int)

    # memory test of slice size (attempt to trigger memory error early)
    try:
        m_test = np.zeros((vox.ncells[0], vox.ncells[1], z_step), dtype=vox.sample_dtype)
        m_test = np.zeros((vox.ncells[0], vox.ncells[1], z_step), dtype=vox.return_dtype)
    except MemoryError:
        print("Size of z_slices is too large for memory. Try more z_slices.")
        raise
    finally:
        m_test = None

    # preallocate voxel hdf5
    with h5py.File(vox.vox_hdf5, mode='w') as hf:
        hf.create_dataset('sample_data', dtype=vox.sample_dtype, shape=vox.ncells, chunks=True, compression='gzip')
        hf.create_dataset('return_data', dtype=vox.return_dtype, shape=vox.ncells, chunks=True, compression='gzip')

    # loop over las_traj chunks
    for ii in range(0, n_chunks):
        # print('Chunk ' + str(ii + 1) + ' of ' + str(n_chunks) + ': ')

        # chunk start and end
        idx_start = ii * vox.las_traj_chunksize
        if ii != (n_chunks - 1):
            idx_end = (ii + 1) * vox.las_traj_chunksize
        else:
            idx_end = n_rows

        with h5py.File(vox.las_traj_hdf5, 'r') as hf:
            ray_1_all = hf['lasData'][idx_start:idx_end, 1:4]
            ray_0_all = hf['trajData'][idx_start:idx_end, 1:4]

        # points in rotated reference frame
        pts_rot = np.matmul(ray_1_all, z_rotmat(vox.cw_rotation))

        # filter to returns within voxel space
        valid = np.all(pts_rot >= vox.origin, axis=1) & np.all(pts_rot <= vox.max, axis=1)
        ray_0 = ray_0_all[valid, :]
        ray_1 = ray_1_all[valid, :]

        # transform returns to vox coords
        rtns_vox = utm_to_vox(vox, ray_1).astype(int)

        # interpolate sensor to bounding box
        ray_bb = interpolate_to_bounding_box(ray_1, ray_0)

        # loop over z_slices
        for zz in range(0, z_slices):
            # print('\tSlice ' + str(zz + 1) + ' of ' + str(z_slices) + ': ')

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

            if len(z0) > 0:

                # load slices from vox
                with h5py.File(vox.vox_hdf5, mode='r') as hf:
                    sample_slice = hf['sample_data'][:, :, z_low:z_high]
                    return_slice = hf['return_data'][:, :, z_low:z_high]


                # add valid returns to return slice
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

                prog_desc = "chunk " + str(ii + 1) + "/" + str(n_chunks) + ": slice " + str(zz + 1) + "/" + str(z_slices)
                for jj in tqdm(range(0, max_step), leave=True, ncols=100, desc=prog_desc):
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

    print("-------- Las Ray Sampling completed--------")
    print('Total time: ' + str(int(time.time() - start_time)) + " seconds")

    return vox


def vox_addition(voxList, vox_out, z_slices=1):

    # inherit from 1st in list
    v0_file = voxList[0]

    # load v0 meta
    vout = load_vox(v0_file, load_data=False)

    # define z_step
    z_step = np.ceil(vout.ncells[2] / z_slices).astype(int)

    # memory test of slice size (attempt to trigger memory error early)
    try:
        m_test = np.zeros((vout.ncells[0], vout.ncells[1], z_step), dtype=vout.sample_dtype)
        m_test = np.zeros((vout.ncells[0], vout.ncells[1], z_step), dtype=vout.return_dtype)
    except MemoryError:
        print("Size of z_slices is too large for memory. Try more z_slices.")
        raise
    finally:
        m_test = None

    # create duplicate file of v0 (inherit all from v0)
    scrap = shutil.copy(v0_file, vox_out)

    # update vout meta
    vout.vox_hdf5 = vox_out
    vout.las_traj_hdf5 = ''
    vout.las_traj_chunksize = -1
    vout.drop_class = -1
    vout.return_set = ''
    vout.below_floor_count = -1

    vout.update_meta()

    # loop through remaining voxel objects in list
    for nn in range(1, len(voxList)):
        vox_n = load_vox(voxList[nn], load_data=False)

        prog_desc = "Vox " + str(nn)
        for zz in tqdm(range(0, z_slices), leave=True, ncols=100, desc=prog_desc):
            z_low = zz * z_step
            if zz != (z_slices - 1):
                z_high = (zz + 1) * z_step
            else:
                z_high = vout.ncells[2]

            with h5py.File(vout.vox_hdf5, mode='r+') as hf_0:
                with h5py.File(vox_n.vox_hdf5, mode='r') as hf_n:
                    hf_0['sample_data'][:, :, z_low:z_high] = hf_0['sample_data'][:, :, z_low:z_high] + hf_n['sample_data'][:, :, z_low:z_high]
                    hf_0['return_data'][:, :, z_low:z_high] = hf_0['return_data'][:, :, z_low:z_high] + hf_0['return_data'][:, :, z_low:z_high]


def beta_lookup_post_calc(vox, z_slices=1, agg_sample_length=None):
    """
    Calculates beta posterior from sample_data and return_data of voxel object.
    :param vox: voxel object, or path (str) pointing to voxel object hdf5 file
    :param z_slices: number of slices of voxel space, increase to avoid memory overflow
    :param agg_sample_length: sample length to be used for aggregation. If none, voxel sample length is used.
    :return: posterior data along with priors and descriptive stats are saved to voxel object hdf5 file in group "post"
    """

    if isinstance(vox, str):
        vox_in = vox
        vox = load_vox(vox_in, load_data=False, load_post=False)

    # set agg_sample length to vox.sample_length if not otherwise specified
    if agg_sample_length is None:
        agg_sample_length = vox.sample_length

    # define z_step
    z_step = np.ceil(vox.ncells[2] / z_slices).astype(int)

    # memory test of slice size (attempt to trigger memory error early)
    try:
        m_test = np.zeros((vox.ncells[0], vox.ncells[1], z_step), dtype=vox.sample_dtype)
        m_test = np.zeros((vox.ncells[0], vox.ncells[1], z_step), dtype=vox.return_dtype)
    except MemoryError:
        print("Size of z_slices is too large for memory. Try more z_slices.")
        raise
    finally:
        m_test = None

    # preallocate counts
    under_count = 0  # count cells with sample < returns
    unsamp_count = 0  # count cells with sample > 0
    valid_count = 0  # cells with sample in valid range
    rate_sum = 0  # cumulative sum of rates
    rate_sq_sum = 0  # cumulative sum of square of rates

    # loop through z_slices
    for zz in tqdm(range(0, z_slices), leave=True, ncols=100, desc="Post calc"):

        z_low = zz * z_step
        if zz != (z_slices - 1):
            z_high = (zz + 1) * z_step
        else:
            z_high = vox.ncells[2]

        # load slice
        with h5py.File(vox.vox_hdf5, mode='r') as hf:
            s_data = hf['sample_data'][:, :, z_low:z_high]
            r_data = hf['return_data'][:, :, z_low:z_high]

        # cells where returns exceed samples
        under = (s_data < r_data)
        under_count += np.sum(under)

        # cells with no samples
        unsamp = (s_data == 0)
        unsamp_count += np.sum(unsamp)

        # estimate prior parameters
        valid = ~under & ~unsamp
        valid_count += np.sum(valid)
        rate = r_data[valid] / s_data[valid]

        rate_sum += np.sum(rate)
        rate_sq_sum += np.sum(rate ** 2)

    # prior mean and variance
    mu = rate_sum / valid_count
    sig2 = (rate_sq_sum - (rate_sum ** 2)/valid_count)/(valid_count - 1)

    # prior hyperparameters
    prior_alpha = ((1 - mu)/sig2 - 1/mu) * (mu ** 2)
    prior_beta = prior_alpha * (1/mu - 1)

    # calculate and write prior lookup
    post_dtype = np.float32  # make sure to use float

    # preallocate posterior if does not exist
    with h5py.File(vox.vox_hdf5, mode='r+') as hf:
        try:
            post = hf.create_group("post")
            post_exists = False
        except ValueError:
            post = hf.get('post')
            post_exists = True
            print("Previous posterior exists -- Writing over")

        if not post_exists:
            post.create_dataset('posterior_alpha', dtype=post_dtype, shape=vox.ncells, chunks=True, compression='gzip')
            post.create_dataset('posterior_beta', dtype=post_dtype, shape=vox.ncells, chunks=True, compression='gzip')


    sample_length_correction = vox.sample_length / agg_sample_length

    # loop through z_slices
    for zz in tqdm(range(0, z_slices), leave=True, ncols=100, desc="Write posterior"):

        z_low = zz * z_step
        if zz != (z_slices - 1):
            z_high = (zz + 1) * z_step
        else:
            z_high = vox.ncells[2]

        with h5py.File(vox.vox_hdf5, mode='r+') as hf:
            s_data = hf['sample_data'][:, :, z_low:z_high]
            r_data = hf['return_data'][:, :, z_low:z_high]

            # correct under-counts
            under = (s_data < r_data)
            s_data[under] = r_data[under]

            kk = r_data
            nn = s_data * sample_length_correction

            hf['/post/posterior_alpha'][:, :, z_low:z_high] = prior_alpha + kk
            hf['/post/posterior_beta'][:, :, z_low:z_high] = prior_beta + nn - kk

    with h5py.File(vox.vox_hdf5, mode='r+') as hf:
        post = hf.get('post')

        # record prior and count values
        if post_exists:
            post['under_count'][()] = under_count
            post['unsamp_count'][()] = unsamp_count
            post['valid_count'][()] = valid_count
            post['prior_alpha'][()] = prior_alpha
            post['prior_beta'][()] = prior_beta
            post['agg_sample_length'][()] = agg_sample_length
        else:

            post.create_dataset('under_count', data=under_count)
            post.create_dataset('unsamp_count', data=unsamp_count)
            post.create_dataset('valid_count', data=valid_count)
            post.create_dataset('prior_alpha', data=prior_alpha)
            post.create_dataset('prior_beta', data=prior_beta)
            post.create_dataset('agg_sample_length', data=agg_sample_length)


def beta(rays, path_returns, path_samples, vox_sample_length, agg_sample_length, prior, weights=1):

    kk = path_returns
    nn = path_samples * vox_sample_length / agg_sample_length

    post_a = prior[0] + kk
    post_b = prior[1] + nn - kk

    # normal approximation of sum
    returns_mean = np.nansum(weights * post_a / (post_a + post_b), axis=1)
    returns_std = np.sqrt(np.nansum(weights * post_a * post_b / ((post_a + post_b) ** 2 * (post_a + post_b + 1)), axis=1))

    rays = rays.assign(returns_mean=returns_mean)
    rays = rays.assign(returns_std=returns_std)

    return rays

def beta_lookup(rays, post_a, post_b, weights=1):

    # normal approximation of sum
    returns_mean = np.nansum(weights * post_a/(post_a + post_b), axis=1)
    returns_std = np.sqrt(np.nansum(weights * post_a * post_b / ((post_a + post_b) ** 2 * (post_a + post_b + 1)), axis=1))

    rays = rays.assign(returns_mean=returns_mean)
    rays = rays.assign(returns_std=returns_std)

    return rays

def single_ray_agg(vox, rays, agg_sample_length, lookup_db="posterior"):

    if lookup_db == 'count':
        a_data = vox.return_data
        b_data = vox.sample_data
    elif lookup_db == 'posterior':
        a_data = vox.posterior_alpha
        b_data = vox.posterior_beta
    else:
        raise Exception('Not a valid lookup_db: ' + lookup_db)

    # pull ray endpoints
    p0 = rays.loc[:, ['x0', 'y0', 'z0']].values
    p1 = rays.loc[:, ['x1', 'y1', 'z1']].values


    # distance between ray start (ground) and end (sky)
    dist = rays.path_length.values

    # calc unit step along ray in x, y, z dims
    xyz_step = ((p1 - p0).T / dist).T

    # random offset for each ray sample series
    offset = np.random.random(len(p0))

    # calculate number of samples
    n_samples = ((dist - offset) / agg_sample_length).astype(int)
    max_steps = np.max(n_samples)

    # preallocate aggregate lists (inherit dtype from data)
    a_path_samps = np.full([len(p0), max_steps], np.nan, dtype=a_data.dtype)
    b_path_samps = np.full([len(p0), max_steps], np.nan, dtype=b_data.dtype)

    # for each sample step
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

            a_path_samps[in_range, ii] = a_data[sample_address]
            b_path_samps[in_range, ii] = b_data[sample_address]


    if lookup_db == 'count':
        prior = (vox.prior_alpha, vox.prior_beta)
        rays_out = beta(rays, a_path_samps, b_path_samps, vox.sample_length, agg_sample_length, prior)
    elif lookup_db == 'posterior':
        rays_out = beta_lookup(rays, a_path_samps, b_path_samps)
    else:
        raise Exception('Not a valid lookup_db: ' + lookup_db)

    return rays_out



def single_ray_group_agg(vox, rays, agg_sample_length, lookup_db, prior):

    # pull ray endpoints
    p0 = rays.loc[:, ['x0', 'y0', 'z0']].values
    p1 = rays.loc[:, ['x1', 'y1', 'z1']].values


    # distance between ray start (ground) and end (sky)
    dist = rays.path_length.values

    # calc unit step along ray in x, y, z dims
    xyz_step = ((p1 - p0).T / dist).T

    # random offset for each ray sample series
    offset = np.random.random(len(p0))

    # calculate number of samples
    n_samples = ((dist - offset) / agg_sample_length).astype(int)
    max_steps = np.max(n_samples)

    # preallocate aggregate lists
    path_samples = np.full([len(p0), max_steps], np.nan)
    path_returns = np.full([len(p0), max_steps], np.nan)

    # for each sample step
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


    if lookup_db == 'count':
        rays_out = beta(rays, path_returns, path_samples, vox.sample_length, agg_sample_length, prior)
    elif lookup_db == 'posterior':
        rays_out = beta_lookup(rays, path_samples, path_returns)
    else:
        raise Exception('Not a valid lookup_db: ' + lookup_db)

    return rays_out


def vox_agg(origin, vox_sub, img_size, max_phi=np.pi/2, max_dist=50, min_dist=0, ref="edge"):

    # use rays to record
    # rays = hemi_vectors(img_size, max_phi, ref=ref)
    rays = point_to_hemi_rays(origin, vox_sub, img_size, max_phi, max_dist=max_dist, min_dist=min_dist)

    # convert min/max to voxel units
    vr_max = max_dist / vox_sub.step[0]
    vr_min = min_dist / vox_sub.step[0]

    # convert origin to vox coords
    v0 = utm_to_vox(vox_sub, origin)
    v0 = v0 - (vox_sub.step / 2)  # shift origin to shift vox corners to vox centers
    v0 = v0 + (np.random.random(3) * 2 - 1) / 10 ** 10  # add small noise to avoid alignment with vox grid (division by zero) -- can we solve this in a cleaner way please?

    tp = v0 + (0, 0, 1)  # test point for debugging

    # all voxel corners within cartesian max_dist
    ub = np.ceil(v0 + np.array([vr_max * np.sin(max_phi), vr_max * np.sin(max_phi), vr_max])).astype(int)
    ub = np.min([vox_sub.ncells - 1, ub], axis=0)  # limit ub to vox_sub size

    lb = np.floor(v0 - np.array([vr_max, vr_max, 0])).astype(int)
    lb = np.max([(0, 0, 0), lb], axis=0)  # limit lb to 0

    vx, vy, vz = np.indices(ub - lb + 1)
    vv = np.array([np.ravel(vx), np.ravel(vy), np.ravel(vz)]).swapaxes(0, 1) + lb

    # calculate distance from origin
    vr = np.sqrt(np.sum((vv - v0) ** 2, axis=1))
    # tr = np.sqrt(np.sum((tp - v0) ** 2))

    # filter to within min/max dist range and in upper hemisphere
    in_range = (vr >= vr_min) & (vr <= vr_max) & (vv[:, 2] >= v0[2])

    # calculate polar coordinates
    v_v = vv[in_range]
    v_r = vr[in_range]
    v_phi = np.arccos((v_v[:, 2] - v0[2])/v_r)
    v_theta = np.arctan2(v_v[:, 0] - v0[0], v_v[:, 1] - v0[1])

    # t_phi = np.arccos((tp[2] - v0[2])/tr)
    t_theta = np.arctan2(tp[0] - v0[0], tp[1] - v0[1])


    # x_index = np.rint((v_phi * img_size / max_phi) / np.sqrt(1 + (v_v[:, 1] - v0[1]) ** 2 / (v_v[:, 0] - v0[0]) ** 2)).astype(int)

    # y_index = np.rint((v_phi * img_size / max_phi) / np.sqrt(1 + (v_v[:, 0] - v0[0]) ** 2 / (v_v[:, 1] - v0[1]) ** 2)).astype(int)

    o_index = (img_size - 1) / 2

    if ref == "center":
        phi_step = max_phi * 2 / (img_size)
    elif ref == "edge":
        phi_step = max_phi * 2 / (img_size - 1)
    else:
        raise Exception("'ref' only takes values of 'center' or 'edge'. Received: " + str(ref))

    x_int = np.rint(o_index + np.sin(v_theta) * v_phi / phi_step).astype(int)
    y_int = np.rint(o_index + np.cos(v_theta) * v_phi / phi_step).astype(int)

    # t_x_int = np.rint(o_index + np.sin(t_theta) * t_phi / phi_step).astype(int)
    # t_y_int = np.rint(o_index + np.cos(t_theta) * t_phi / phi_step).astype(int)


    v_address = (v_v[:, 0], v_v[:, 1], v_v[:, 2])

    samps = vox_sub.sample_data[v_address]
    rets = vox_sub.return_data[v_address]

    weights = np.prod(vox_sub.step) / (4 * phi_step * np.cos(phi_step) * v_r ** 2)
    weights = np.prod(vox_sub.step) / (4 * phi_step * np.cos(phi_step) * vox_sub.sample_length * (v_r * vox_sub.step[0]) ** 2)
    weights = np.prod(vox_sub.step) / (4 * phi_step * np.sin(phi_step) * vox_sub.sample_length * (v_r * vox_sub.step[0]) ** 2)
    weights = np.prod(vox_sub.step) / (phi_step * np.sin(phi_step) * (v_r * vox_sub.step[0]) ** 2)
    # weights = 1 / (v_r * vox_sub.step[0]) ** 2


    rays.loc[:, "id"] = rays.index
    voxels = pd.DataFrame({"x_index": x_int,
                           "y_index": y_int,
                           "vox_mean": weights * rets / (rets + samps),
                           "vox_var": weights * rets * samps / ((rets + samps) ** 2 * (rets + samps + 1)),
                           "weights": weights})


    peace = voxels.groupby(["x_index", "y_index"], as_index=False).agg(
        returns_mean=pd.NamedAgg(column="vox_mean", aggfunc=np.sum),
        returns_var=pd.NamedAgg(column="vox_var", aggfunc=np.sum),
        voxel_count=pd.NamedAgg(column="weights", aggfunc=np.size),
        weight_sum=pd.NamedAgg(column="weights", aggfunc=np.sum))
    peace.loc[:, 'vol'] = peace.voxel_count * np.prod(vox_sub.step)
    peace.loc[:, 'arr'] = (3 * peace.voxel_count * np.prod(vox_sub.step) / ((2 * max_phi) / img_size) ** 2) ** (1/3)

    output = pd.merge(rays, peace, how="left", on=["x_index", "y_index"])

    lala = pd.merge(rays_out, output, how="outer", on=["x_index", "y_index"])

    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt

    plt.scatter(lala.returns_mean_x, lala.returns_mean_y, alpha=.05)
    plt.scatter(lala.returns_mean_x, lala.returns_mean_y * lala.path_length_y / lala.weight_sum, alpha=.05)
    plt.scatter(lala.path_length_y, lala.weight_sum)
    plt.scatter(lala.path_length_y, lala.weight_sum / lala.path_length)
    plt.scatter(lala.path_length_y, lala.arr)  # this looks good! I think we are getting the right number of voxels in the search area, quite reassuring.

    #
    # returns_mean = np.nansum(weights * post_a/(post_a + post_b), axis=1)
    # returns_std = np.sqrt(np.nansum(weights * post_a * post_b / ((post_a + post_b) ** 2 * (post_a + post_b + 1)), axis=1))
    #
    #
    # peace.agg(np.size)
    #
    # # save this for the end...
    # peace = pd.merge(rays.loc[:, ["x_index", "y_index", "id"]], voxels, how="left", on=["x_index", "y_index"])  # some ids are missing values. why would this be?
    #
    #
    #
    # bins, indices, inverse, vox_count = np.unique(np.array([x_index, y_index]), axis=1, return_counts=True, return_inverse=True, return_index=True)  # why are counts not ~uniform? This is problematic. either include more points below, or normalize by (weighted) number of points.
    #
    # bins_df = pd.DataFrame({"x_index": bins[0],
    #                      "y_index": bins[1],
    #                      "vox_count": vox_count.astype(int)})
    # bins_df = bins_df.reset_index()
    # bins_df.columns = ['id', 'x_index', 'y_index', 'vox_count']
    #
    # rays_in = pd.merge(rays, bins_df, how="left", on=["x_index", "y_index"])  # getting nans out of this, something is wrong. The problem line above is a good indicator.
    #
    # vol_returns = np.full([len(rays), np.max(vox_count)], np.nan)
    # vol_samples = np.full([len(rays), np.max(vox_count)], np.nan)
    # vol_weights = np.full([len(rays), np.max(vox_count)], np.nan)
    #
    # ii = 0
    # # for each ray (group by?)
    # for ii in range(0, len(rays_in)):
    #     print(str(ii + 1) + ' of ' + str(len(rays_in)))
    #     # weigh by 1/r^2
    #
    #     if not np.isnan(rays_in.id[ii]):  # wickedly slow!! this needs some serious rethinking if this is going to be functional, let alone faster than the single ray sampling....
    #         # list of values
    #         vox_coords = vv[in_range, :][inverse == rays_in.id[ii], :]
    #         vox_address = (vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2])
    #
    #         # pull values
    #         vol_returns[ii, 0:rays_in.vox_count[ii].astype(int)] = vox_sub.return_data[vox_address]
    #         vol_samples[ii, 0:rays_in.vox_count[ii].astype(int)] = vox_sub.sample_data[vox_address]
    #         vol_weights[ii, 0:rays_in.vox_count[ii].astype(int)] = 1 / vr[in_range][inverse == rays_in.id[ii]] ** 2
    #
    # if lookup_db == 'count':
    #     #rays_out = beta(rays, vol_samples, vol_returns, vox_sub.sample_length, agg_sample_length, prior)
    #     pass
    # elif lookup_db == 'posterior':
    #     rays_out = beta_lookup(rays, vol_samples, vol_returns, weights=vol_weights)
    # else:
    #     raise Exception('Not a valid lookup_db: ' + lookup_db)

    return rays_out


def agg_ray_chunk(chunksize, vox, rays, agg_method, agg_sample_length, lookup_db='posterior', prior=None, commentation=False):
    # this needs to be rethought if I am hoping to use this function for more than just single_ray_agg...
    n_rays = len(rays)

    # allow null case
    if chunksize is None:
        chunksize = n_rays

    n_chunks = np.ceil(n_rays / chunksize).astype(int)

    if n_chunks > 1:
        print("\n\t", end='')

    all_rays_out = rays[0:0].copy()

    # chunk aggregation
    for ii in range(0, n_chunks):
        if n_chunks > 1:
            print('Chunk ' + str(ii + 1) + ' of ' + str(n_chunks) + ': ', end='')

        # chunk start and end
        idx_start = ii * chunksize
        if ii != (n_chunks - 1):
            idx_end = (ii + 1) * chunksize
        else:
            idx_end = n_rays

        # extract ray chunk
        chunk_rays_in = rays.iloc[idx_start:idx_end, ]
        set_pts = np.concatenate((chunk_rays_in.loc[:, ['x0', 'y0', 'z0']].values, chunk_rays_in.loc[:, ['x1', 'y1', 'z1']].values), axis=0)

        # subset vox
        vox_sub = subset_vox(set_pts, vox, 0, lookup_db)

        # aggregate
        if agg_method == "single_ray_agg":
            chunk_rays_out = single_ray_agg(vox_sub, chunk_rays_in, agg_sample_length, lookup_db)
        else:
            raise Exception("Unknown agg_method: " + agg_method)

        # concatenat output
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
    rays = rays.assign(path_length=np.sqrt(np.sum((p1_bb - p0) ** 2, axis=1)))

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


def hemi_vectors(img_size, max_phi, ref="center"):
    # identify vectors within hemisphere

    # convert img index to phi/theta
    img_origin = (img_size - 1) / 2
    # cell index
    template = np.full([img_size, img_size], True)
    index = np.where(template)

    rays = pd.DataFrame({'x_index': index[0],
                         'y_index': index[1]})

    # calculate phi and theta
    if ref == "center":
        rays = rays.assign(phi=np.sqrt((rays.x_index - img_origin) ** 2 + (rays.y_index - img_origin) ** 2) * max_phi * 2 / (img_size - 1))
    elif ref == "edge":
        rays = rays.assign(phi=np.sqrt((rays.x_index - img_origin) ** 2 + (rays.y_index - img_origin) ** 2) * max_phi * 2 / img_size)
    else:
        raise Exception("'ref' only takes values of 'center' or 'edge'. Received: " + str(ref))
    rays = rays.assign(theta=np.arctan2((rays.x_index - img_origin), (rays.y_index - img_origin)) + np.pi)

    # fix division by zero case
    rays.loc[np.isnan(rays.phi), 'phi'] = 0

    # remove rays which exceed max_phi (beyond image horizon)
    rays = rays[rays.phi <= max_phi]

    return rays


def photo_vectors(phi_range, phi_count, theta_range, theta_count):
    # mercator projection (min error @ phi = pi/2, max @ phi = 0 or phi=pi)

    # identify vectors within hemisphere
    img_size = np.array((phi_count, theta_count))

    # convert img index to phi/theta
    img_origin = (img_size - 1) / 2

    # cell index
    template = np.full(img_size, True)
    index = np.where(template)

    rays = pd.DataFrame({'y_index': index[0],
                         'x_index': index[1]})
    # "y is phi" (so x in theta)

    phi_step = np.diff(phi_range) / phi_count
    theta_step = np.diff(theta_range) / theta_count

    rays = rays.assign(phi=(rays.y_index - img_origin[0]) * phi_step[0] + np.mean(phi_range))
    rays = rays.assign(theta=(rays.x_index.values - img_origin[1]) * theta_step[0] + np.mean(theta_range))

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
    rays = rays.assign(path_length=np.sqrt(np.sum((p1_bb - p0) ** 2, axis=1)))

    return rays


def point_to_photo_rays(origin, vox, img_size, max_phi=np.pi/2, max_dist=50, min_dist=0):
    # needs reworking

    rays = photo_vectors(img_size, max_phi)

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
    rays = rays.assign(path_length=np.sqrt(np.sum((p1_bb - p0) ** 2, axis=1)))

    return rays


def hemi_rays_to_img(rays_out, img_path, img_size, area_factor):
    import imageio

    rays_out = rays_out.assign(transmittance=np.exp(-1 * area_factor * rays_out.returns_mean))
    template = np.full([img_size, img_size], 1.0)
    template[(rays_out.y_index.values, rays_out.x_index.values)] = rays_out.transmittance

    img = np.rint(template * 255).astype(np.uint8)
    imageio.imsave(img_path, img)


def las_to_vox(vox, z_slices, run_las_traj=True, fail_overflow=False, posterior_calc=True):
    if run_las_traj:
        # interpolate trajectory
        laslib.las_traj(vox.las_in, vox.traj_in, vox.las_traj_hdf5, vox.las_traj_chunksize, vox.return_set, vox.drop_class)

    # sample voxel space from las_traj hdf5
    vox = las_ray_sample_by_z_slice(vox, z_slices, fail_overflow)
    vox.save_meta()

    if posterior_calc:
        vox = load_vox(vox.vox_hdf5, load_data=True)
        beta_lookup_post_calc(vox)

    return vox


def vox_to_las(vox, las_out, samps_per_vox=1, sample_threshold=0):
    # generate las file with point density according to return rate

    import laspy

    # load vox if not already done
    if isinstance(vox, str):
        print("loading vox...", end="")
        vox_in = vox
        vox = load_vox(vox_in, load_data=True, load_post=True, load_post_data=True)
        print("done")

    rate = vox.posterior_alpha / (vox.posterior_alpha + vox.posterior_beta)  # units of returns per agg sample length
    valid = (vox.sample_data >= sample_threshold)  # do not try to sample voxels with sampled below the sample threshold

    # sampled = vox.sample_data > 0

    vox.sample_data = None
    vox.return_data = None
    vox.posterior_alpha = None
    vox.posterior_beta = None

    for ii in tqdm(range(samps_per_vox), desc="vox samples", ncols=100):
        accept = (np.random.random(rate.shape) <= rate) & valid
        vox_accept = np.where(accept)
        # vox_classification = sampled[accept]
        vox_address = np.array([vox_accept[0], vox_accept[1], vox_accept[2]]).swapaxes(0, 1)

        if ii == 0:
            vox_address_all = vox_address
            # vox_classification_all = vox_classification
        else:
            vox_address_all = np.concatenate((vox_address_all, vox_address), axis=0)
            # vox_classification_all = np.concatenate((vox_classification_all, vox_classification), axis=0)

    # add sub-voxel noise
    vox_address_noise = vox_address_all + np.random.random(vox_address_all.shape)

    # convert to utm
    utm_points = vox_to_utm(vox, vox_address_noise)

    # hdr = laspy.header.Header(file_version=1.4, point_format=7)
    hdr = laspy.header.Header()
    mins = np.floor(np.min(utm_points, axis=0))
    maxs = np.ceil(np.max(utm_points, axis=0))
    scale = [0.00025, 0.00025, 0.00025]
    print("writing las...", end="")
    with laspy.file.File(las_out, mode="w", header=hdr) as outfile:
        # header attributes
        outfile.software_id = "las_ray_sampling.py/vox_to_las"
        outfile.header.offset = mins
        outfile.header.scale = scale
        outfile.header.min = mins
        outfile.header.max = maxs

        # points
        outfile.x = utm_points[:, 0]
        outfile.y = utm_points[:, 1]
        outfile.z = utm_points[:, 2]
        # outfile.classification = vox_classification_all.astype(np.uint8)
    print("done")


def subset_vox(pts, vox, buffer, lookup_db='posterior'):

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

    # from post
    vox_sub.prior_alpha = vox.prior_alpha
    vox_sub.prior_beta = vox.prior_beta
    vox_sub.agg_sample_length = vox.agg_sample_length

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


    if lookup_db == 'count':
        with h5py.File(vox_sub.vox_hdf5, mode='r') as hf:
            vox_sub.sample_data = hf['sample_data'][vox_sub_min[0]:vox_sub_max[0], vox_sub_min[1]:vox_sub_max[1], vox_sub_min[2]:vox_sub_max[2]]
            vox_sub.return_data = hf['return_data'][vox_sub_min[0]:vox_sub_max[0], vox_sub_min[1]:vox_sub_max[1], vox_sub_min[2]:vox_sub_max[2]]
    elif lookup_db == 'posterior':
        with h5py.File(vox_sub.vox_hdf5, mode='r') as hf:
            vox_sub.posterior_alpha = hf['/post/posterior_alpha'][vox_sub_min[0]:vox_sub_max[0], vox_sub_min[1]:vox_sub_max[1], vox_sub_min[2]:vox_sub_max[2]]
            vox_sub.posterior_beta = hf['/post/posterior_beta'][vox_sub_min[0]:vox_sub_max[0], vox_sub_min[1]:vox_sub_max[1], vox_sub_min[2]:vox_sub_max[2]]
    else:
        raise Exception('invalid specification for "lookup_db": ' + str(lookup_db))

    return vox_sub


class RaySampleHemiMetaObj(object):
    def __init__(self):
        # preload metadata
        self.id = None
        self.file_name = None
        self.file_dir = None
        self.origin = None
        self.agg_sample_length = None
        self.max_phi_rad = None
        self.max_distance = None
        self.min_distance = None
        self.img_size = None
        self.prior = None
        self.lookup_db = None
        self.config_id = None
        self.agg_method = None


class RaySampleGridMetaObj(object):
    def __init__(self):
        # preload metadata
        self.id = None
        self.file_name = None
        self.file_dir = None
        self.src_ras_file = None
        self.mask_file = None
        self.agg_sample_length = None
        self.phi = None
        self.theta = None
        self.max_distance = None
        self.min_distance = None
        self.prior = None
        self.lookup_db = None
        self.config_id = None
        self.agg_method = None

class RaySamplePhotoMetaObj(object):
    def __init__(self):
        # preload metadata
        self.id = None
        self.file_name = None
        self.file_dir = None
        self.origin = None
        self.agg_sample_length = None
        self.min_distance = None
        self.max_distance = None
        self.phi_range = None
        self.phi_count = None
        self.theta_range = None
        self.theta_count = None
        self.prior = None
        self.lookup_db = None
        self.config_id = None
        self.agg_method = None


def rs_hemigen(rshmeta, vox, tile_count_1d=1, n_cores=1, initial_index=0):

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
                        "agg_sample_length": rshmeta.agg_sample_length,
                        "img_size_px": rshmeta.img_size,
                        "max_phi_rad": rshmeta.max_phi_rad,
                        "min_distance_m": rshmeta.min_distance,
                        "max_distance_m": rshmeta.max_distance,
                        "lookup_db": rshmeta.lookup_db,
                        "config_id": rshmeta.config_id,
                        "agg_method": rshmeta.agg_method,
                        "created_datetime": np.nan,
                        "computation_time_s": np.nan})

    # resent index in case of rollover indexing
    rshm = rshm.reset_index(drop=True)

    # export phi_theta_lookup of vectors in grid
    vector_set = hemi_vectors(rshmeta.img_size, rshmeta.max_phi_rad)
    vector_set.to_csv(rshm.file_dir[0] + "phi_theta_lookup.csv", index=False)

    if initial_index is not 0:
        rshm = rshm.iloc[initial_index:, :]

    if tile_count_1d == 1:
        # single tile, single core

        # preallocate log file
        log_path = rshmeta.file_dir + "rshmetalog.csv"
        if not os.path.exists(log_path):
            with open(log_path, mode='w', encoding='utf-8') as log:
                log.write(",".join(rshm.columns) + '\n')
            log.close()

        rshm = rshm_iterate(rshm, rshmeta, vox, log_path, process_id=0, nrows=1)

    else:
        # multiple tiles
        from scipy.stats import binned_statistic_2d
        from itertools import repeat
        import multiprocessing.pool as mpp

        # create dir for tile log files
        if not os.path.exists(rshmeta.file_dir + "\\tile_logs\\"):
            os.makedirs(rshmeta.file_dir + "\\tile_logs\\")

        # batch tiling
        bin_counts, x_bins, y_bins, bin_num = binned_statistic_2d(rshm.x_utm11n, rshm.y_utm11n, rshm.y_utm11n, statistic='count', bins=tile_count_1d)
        rshm.loc[:, 'tile_id'] = bin_num
        tiles = np.unique(bin_num)

        rshm_list = []
        log_path_list = []
        for tt in tiles:

            # preallocate tile log file
            log_path = rshmeta.file_dir + "\\tile_logs\\rshmetalog_tile_" + str(tt) + ".csv"
            log_path_list.append(log_path)
            if not os.path.exists(log_path):
                with open(log_path, mode='w', encoding='utf-8') as log:
                    log.write(",".join(rshm.columns) + '\n')
                log.close()

            rshm_tile = rshm.loc[rshm.tile_id == tt, :].copy()
            rshm_list.append(rshm_tile)

        if n_cores > 1:
            # multiple cores
            with mpp.ThreadPool(processes=n_cores) as pool:
                # mm = pool.starmap(rshm_iterate, zip(rshm_list, repeat(rshmeta), repeat(vox), log_path_list, np.arange(0, len(tiles)), repeat(len(tiles) + 1)))
                mm = pool.starmap(rshm_iterate, zip(rshm_list, repeat(rshmeta), repeat(vox), log_path_list, np.arange(0, len(tiles)), repeat(len(tiles))))
        else:
            # single core
            for jj in range(0, len(tiles)):
                rshm_tile = rshm_iterate(rshm_list[jj], rshmeta, vox, log_path_list[jj], jj, 1)

        # compile tile log files
        t_loglist = os.listdir(rshmeta.file_dir + "\\tile_logs\\")
        log_comp = rshm.copy()
        log_comp.drop(log_comp.index, inplace=True)

        temp_log_count = []
        for ff in t_loglist:
            temp_log = pd.read_csv(rshmeta.file_dir + "\\tile_logs\\" + ff)
            temp_log_count.append(len(temp_log))
            log_comp = log_comp.append(temp_log, ignore_index=True)

        log_comp.to_csv(rshmeta.file_dir + "rshmetalog.csv")

    print("-------- Ray Sample Hemigen completed--------")
    print(str(rshmeta.origin.shape[0]) + " images generated in " + str(
        int(time.time() - tot_time)) + " seconds")

    return rshm


def rshm_iterate(rshm, rshmeta, vox, log_path, process_id=0, nrows=4):

    vox_sub = subset_vox(rshm.loc[:, ['x_utm11n', 'y_utm11n', 'elevation_m']].values, vox, rshmeta.max_distance, rshmeta.lookup_db)

    for ii in tqdm(range(0, len(rshm)), position=process_id, desc=str(process_id), leave=True, ncols=100, nrows=nrows + 1):
        it_time = time.time()

        iid = rshm.index[ii]

        origin = (rshm.x_utm11n.iloc[ii], rshm.y_utm11n.iloc[ii], rshm.elevation_m.iloc[ii])

        if rshmeta.agg_method == "single_ray_agg":
            # calculate rays
            rays_in = point_to_hemi_rays(origin, vox_sub, rshmeta.img_size, rshmeta.max_phi_rad, max_dist=rshmeta.max_distance, min_dist=rshmeta.min_distance)

            # # sample rays
            # vox_old = vox
            # vox = vox_sub
            # rays = rays_in
            # agg_sample_length = rshmeta.agg_sample_length
            # lookup_db = rshmeta.lookup_db
            rays_out = single_ray_agg(vox_sub, rays_in, rshmeta.agg_sample_length, lookup_db=rshmeta.lookup_db)
        elif rshmeta.agg_method == "vox_agg":

            # vox_bu = vox
            # vox = vox_sub
            # img_size = rshmeta.img_size
            # max_phi = rshmeta.max_phi_rad
            # max_dist = rshmeta.max_distance
            # min_dist = rshmeta.min_distance
            rays_out = vox_agg(origin, vox_sub, rshmeta.img_size, rshmeta.max_phi_rad, rshmeta.max_distance, rshmeta.min_distance)
        else:
            raise Exception("Unknown agg_method: " + rshmeta.agg_method)


        # format to image
        template = np.full((rshmeta.img_size, rshmeta.img_size, 2), np.nan)
        template[(rays_out.y_index.values, rays_out.x_index.values, 0)] = rays_out.returns_mean
        template[(rays_out.y_index.values, rays_out.x_index.values, 1)] = rays_out.returns_std
        # write image
        tiff.imsave(rshm.file_dir.iloc[ii] + rshm.file_name.iloc[ii], template)

        # log meta

        rshm.loc[iid, "created_datetime"] = time.strftime('%Y-%m-%d %H:%M:%S')
        rshm.loc[iid, "computation_time_s"] = int(time.time() - it_time)

        # write to log file
        rshm.iloc[ii:ii + 1].to_csv(log_path, encoding='utf-8', mode='a', header=False, index=False)

    return rshm


def rs_gridgen(rsgmeta, vox, chunksize=1000000, initial_index=0, commentation=True):

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
                        "agg_sample_length": rsgmeta.agg_sample_length,
                        "min_distance_m": rsgmeta.min_distance,
                        "max_distance_m": rsgmeta.max_distance,
                        "lookup_db": rsgmeta.lookup_db,
                        "config_id": rsgmeta.config_id,
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

    for ii in tqdm(range(initial_index, len(rsgm)), desc="angle set", ncols=100):
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
        # agg_sample_length = rsgmeta.agg_sample_length
        # prior = rsgmeta.prior
        # commentation = True
        # method = rsgmeta.agg_method
        rays_out = agg_ray_chunk(chunksize, vox, rays_in, rsgmeta.agg_method, rsgmeta.agg_sample_length, rsgmeta.lookup_db, rsgmeta.prior, commentation)

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

def rs_photogen(rspmeta, vox, initial_index=0):
    """
    generates ray sampling outputs to mimic mercator projection photos with:
        focal points at rspmeta.origin (n x 3 array: x, y, z)
        zenith angle range of rspmeta.phi_range (n x 2 array: min, max) with resolution of rspmeta.phi_count
        azimuth angle range of rspmeta.theta_range (n x 2 array: min, max) with resolution of rspmeta.theta_count
    :param rspmeta: ray sampling config object
    :param vox: voxel space object
    :param initial_index: starting index (for resuming after a crash, for example)
    :return:
    """

    ### not yet troubleshooted! ###

    tot_time = time.time()

    # handle case with only one output
    if rspmeta.origin.shape.__len__() == 1:
        rspmeta.origin = np.array([rspmeta.origin])
    if rspmeta.phi_range.shape.__len__() == 1:
        rspmeta.phi_range = np.array([rspmeta.phi_range])
    # if rspmeta.phi_count.shape.__len__() == 1:
    #     rspmeta.phi_count = np.array([rspmeta.phi_count])
    if rspmeta.theta_range.shape.__len__() == 1:
        rspmeta.theta_range = np.array([rspmeta.theta_range])
    # if rspmeta.theta_count.shape.__len__() == 1:
    #     rspmeta.theta_count = np.array([rspmeta.theta_count])
    if type(rspmeta.file_name) == str:
        rspmeta.file_name = [rspmeta.file_name]

    # QC: ensure origins and file_names have same length
    if rspmeta.origin.shape[0] != rspmeta.file_name.__len__():
        raise Exception('origin and file_name have different lengths, execution halted.')
    if rspmeta.phi_range.shape[0] != rspmeta.file_name.__len__():
        raise Exception('phi_range and file_name have different lengths, execution halted.')
    if rspmeta.phi_count.shape[0] != rspmeta.file_name.__len__():
        raise Exception('phi_count and file_name have different lengths, execution halted.')
    if rspmeta.theta_range.shape[0] != rspmeta.file_name.__len__():
        raise Exception('theta_range and file_name have different lengths, execution halted.')
    if rspmeta.theta_count.shape[0] != rspmeta.file_name.__len__():
        raise Exception('theta_count and file_name have different lengths, execution halted.')

    rspm = pd.DataFrame({"id": rspmeta.id,
                        "file_name": rspmeta.file_name,
                        "file_dir": rspmeta.file_dir,
                        "x_utm11n": rspmeta.origin[:, 0],
                        "y_utm11n": rspmeta.origin[:, 1],
                        "elevation_m": rspmeta.origin[:, 2],
                        "src_las_file": vox.las_in,
                        "vox_step": vox.step[0],
                        "vox_sample_length": vox.sample_length,
                        "src_return_set": vox.return_set,
                        "src_drop_class": vox.drop_class,
                        "agg_sample_length": rspmeta.agg_sample_length,
                        "min_distance_m": rspmeta.min_distance,
                        "max_distance_m": rspmeta.max_distance,
                        "phi_min_rad": rspmeta.phi_range[:, 0],
                        "phi_max_rad": rspmeta.phi_range[:, 1],
                        "phi_count": rspmeta.phi_count,
                        "theta_min_rad": rspmeta.theta_range[:, 0],
                        "theta_max_rad": rspmeta.theta_range[:, 1],
                        "theta_count": rspmeta.theta_count,
                        "lookup_db": rspmeta.lookup_db,
                        "config_id": rspmeta.config_id,
                        "agg_method": rspmeta.agg_method,
                        "prior": str(rspmeta.prior),
                        "created_datetime": None,
                        "computation_time_s": None})

    # resent indexing in case of rollover indexing
    rspm = rspm.reset_index(drop=True)

    # preallocate log file
    log_path = rspmeta.file_dir + "rsgmetalog.csv"
    if not os.path.exists(log_path):
        with open(log_path, mode='w', encoding='utf-8') as log:
            log.write(",".join(rspm.columns) + '\n')
        log.close()

    # export phi_theta_lookup of vectors in grid
    vector_set = photo_vectors(rspmeta.phi_range, rspmeta.phi_count[0], rspmeta.theta_range, rspmeta.theta_count[0])
    vector_set.to_csv(rspm.file_dir[0] + "phi_theta_lookup.csv", index=False)

    vox_sub = subset_vox(rspm.loc[:, ['x_utm11n', 'y_utm11n', 'elevation_m']].values, vox, rspmeta.max_distance, rspmeta.lookup_db)

    for ii in tqdm(range(0, len(rspm)), leave=True, ncols=100):
        it_time = time.time()

        iid = rspm.index[ii]

        origin = (rspm.x_utm11n.iloc[ii], rspm.y_utm11n.iloc[ii], rspm.elevation_m.iloc[ii])

        if rspmeta.agg_method == "single_ray_agg":
            # calculate rays
            rays_in = point_to_hemi_rays(origin, vox_sub, rshmeta.img_size, rshmeta.max_phi_rad, max_dist=rshmeta.max_distance, min_dist=rshmeta.min_distance)

            # # sample rays
            # vox_old = vox
            # vox = vox_sub
            # rays = rays_in
            # agg_sample_length = rshmeta.agg_sample_length
            # lookup_db = rshmeta.lookup_db
            rays_out = single_ray_agg(vox_sub, rays_in, rshmeta.agg_sample_length, lookup_db=rshmeta.lookup_db)
        elif rspmeta.agg_method == "vox_agg":

            # vox_bu = vox
            # vox = vox_sub
            # img_size = rshmeta.img_size
            # max_phi = rshmeta.max_phi_rad
            # max_dist = rshmeta.max_distance
            # min_dist = rshmeta.min_distance
            rays_out = vox_agg(origin, vox_sub, rshmeta.img_size, rshmeta.max_phi_rad, rshmeta.max_distance, rshmeta.min_distance)
        else:
            raise Exception("Unknown agg_method: " + rshmeta.agg_method)


        # format to image
        template = np.full((rshmeta.img_size, rshmeta.img_size, 2), np.nan)
        template[(rays_out.y_index.values, rays_out.x_index.values, 0)] = rays_out.returns_mean
        template[(rays_out.y_index.values, rays_out.x_index.values, 1)] = rays_out.returns_std
        # write image
        tiff.imsave(rshm.file_dir.iloc[ii] + rshm.file_name.iloc[ii], template)

        # log meta

        rshm.loc[iid, "created_datetime"] = time.strftime('%Y-%m-%d %H:%M:%S')
        rshm.loc[iid, "computation_time_s"] = int(time.time() - it_time)

        # write to log file
        rshm.iloc[ii:ii + 1].to_csv(log_path, encoding='utf-8', mode='a', header=False, index=False)

    return rspm
