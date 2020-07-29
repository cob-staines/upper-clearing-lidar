import numpy as np
import pandas as pd

# need to import las/traj results, do this later
# for not make up mock scenario

n_points = 1
n_min = 0
n_max = 10

las_traj = pd.DataFrame({'x0': np.random.random(n_points),
                         'y0': np.random.random(n_points),
                         'z0': np.random.random(n_points),
                         'x1': np.random.random(n_points),
                         'y1': np.random.random(n_points),
                         'z1': np.random.random(n_points)})
las_traj = las_traj * (n_max - n_min) + n_min

# begin algorithm
# p0 = source, p1 = last return
p0 = np.array(las_traj.loc[:, ['x0', 'y0', 'z0']])
p1 = np.array(las_traj.loc[:, ['x1', 'y1', 'z1']])

# calculate distance between source and last return
dist = np.sqrt(np.sum((p1 - p0) ** 2, axis=1))  # check axis, make sure correct number of points!

tracer_step = 1

# calc xyz step
xyz_step = (p1 - p0) * tracer_step/dist[:, np.newaxis]

# iterate until t_dist > np.max(dist)
t_dist = 0
ii = 0
pointlist = []
while t_dist < np.max(dist):
    ii = ii + 1
    t_dist = ii * tracer_step

    selection = (dist > t_dist)
    pi = xyz_step[selection, :] * t_dist + p0[selection]
    if ii == 1:
        tracer_points = pi
    else:
        tracer_points = np.concatenate((tracer_points, pi), axis=0)
