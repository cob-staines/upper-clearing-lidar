import numpy as np

p0 = [0, 0, 0]
nn = 10000

samples = 3000

ratio = np.full(samples, np.nan)
for ii in range(0, samples):
    p1 = np.random.randint(-1000, 1000, 3)
    ll = np.sqrt(np.sum((p1 - p0) ** 2))

    t = np.arange(0, nn + 1)
    xyz_step = (p1 - p0)/nn

    xi = xyz_step[0] * t + p0[0]
    yi = xyz_step[1] * t + p0[1]
    zi = xyz_step[2] * t + p0[2]

    tracer = list(zip(xi.astype(int), yi.astype(int), zi.astype(int)))

    # number of cells traversed
    cell_count = list(set(tracer)).__len__()

    ratio[ii] = ll/cell_count

np.mean(ratio)

# compare with theoretical
2 * np.pi/(6 + np.pi)
