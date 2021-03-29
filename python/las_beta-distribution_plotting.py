

if __name__ == "__main__":
    import numpy as np
    from scipy.stats import beta
    import matplotlib.pyplot as plt
    import seaborn as sns

    import vox_045_050_052_config as vc
    vox = vc.vox

    sns.set_palette("deep", desat=.6)
    sns.set_context(rc={"figure.figsize": (8, 4)})
    x = np.linspace(0, 1, 10000)
    params = [
        # (0.5, 0.5),
        # (1, 1),
        # (4, 3),
        # (2, 5),
        # (6, 6)
        (vox.prior_alpha, vox.prior_beta)
        # (0.00944653, 0.62285121),  # 149 (snow free)
        # (0.00952118, 0.90645839)   # 045-050-052 (snow on combined)
    ]
    neq = []
    for p in params:
        y = beta.pdf(x, p[0], p[1])
        plt.plot(x, y, label="$\\alpha=%s$, $\\beta=%s$" % p)
        # calculate equivalent sample size
        neq.append(p[0] + p[1] + 1)
    plt.xlabel("$\\theta$, Probability")
    plt.ylabel("Density")
    plt.legend(title="Parameters")
    plt.show()

    # check posterior data types
    type(vox.posterior_alpha[0, 0, 0])
    type(vox.posterior_beta[0, 0, 0])

    # kk = path_returns
    # nn = path_samples * vox_sample_length / agg_sample_length
    kk = 0
    nn = 0
    weights = 1

    post_a = kk + vox.prior_alpha
    post_b = nn + kk + vox.prior_beta

    # normal approximation of sum
    returns_mean = weights * post_a/(post_a + post_b)
    returns_std = np.sqrt(weights * post_a * post_b / ((post_a + post_b) ** 2 * (post_a + post_b + 1)))

    # calculate returns for unsampled ray of 50m
    returns_mean * 50 / vox.agg_sample_length
    returns_std * 50 / vox.agg_sample_length

