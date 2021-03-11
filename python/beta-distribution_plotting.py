import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    sns.set_palette("deep", desat=.6)
    sns.set_context(rc={"figure.figsize": (8, 4)})
    x = np.linspace(0, 1, 10000)
    params = [
        # (0.5, 0.5),
        # (1, 1),
        # (4, 3),
        # (2, 5),
        # (6, 6)
        (0.00944653, 0.62285121),  # 149 (snow free)
        (0.00952118, 0.90645839)   # 045-050-052 (snow on combined)
    ]
    neq = []
    for p in params:
        y = beta.pdf(x, p[0], p[1])
        plt.plot(x, y, label="$\\alpha=%s$, $\\beta=%s$" % p)
        # calculate equivalent sample size
        neq.append(p[0] + p[1] + 1)
    plt.xlabel("$\\theta$, Fairness")
    plt.ylabel("Density")
    plt.legend(title="Parameters")
    plt.show()
