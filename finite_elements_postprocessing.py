import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from main import min_of_quadratic, DerivedGaussianExcitation


def read_comsol_export(filename):
    data = pd.read_csv(
        filename,
        skiprows=range(4),
        delim_whitespace=True
    )
    # data.drop(labels=data.columns[4:], inplace=True, axis=1)
    return np.array(data)


def postprocessing():
    d = read_comsol_export("../../../git_ignore/CEM2023/gaussian.txt")

    t_s, sigma_s2 = (10 - 3.93)*1e-9, 0

    t = d[:, 0]
    sigma2 = d[:, 6]/d[:, 4]

    t_cutoff = 11e-9
    indices = t < t_cutoff
    t, sigma2 = t[indices], sigma2[indices]
    fit = np.polyfit(t, sigma2, 2)
    t_min, val_min = min_of_quadratic(fit)
    plt.figure()
    plt.plot(t*1e9, sigma2, "k-")
    plt.plot(t_min*1e9, val_min, "ro")
    plt.plot(t*1e9, np.polyval(fit, t), "r--")
    plt.plot(t_s*1e9, sigma_s2, "kx")
    plt.show()


if __name__ == "__main__":
    matplotlib.use("TkAgg")
    postprocessing()
