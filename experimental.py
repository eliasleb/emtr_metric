import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import re
import pickle
import sys
from functools import reduce


# REGEXP_FLOAT = r"[+-]?(\d+([.]\d*)?([eE][+-]?\d+)?|[.]\d+([eE][+-]?\d+)?)"
REGEXP_FLOAT = r"([+-]?(\d+([.]\d*)?([eE][+-]?\d+)?|[.]\d+([eE][+-]?\d+)?))"


def read_data(filename):
    data = pd.read_csv(
        filename,
        skiprows=range(8)
    )
    names = " ".join(data.columns)
    t = np.array([
        float(x[0]) for x in re.findall(rf"@ t={REGEXP_FLOAT}", names)
    ])
    assert t.size == data.columns.size - 2
    x, y = np.array(data.iloc[:, 0])[:, np.newaxis], np.array(data.iloc[:, 1])[:, np.newaxis]
    field = np.array(data.iloc[:, 2:])
    return t, x, y, field


def read_data_helper_cst_studio(filename, dt=0.01):

    data = pd.read_csv(
        filename,
        delim_whitespace=True,
        skiprows=range(3),
        names=["x", "y", "z", "fx", "fy", "fz"]
    )
    data = np.array([pd.to_numeric(col, errors="coerce") for col in data.values])
    is_bad_row = np.any(np.isnan(data), axis=1)
    n_samples = np.sum(is_bad_row) + 1
    t = np.linspace(0, dt * (n_samples - 1), n_samples)
    data = data[np.invert(is_bad_row), :]
    n_pos = np.where(is_bad_row)[0][0]
    assert n_pos * n_samples == data.shape[0]
    data = data.reshape((n_samples, n_pos, 6))
    return t, data[:, :, :3], data[:, :, -3:]


def read_and_process(filename_e, filename_h):
    t, pos, e_field, h_field, n_points_row = 0., None, None, None, None
    try:
        max_e_field = np.genfromtxt("Res_01/max_e_field.txt")
    except FileNotFoundError:
        max_e_field = None
    with open(filename_e) as fd_e:
        with open(filename_h) as fd_h:
            while True:
                line_e, line_h = fd_e.readline(), fd_h.readline()
                if not line_e:
                    break
                matches = re.findall(rf"time=({REGEXP_FLOAT})", line_e)
                if matches:
                    t = float(matches[0][0])
                    if e_field is not None:
                        n_points_row, max_e_field = process_slice(
                            t, pos, e_field, h_field, n_points_row, max_e_field=max_e_field
                        )
                    e_field, h_field, pos = [], [], []
                    continue
                matches_e, matches_h = re.findall(REGEXP_FLOAT, line_e), re.findall(REGEXP_FLOAT, line_h)
                if len(matches_e) != 6:
                    continue
                r = [float(xi[0]) for xi in matches_e[:3]]
                e_i, h_i = [float(xi[0]) for xi in matches_e[3:]], [float(xi[0]) for xi in matches_h[3:]]
                pos.append(r)
                e_field.append(e_i)
                h_field.append(h_i)
    np.savetxt(
        "Res_01/max_e_field.txt",
        max_e_field.ravel()
    )


def process_slice(t, pos, e, h, shape=None, downsample=1, max_e_field=None, dim_changing=0, order="F"):
    pos, e, h = np.array(pos), np.array(e), np.array(h)
    if np.max(np.abs(e)) < 1e-18:
        print("Skipping...")
        return shape, max_e_field
    if shape is None:
        dx = pos[1, dim_changing] - pos[0, dim_changing]
        shape = tuple([int(si) for si in np.floor((np.max(pos, axis=0) - np.min(pos, axis=0)) / dx + 1)])
        assert reduce(lambda x, y: x * y, shape) == pos.shape[0]
    pos = pos.reshape(shape + (3, ), order=order)
    e = e.reshape(shape + (3, ), order=order)
    h = h.reshape(shape + (3, ), order=order)
    eps_0, mu_0 = 8.854e-12, 4 * np.pi * 1e-7
    norm_e2 = np.sum(e**2, axis=-1)
    norm_b2 = np.sum((mu_0 * h)**2, axis=-1)
    energy_density = 1/2 * eps_0 * norm_e2 + 1/2 / mu_0 * norm_b2
    norm_e = np.sqrt(norm_e2)
    if max_e_field is None:
        max_e_field = np.zeros(norm_e.shape)
    max_e_field = np.maximum(max_e_field, norm_e)
    # mask = max_e_field > 3e11
    # source_pos, r_mask = (15, 6.5, ), 2
    # mask = (pos[:, :, 0] - source_pos[0])**2 + (pos[:, :, 2] - source_pos[1])**2 < r_mask**2
    # norm_e[mask], energy_density[mask] = 0., 0.

    total_energy = np.sum(energy_density)
    mean_pos = np.sum(pos * energy_density[:, :, :, None], axis=(0, 1, 2)) / total_energy
    mean_pos2 = np.sum(pos**2 * energy_density[:, :, :, None], axis=(0, 1, 2)) / total_energy
    mean_xy = np.sum(pos[:, :, :, 0] * pos[:, :, :, 1] * energy_density) / total_energy
    entropy = np.mean(norm_e**2, axis=(0, 1, 2))**2 / np.mean(norm_e**4, axis=(0, 1, 2))
    mean_e_space = np.mean(norm_e, axis=(0, 1, 2))
    space_kurtosis = np.mean((norm_e - mean_e_space)**2, axis=(0, 1, 2))**2 \
                     / np.mean((norm_e - mean_e_space)**4, axis=(0, 1, 2))
    np.set_printoptions(threshold=sys.maxsize)
    to_print = norm_e[::downsample, ::downsample]
    print(
        t,
        total_energy,
        mean_pos[0], mean_pos[1], mean_pos[2],
        mean_pos2[0], mean_pos2[1], mean_pos2[2],
        mean_xy,
        entropy,
        space_kurtosis,
        to_print.shape[0], to_print.shape[1], to_print.shape[2],
        sep=",",
        end=","
    )
    for ei in to_print.ravel():
        print(ei, end=",")
    print()

    index_middle = shape[2] // 2

    plt.clf()
    plt.subplot(2, 1, 1)
    plt.contourf(
        pos[::downsample, ::downsample, index_middle, 0], pos[::downsample, ::downsample, index_middle, 1],
        norm_e[::downsample, ::downsample, index_middle],
        levels=np.linspace(0, 1e11, 11),
        cmap=plt.get_cmap("jet")
    )
    plt.plot((mean_pos[0], ), (mean_pos[1], ), "rx", markersize=10)
    plt.colorbar()
    plt.title(f"t = {t*1e3:.2f} ns")
    plt.subplot(2, 1, 2)
    plt.contourf(
        pos[::downsample, ::downsample, index_middle, 0], pos[::downsample, ::downsample, index_middle, 1],
        max_e_field[::downsample, ::downsample, index_middle],
        cmap=plt.get_cmap("jet"),
        # levels=np.linspace(0, 1, 11)
    )
    plt.colorbar()
    plt.waitforbuttonpress(.0001)
    return shape, max_e_field


def re_normalize(x, a=0., b=1.):
    return (x - np.min(x)) / (np.max(x) - np.min(x)) * (b - a) + a


def location_error(pos, source_loc, array_max, case="3D"):
    if case == "3D":
        arg = np.unravel_index(np.nanargmax(array_max), array_max.shape)
        prediction = pos[0, arg[0], arg[1], arg[2], :]
    else:
        arg = np.nanargmax(array_max)
        prediction = np.array((pos[0][arg], pos[1][arg]))

    return np.sqrt(np.sum((source_loc - prediction)**2))


def experimental(filename_e, filename_h, dx=0.5, order="F", t_min=5., t_max=10.):
    try:
        with open("save.pickle", "rb") as fd:
            t, pos, e, h = pickle.load(fd)

            ######
            t = t * .1

    except FileNotFoundError:
        t, pos, e = read_data_helper_cst_studio(filename_e)
        print("Done importing E")
        t_h, pos_h, h = read_data_helper_cst_studio(filename_h)
        print("Done importing H")

        assert t.shape == t_h.shape
        assert pos.shape == pos_h.shape
        assert e.shape == h.shape

        with open("save.pickle", "wb") as fd:
            pickle.dump((t, pos, e, h), fd)

    keep = (t > t_min) * (t < t_max)
    t, pos, e, h = t[keep], pos[keep, ...], e[keep, ...], h[keep, ...]

    ranges = np.max(pos, axis=(0, 1,)) - np.min(pos, axis=(0, 1,))
    shape = (ranges/dx + 1).astype(int)
    assert np.prod(shape) == pos.shape[1]
    shape = tuple([dim for dim in shape])
    shape = (t.size, ) + shape + (3, )
    pos, e, h = pos.reshape(shape, order=order), e.reshape(shape, order=order), h.reshape(shape, order=order)
    eps_0, mu_0 = 8.854e-12, 4 * np.pi * 1e-7
    energy_density = 1/2 * eps_0 * np.sum(e**2, axis=-1) + 1/2 * mu_0 * np.sum(h**2, axis=-1)
    norm_e = np.sqrt(np.sum(e**2, axis=-1))

    mask = ((pos[..., 0] - 15)**2 + (pos[..., 2] - 6.5)**2 <= 2**2) * (pos[..., 1] > 10)
    energy_density[mask] = 0.
    norm_e[mask] = 0.

    ind_middle = pos.shape[3] // 2
    max_norm_e_t = np.max(norm_e, axis=0)
    max_norm_e_x = np.max(norm_e, axis=(1, 2, 3))

    total_energy = np.nansum(energy_density, axis=(1, 2, 3)).squeeze()
    expected_pos = np.sum(pos * energy_density[..., None], axis=(1, 2, 3)) / total_energy[..., None]
    energy_location_variance = np.nansum(pos**2 * energy_density[..., np.newaxis], axis=(1, 2, 3)) / \
                               total_energy[..., None] - expected_pos**2
    metric = np.sqrt(np.sum(energy_location_variance, axis=-1))
    ind_metric = np.nanargmin(metric)
    n_space = shape[1] * shape[2] * shape[3]
    entropy = n_space * np.nansum(norm_e**2, axis=(1, 2, 3))**2 / np.nansum(norm_e**4, axis=(1, 2, 3))
    ind_entropy = np.nanargmin(entropy)

    e_e_x = np.nanmean(norm_e, axis=(1, 2, 3))[:, None, None, None]
    e_e_t = np.nanmean(norm_e, axis=0)[None, ...]
    space_kurtosis = n_space * np.nansum((norm_e - e_e_x) ** 4, axis=(1, 2, 3))\
                     / np.nansum((norm_e - e_e_x) ** 2, axis=(1, 2, 3)) ** 2
    time_kurtosis = t.size * np.nansum((norm_e - e_e_t) ** 4, axis=0) / np.nansum((norm_e - e_e_t) ** 2, axis=0)**2
    ind_space_kurtosis = np.nanargmax(space_kurtosis)
    ind_max_x = np.nanargmax(max_norm_e_x)

    t_focus = 8.43
    print(
        f"Focusing time error energy loc var = {t_focus - t[ind_metric]}"
    )
    print(
        f"Focusing time error entropy = {t_focus - t[ind_entropy]}"
    )
    print(
        f"Focusing time error kurt = {t_focus - t[ind_space_kurtosis]}"
    )
    print(
        f"Focusing time error max = {t_focus - t[ind_max_x]}"
    )

    # indices_middle = [[0, ] * 3, [s // 2 for s in pos.shape[1:-1]], [s - 1 for s in pos.shape[1:-1]]]
    # plt.figure(figsize=(15, 12))
    # names = "xyz"
    # for i, ti in enumerate(t):
    #     if i % 2 != 0:
    #         continue
    #     plt.clf()
    #     sample_max = np.max(energy_density[i, ...])
    #     for plt_ind, ind2 in enumerate(indices_middle):
    #         for dim, ind_mid in enumerate(ind2):
    #             plt.subplot(3, 3, dim + 1 + 3 * plt_ind)
    #             dims = list(range(3))
    #             dims.remove(dim)
    #             s = [slice(i, i+1), ] + [slice(None), ] * 3
    #             s[dim + 1] = slice(ind_mid, ind_mid + 1)
    #             s = tuple(s)
    #             plt.contourf(
    #                 pos[s + (dims[0], )].squeeze(), pos[s + (dims[1], )].squeeze(),
    #                 energy_density[s].squeeze(),
    #                 levels=np.linspace(0, sample_max, 11),
    #                 cmap="jet"
    #             )
    #             plt.xlabel(names[dims[0]])
    #             plt.ylabel(names[dims[1]])
    #             plt.plot(expected_pos[i, dims[0]], expected_pos[i, dims[1]], "rx", markersize=10)
    #             if dim == 1 and plt_ind == 0:
    #                 plt.title(f"t = {ti:.2f} ns ({i=})")
    #             if dim == 2 and plt_ind == 1:
    #                 plt.colorbar()
    #     plt.tight_layout()
    #     plt.waitforbuttonpress()

    source_loc = np.array((9, 16.4, 6.5))

    print(
        f"Loc error max = {location_error(pos, source_loc, max_norm_e_t)}"
    )
    print(
        f"Loc error entropy = {location_error(pos, source_loc, energy_density[ind_entropy, ...])}"
    )
    print(
        f"Loc error space kurt = {location_error(pos, source_loc, energy_density[ind_space_kurtosis, ...])}"
    )
    print(
        f"Loc error time kurt = {location_error(pos, source_loc, time_kurtosis)}"
    )
    print(
        f"Loc error energy loc var = {location_error(pos, source_loc, energy_density[ind_metric, ...])}"
    )
    print(
        f"Loc error avg energy loc = {np.sqrt(np.sum((source_loc - expected_pos[ind_metric, ...])**2))}"
    )
    print(
        f"Loc error max x = {location_error(pos, source_loc, energy_density[ind_max_x, ...])}"
    )

    plt.figure(figsize=(5, 3.2))
    ax = plt.gca()
    z_plot = (metric - np.nanmin(metric))
    z_plot = z_plot / np.nanmax(z_plot)
    z_plot[np.isnan(z_plot)] = 1.

    ind_sort = np.argsort(z_plot)[::-1]

    for i, (t_i, y_i, z_i) in enumerate(zip(t, metric, 1-z_plot)):
        if i == 0:
            continue
        plt.plot((t[i-1], t_i, ), (metric[i-1], y_i, ),
                 color=plt.get_cmap("jet")(z_i))
    plt.ylabel("Energy location standard deviation (cm)")
    plt.plot((t[ind_metric], ), (metric[ind_metric], ), "r.", markersize=10)
    # plt.ylim(.9 * np.nanmin(metric), 1.1 * np.nanmax(metric))
    ax2 = ax.twinx()
    scale = 1e-12
    ax2.plot(t, total_energy * dx * dx * 1e-4 * scale**2 * 1e15, "r--")
    ax2.set_ylabel("Domain energy (fJ)")
    # ax2.set_ylim(0, 150)
    ax.set_xlabel("Time (ns)")
    plt.xlim(t_min, t_max)

    plt.tight_layout()
    plt.savefig("energy_loc_std_vs_time_exp.pdf")

    plt.figure(figsize=(5, 3.2))
    plt.plot(t, entropy, "k-")
    # plt.ylim(.9 * np.nanmin(entropy), 1.1 * np.nanmax(entropy))
    plt.xlabel("Time (ns)")
    plt.xlim(t_min, t_max)
    plt.title("Entropy")
    plt.plot((t[ind_entropy], ), (entropy[ind_entropy], ), "r.", markersize=10)
    plt.tight_layout()
    plt.savefig("entropy_vs_time_exp.pdf")

    plt.figure(figsize=(5, 3.2))
    plt.plot(t, space_kurtosis, "k-")
    plt.xlabel("Time (ns)")
    plt.xlim(t_min, t_max)
    plt.title("Space kurtosis")
    plt.plot((t[ind_space_kurtosis], ), (space_kurtosis[ind_space_kurtosis], ), "r.", markersize=10)
    plt.tight_layout()
    plt.savefig("space_kurt_vs_time_exp.pdf")

    plt.figure(figsize=(5, 3.2))
    plt.plot(t, max_norm_e_x * scale, "k-")
    plt.xlabel("Time (ns)")
    plt.xlim(t_min, t_max)
    plt.title("Maximum electric field norm (V/m)")
    plt.plot((t[ind_max_x], ), (max_norm_e_x[ind_max_x] * scale, ), "r.", markersize=10)
    plt.tight_layout()
    plt.savefig("max_vs_time_exp.pdf")

    plt.figure(figsize=(5, 3.2))
    z_contour = energy_density[ind_metric, :, :, ind_middle].copy()
    contours = plt.contour(
        pos[ind_metric, :, :, ind_middle, 0], pos[ind_metric, :, :, ind_middle, 1], z_contour * scale**2 * 1e-15,
        levels=6,
        colors="k",
        linewidths=.8
    )
    ax = plt.gca()
    ax.clabel(contours, contours.levels, inline=True, zorder=9, fontsize=7)
    plt.contourf(
        pos[ind_metric, :, :, ind_middle, 0], pos[ind_metric, :, :, ind_middle, 1],
        energy_density[ind_metric, :, :, ind_middle],
        cmap=plt.get_cmap("jet"),
        levels=30
    )
    plt.scatter(
        expected_pos[ind_sort, 0], expected_pos[ind_sort, 1],
        c=1-z_plot[ind_sort],
        cmap=plt.get_cmap("jet"),
        marker=".",
        zorder=10
    )
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title(f"Energy density (fW/m3) at t = {t[ind_metric]:.1f} ns")
    plt.tight_layout()
    plt.savefig("field_at_min_std_exp.pdf")

    plt.figure(figsize=(5, 3.2))
    z_contour = time_kurtosis.copy()
    contours = plt.contour(
        pos[ind_space_kurtosis, :, :, ind_middle, 0], pos[ind_space_kurtosis, :, :, ind_middle, 1],
        z_contour[..., ind_middle],
        levels=6,
        colors="k",
        linewidths=.8
    )
    ax = plt.gca()
    ax.clabel(contours, contours.levels, inline=True, zorder=9, fontsize=7)
    plt.contourf(
        pos[ind_space_kurtosis, :, :, ind_middle, 0], pos[ind_space_kurtosis, :, :, ind_middle, 1],
        time_kurtosis[..., ind_middle],
        cmap=plt.get_cmap("jet"),
        levels=30,
    )
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title(f"Time kurtosis (1)")
    plt.tight_layout()
    plt.savefig("time_kurt_exp.pdf")

    plt.figure(figsize=(5, 3.2))
    z_contour = energy_density[ind_entropy, :, :, ind_middle].copy()
    contours = plt.contour(
        pos[ind_space_kurtosis, :, :, ind_middle, 0], pos[ind_space_kurtosis, :, :, ind_middle, 1],
        z_contour * scale**2 * 1e-15,
        levels=6,
        colors="k",
        linewidths=.8
    )
    ax = plt.gca()
    ax.clabel(contours, contours.levels, inline=True, zorder=9, fontsize=7)

    plt.contourf(
        pos[ind_space_kurtosis, :, :, ind_middle, 0], pos[ind_space_kurtosis, :, :, ind_middle, 1],
        energy_density[ind_entropy, :, :, ind_middle],
        cmap=plt.get_cmap("jet"),
        levels=30
    )
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title(f"Energy density (fJ/m3) at t = {t[ind_entropy]:.1f} ns")
    plt.tight_layout()
    plt.savefig("field_at_min_entropy_exp.pdf")

    plt.figure(figsize=(5, 3.2))
    z_contour = max_norm_e_t.copy()
    contours = plt.contour(
        pos[ind_space_kurtosis, :, :, ind_middle, 0], pos[ind_space_kurtosis, :, :, ind_middle, 1],
        z_contour[..., ind_middle]*scale,
        levels=np.linspace(.2, .5, 4),
        colors="k",
        linewidths=.8
    )
    ax = plt.gca()
    ax.clabel(contours, contours.levels, inline=True, zorder=9, fontsize=7)

    plt.contourf(
        pos[ind_space_kurtosis, :, :, ind_middle, 0], pos[ind_space_kurtosis, :, :, ind_middle, 1],
        max_norm_e_t[..., ind_middle] * scale,
        cmap=plt.get_cmap("jet"),
        levels=30
    )
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title("Maximum electric field norm (V/m)")

    plt.tight_layout()
    plt.savefig("max_e_field_exp.pdf")

    plt.show()


def read_data_comsol():
    t, x, y, w = read_data(filename="../../git_ignore/CEM2023/tr_w_8_sensors.txt")
    _, _, _, e = read_data(filename="../../git_ignore/CEM2023/tr_e_8_sensors.txt")
    dx = float((x[1] - x[0])[0])
    total_energy = np.nansum(w, axis=0)
    total_energy[0] = 1e-18
    e[:, 0] = 1e-18
    nx, ny = 161, 71
    n = nx * ny
    assert n == x.size

    e_x, e_y = np.sum(x * w, axis=0) / total_energy, np.sum(y * w, axis=0) / total_energy
    e_xx, e_yy = np.sum(x**2 * w, axis=0) / total_energy, np.sum(y**2 * w, axis=0) / total_energy
    var_x, var_y = e_xx - e_x**2, e_yy - e_y**2
    std_x, std_y = np.sqrt(var_x), np.sqrt(var_y)

    e_e_x = np.nanmean(e, axis=0)
    e_e_t = np.nanmean(e, axis=1)[:, np.newaxis]
    space_kurtosis = n * np.nansum((e - e_e_x) ** 4, axis=0) / np.nansum((e - e_e_x) ** 2, axis=0) ** 2
    time_kurtosis = t.size * np.nansum((e - e_e_t) ** 4, axis=1) / np.nansum((e - e_e_t) ** 2, axis=1) ** 2

    entropy = n * np.nansum(e**2, axis=0)**2 / np.nansum(e**4, axis=0)

    aggregate = np.sqrt(std_x * std_y)

    t_cutoff = 27e-9
    aggregate[t < t_cutoff] = np.nan
    space_kurtosis[t < t_cutoff] = np.nan
    entropy[t < t_cutoff] = np.nan
    total_energy[t < t_cutoff] = np.nan

    w = np.reshape(w, (ny, nx, t.size))
    e = np.reshape(e, (ny, nx, t.size))
    max_e_t = np.max(e, axis=-1)
    max_e_x = np.max(e, axis=(0, 1))

    i_aggregate = np.nanargmin(aggregate)
    i_kurtosis = np.nanargmax(space_kurtosis)
    i_entropy = np.nanargmin(entropy)
    i_max = np.argmax(max_e_x)

    time_kurtosis = np.reshape(time_kurtosis, (ny, nx, ))
    x2, y2 = np.linspace(np.min(x), np.max(x), nx), np.linspace(np.min(y), np.max(y), ny)
    t_min, t_max = 30, 59
    z_plot = (aggregate - np.nanmin(aggregate))
    z_plot = z_plot / np.nanmax(z_plot)
    z_plot[np.isnan(z_plot)] = 1.
    ind_sort = np.argsort(z_plot)[::-1]
    gamma = 1.31  # ns
    sim_duration = 30 * gamma
    t_star = 3 * gamma
    t_focus = sim_duration - t_star
    pos = (x.squeeze(), y.squeeze(), )
    x_src, y_src = .3, .2
    source_loc = np.array((x_src, y_src, ))
    mask = np.invert(
        (x > .44) * (y < .211) * (x < .7) + (x > 1.1) * (x < 1.3) * (y > .2) * (y < .4)
    )
    mask = np.reshape(mask, (ny, nx))
    print(
        f"Focusing time error energy loc var = {t_focus - t[i_aggregate]*1e9}"
    )
    print(
        f"Focusing time error entropy = {t_focus - t[i_entropy]*1e9}"
    )
    print(
        f"Focusing time error kurt = {t_focus - t[i_kurtosis]*1e9}"
    )
    print(
        f"Focusing time error for max = {t_focus - t[i_max]*1e9}"
    )
    print(
        f"Loc error max = {1e2*location_error(pos, source_loc, max_e_t * mask, case='2D')}"
    )
    print(
        f"Loc error entropy = {1e2*location_error(pos, source_loc, w[..., i_entropy] * mask, case='2D')}"
    )
    print(
        f"Loc error space kurt = {1e2*location_error(pos, source_loc, w[..., i_kurtosis] * mask, case='2D')}"
    )
    print(
        f"Loc error time kurt = {1e2*location_error(pos, source_loc, time_kurtosis * mask, case='2D')}"
    )
    print(
        f"Loc error energy loc var = {1e2*location_error(pos, source_loc, w[..., i_aggregate] * mask, case='2D')}"
    )
    print(
        f"Loc error max x = {1e2*location_error(pos, source_loc, w[..., i_max] * mask, case='2D')}"
    )
    print(
        f"Loc error avg energy loc = {1e2*np.sqrt((e_x[i_aggregate] - x_src)**2 + (e_y[i_aggregate] - y_src)**2)}"
    )

    plt.figure(figsize=(5, 3.2))
    plt.plot(
        (t_focus, t_focus, ),
        (.9 * np.nanmin(aggregate), 1.1 * np.nanmax(aggregate), ),
        "b-"
    )
    ax = plt.gca()
    # plt.plot(t*1e9, aggregate, "-", c=1-z_plot)
    for i, (t_i, y_i, z_i) in enumerate(zip(t, aggregate, 1-z_plot)):
        if i == 0:
            continue
        plt.plot((t[i-1]*1e9, t_i*1e9, ), (aggregate[i-1], y_i, ),
                 color=plt.get_cmap("jet")(z_i))
    plt.ylabel("Energy location standard deviation (m)")
    plt.plot((t[i_aggregate]*1e9, ), (aggregate[i_aggregate], ), "r.", markersize=10)
    plt.ylim(.9 * np.nanmin(aggregate), 1.1 * np.nanmax(aggregate))
    ax2 = ax.twinx()
    ax2.plot(t*1e9, total_energy * dx * dx, "r--")
    ax2.set_ylabel("Domain energy (pJ)")
    ax2.set_ylim(0, 150)
    ax2.set_xlim(t_min, t_max)
    ax.set_xlabel("Time (ns)")
    plt.tight_layout()
    plt.savefig("energy_loc_std_vs_time.pdf")

    plt.figure(figsize=(5, 3.2))
    plt.plot(
        (t_focus, t_focus, ),
        (.9 * np.nanmin(entropy), 1.1 * np.nanmax(entropy), ),
        "b-"
    )
    plt.plot(t*1e9, entropy, "k-")
    plt.ylim(.9 * np.nanmin(entropy), 1.1 * np.nanmax(entropy))
    plt.xlabel("Time (ns)")
    plt.xlim(t_min, t_max)
    plt.title("Entropy (1)")
    plt.plot((t[i_entropy]*1e9, ), (entropy[i_entropy], ), "r.", markersize=10)
    plt.tight_layout()
    plt.savefig("entropy_vs_time.pdf")

    plt.figure(figsize=(5, 3.2))
    plt.plot(
        (t_focus, t_focus, ),
        (.9 * np.nanmin(space_kurtosis), 1.1 * np.nanmax(space_kurtosis), ),
        "b-"
    )
    plt.ylim(.9 * np.nanmin(space_kurtosis), 1.1 * np.nanmax(space_kurtosis))
    plt.plot(t*1e9, space_kurtosis, "k-")
    plt.xlabel("Time (ns)")
    plt.xlim(t_min, t_max)
    plt.title("Space kurtosis (1)")
    plt.plot((t[i_kurtosis]*1e9, ), (space_kurtosis[i_kurtosis], ), "r.", markersize=10)
    plt.tight_layout()
    plt.savefig("space_kurt_vs_time.pdf")

    plt.figure(figsize=(5, 3.2))
    z_contour = w[:, :, i_aggregate].copy()
    z_contour[
        (x2[:, np.newaxis].T > .5)*(x2[:, np.newaxis].T < .7)*(y2[np.newaxis, :].T > 0)*(y2[np.newaxis, :].T < .2)
    ] = np.nan
    contours = plt.contour(
        x2, y2, z_contour,
        levels=np.arange(0, 800, 100),
        colors="k",
        linewidths=.8
    )
    ax = plt.gca()
    ax.clabel(contours, contours.levels, inline=True, zorder=9, fontsize=7)
    plt.contourf(
        x2, y2, w[:, :, i_aggregate]/np.nanmax(w[:, :, i_aggregate])*2,
        cmap=plt.get_cmap("jet"),
        levels=np.linspace(0, 1)
    )
    plt.scatter(
        e_x[ind_sort], e_y[ind_sort],
        c=1-z_plot[ind_sort],
        cmap=plt.get_cmap("jet"),
        marker=".",
        zorder=10
    )
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    x_src, y_src = .3, .2
    plt.plot((x_src, ), (y_src, ), "rx")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"Energy density (pW/m3) at t = {t[i_aggregate]*1e9:.1f} ns")
    plt.tight_layout()
    plt.savefig("field_at_min_std.pdf")

    plt.figure(figsize=(5, 3.2))
    z_contour = time_kurtosis.copy()
    z_contour[
        (x2[:, np.newaxis].T > .5)*(x2[:, np.newaxis].T < .7)*(y2[np.newaxis, :].T > 0)*(y2[np.newaxis, :].T < .2)
    ] = np.nan
    contours = plt.contour(
        x2, y2, z_contour,
        levels=np.arange(0, 7, 1),
        colors="k",
        linewidths=.8
    )
    ax = plt.gca()
    ax.clabel(contours, contours.levels, inline=True, zorder=9, fontsize=7)
    plt.contourf(
        x2, y2, time_kurtosis,
        cmap=plt.get_cmap("jet"),
        levels=np.linspace(0, 7)
    )
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    plt.plot((x_src, ), (y_src, ), "bx")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"Time kurtosis (1)")
    plt.tight_layout()
    plt.savefig("time_kurt.pdf")

    plt.figure(figsize=(5, 3.2))
    z_contour = w[:, :, i_entropy].copy()
    z_contour[
        (x2[:, np.newaxis].T > .5)*(x2[:, np.newaxis].T < .7)*(y2[np.newaxis, :].T > 0)*(y2[np.newaxis, :].T < .2)
    ] = np.nan
    contours = plt.contour(
        x2, y2, z_contour,
        levels=np.arange(0, 450, 50),
        colors="k",
        linewidths=.8
    )
    ax = plt.gca()
    ax.clabel(contours, contours.levels, inline=True, zorder=9, fontsize=7)

    plt.contourf(
        x2, y2, w[:, :, i_entropy],
        cmap=plt.get_cmap("jet"),
        levels=30
    )
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    x_src, y_src = .3, .2
    plt.plot((x_src, ), (y_src, ), "rx")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"Energy density (pW/m3) at t = {t[i_entropy]*1e9:.1f} ns")
    plt.tight_layout()
    plt.savefig("field_at_min_entropy.pdf")

    plt.figure(figsize=(5, 3.2))
    z_contour = max_e_t.copy()
    z_contour[
        (x2[:, np.newaxis].T > .5)*(x2[:, np.newaxis].T < .7)*(y2[np.newaxis, :].T > 0)*(y2[np.newaxis, :].T < .2)
    ] = np.nan
    z_contour[
        (x2[:, np.newaxis].T > 1)*(x2[:, np.newaxis].T < 1.3)*(y2[np.newaxis, :].T > 0.199)*(y2[np.newaxis, :].T < .39)
    ] = np.nan
    contours = plt.contour(
        x2, y2, z_contour/1e6,
        levels=np.linspace(0, 14, 8),
        colors="k",
        linewidths=.8
    )
    ax = plt.gca()
    ax.clabel(contours, contours.levels, inline=True, zorder=9, fontsize=7)

    plt.contourf(
        x2, y2, max_e_t,
        cmap=plt.get_cmap("jet"),
        levels=30
    )
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Maximum electric field norm (V/m)")
    plt.plot((x_src, ), (y_src, ), "rx")

    plt.tight_layout()
    plt.savefig("max_e_field.pdf")

    plt.figure(figsize=(5, 3.2))
    max_e_x = max_e_x / 1e6
    plt.plot(
        (t_focus, t_focus, ),
        (.9 * np.nanmin(max_e_x), 1.1 * np.nanmax(max_e_x), ),
        "b-"
    )
    plt.ylim(.9 * np.nanmin(max_e_x), 1.1 * np.nanmax(max_e_x))
    plt.plot(t*1e9, max_e_x, "k-")
    plt.xlabel("Time (ns)")
    plt.xlim(t_min, t_max)
    plt.title("Maximum electric field norm (MV/m)")
    plt.plot((t[i_max]*1e9, ), (max_e_x[i_max], ), "r.", markersize=10)
    plt.tight_layout()
    plt.savefig("max_vs_time.pdf")

    plt.figure()

    plt.contourf(
        x2, y2, w[..., i_kurtosis],
        cmap=plt.get_cmap("jet"),
    )
    plt.savefig("field_at_min_kurtosis.pdf")

    plt.show()


def read_cst_excitation(filename):
    data = pd.read_csv(
        filename,
        delim_whitespace=True,
        names=["t", "x"],
        skiprows=range(1)
    )
    t, x = np.array(data.t), np.array(data.x)
    x = x / np.max(np.abs(x))
    t = t * 1e9

    plt.figure(figsize=(5, 3.2))

    plt.plot(t, np.flip(x), "k-")
    plt.xlim(np.min(t), np.max(t))
    plt.xlabel("Time (ns)")
    plt.title("Normalized time-reversal mirror measurement")

    plt.tight_layout()
    plt.savefig("excitation.pdf")
    plt.show()


def main():
    read_cst_excitation("signal$1.isf")
    experimental(
        "../../git_ignore/CEM2023/e-field.txt",
        "../../git_ignore/CEM2023/h-field.txt"
    )
    read_data_comsol()


if __name__ == "__main__":
    matplotlib.use("TkAgg")
    plt.rcParams["font.family"] = "EB Garamond"

    main()
