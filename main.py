import numpy as np
import meep as mp
import matplotlib.pyplot as plt
import multiprocessing
import imageio.v3
import imageio
import os
import pickle
from abc import ABC, abstractmethod
import itertools
from matplotlib.gridspec import GridSpec
import matplotlib
from math import floor

if __name__ == "__main__":
    import proplot as pplt

    # ELECTRIC_COMPONENTS = [mp.Ex, mp.Ey, mp.Ez, mp.Dx, mp.Dy, mp.Dz]
    # MAGNETIC_COMPONENTS = [mp.Bx, mp.By, mp.Bz, mp.Hx, mp.Hy, mp.Hz]

    plt.rcParams["font.family"] = "helvetica"


def write_gif(filename, filenames):
    with imageio.get_writer(filename, mode='I', fps=30) as writer:
        for filename in filenames:
            image = imageio.v3.imread(filename)
            writer.append_data(image)

    for filename in set(filenames):
        os.remove(filename)


def flatten(list_of_lists):
    """https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists"""
    return [item for sublist in list_of_lists for item in sublist]


def expected_energy_location(r, z, ue):
    """
    Computes the expected energy location. Note that ue must sum to 1.
    :param r:
    :param z:
    :param ue: Energy density (must sum to 1)
    :return: An array of length 3 containing the three expected energy locations.
    """
    r, z = r[np.newaxis, :], z[:, np.newaxis]
    return [0., np.trapz(np.trapz(z * ue * r * 2 * np.pi))]


def energy_location_variance(r, z, ue, expected_energy_location_):
    r, z = r[np.newaxis, :], z[:, np.newaxis]
    return [np.trapz(np.trapz(((xi - expected_energy_location_[i]) ** 2 * ue * 2 * np.pi * r)))
            for i, xi in enumerate([r, z, ])]


def compute_entropy(*args):
    return [(np.sum(ei**2)**2) / np.sum(ei**4) for ei in args]


def min_of_quadratic(poly):
    """poly[0] x^2 + poly[1] x + poly[2]"""
    a, b, c = tuple(poly)
    # det = b ** 2 - 4 * a * c
    if a <= 0:
        return np.nan, np.nan
    t_min = - b / 2 / a
    val_min = np.polyval(poly, t_min)
    return t_min, val_min


# def test_function(x, x0, length):
#     return ((x - x0) / length > -.5) * ((x - x0) / length < .5)
#     x = (x - x0) / length * 2
#     y = np.exp(-1 / (1 - x**2))
#     y[x <= -1.] = 0.
#     y[x >= 1.] = 0.
#     y[np.isnan(y)] = 0.
#     return y


class SourceExcitation(ABC):

    @abstractmethod
    def shape(self, t):
        pass

    @property
    @abstractmethod
    def frequency(self):
        pass

    @property
    @abstractmethod
    def standard_time(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def get_stats(self, use_derivative=True) -> tuple[float, float]:
        t = self.standard_time
        dt = t[1] - t[0]
        y = self.shape(t)
        if use_derivative:
            dy = np.gradient(y) / dt
            pdf = dy ** 2
        else:
            pdf = y ** 2
        pdf = pdf / np.sum(pdf) / dt
        mean = np.sum(t * pdf) * dt
        var = np.sum((t - mean)**2 * pdf) * dt
        return mean, var


class DerivedGaussianExcitation(SourceExcitation):

    def __init__(self, f):
        self._f = f
        self.gamma = 1 / f / np.pi
        self.t0 = 3 * self.gamma

    def shape(self, t_):
        return np.exp(-((t_ - self.t0) / self.gamma) ** 2) * (4 * ((t_ - self.t0) / self.gamma) ** 2 - 2)

    def __str__(self):
        return "DerivedGaussianExcitation"

    @property
    def frequency(self):
        return self._f

    @property
    def standard_time(self):
        return np.linspace(-1, 1, 500) * 3 * self.gamma + self.t0


class AsymmetricGaussianExcitation(DerivedGaussianExcitation):

    def __init__(self, f):
        self._f = f
        self.gamma = 1 / f / np.pi
        self.t0 = 3 * self.gamma
        super().__init__(f)

    def shape(self, t_):
        return super().shape(t_) + .5 * super().shape(t_ - self.t0)

    def __str__(self):
        return "AsymmetricGaussianExcitation"

    @property
    def frequency(self):
        return self._f

    @property
    def standard_time(self):
        return np.linspace(-4, 4, 1000) * 3 * self.gamma + self.t0


class WindowedSineExcitation(SourceExcitation):

    def __init__(self, f, n_periods):
        self._frequency = f
        self.n_periods = n_periods

    def __str__(self):
        return "WindowedSineExcitation"

    def shape(self, t_):
        # sigma = self.n_periods / self.f / 3
        t0 = self.n_periods / self._frequency
        return np.sin(2 * np.pi * self._frequency * t_) * (t_ > 0) * (t_ < t0)

    @property
    def frequency(self):
        return self._frequency

    @property
    def standard_time(self):
        return np.linspace(0, self.n_periods / self._frequency, 500)


class AsymmetricExcitation(SourceExcitation):

    def __init__(self, f1, f2, delay, amplitude_ratio=2):
        self._frequency = max(f1, f2)
        self._f1, self._f2 = f1, f2
        self._delay = delay
        self._amplitude_ratio = amplitude_ratio

    def __str__(self):
        return "AsymmetricExcitation"

    def shape(self, t_):
        t2 = self._delay + 1 / self._f1
        return (t_ > 0) * (t_ < 1 / self._f1) * self._amplitude_ratio * np.sin(2 * np.pi * self._f1 * t_) + \
               (t_ > t2) * (t_ < t2 + 1 / self._f2) * np.sin(2 * np.pi * self._f2 * t_)

    @property
    def frequency(self):
        return self._frequency

    @property
    def standard_time(self):
        return np.linspace(0, 1 / self._f1 + self._delay + 1 / self._f2, 1000)


def find_r_travelled(simulation):
    e_field = np.sum(simulation.get_efield()**2, axis=-1)**.5
    x, y, z, _ = simulation.get_array_metadata()
    e_field_flat = np.zeros((e_field.size, ))
    r_flat = e_field_flat.copy()
    for linear_ind, (ind, e_field_i) in enumerate(np.ndenumerate(e_field)):
        e_field_flat[linear_ind] = e_field[ind]
        r_flat[linear_ind] = np.sqrt(x[ind[0]]**2 + y[ind[1]]**2 + z[ind[2]]**2)

    assert np.all(e_field_flat >= 0.)
    max_e = np.max(e_field_flat)
    r_flat[e_field_flat / max_e < 1e-2] = 0.
    r_max = np.max(r_flat)
    return r_max


def round_to(x):
    i = 2
    return floor(x * 10 ** i) / 10 ** i


def run_simulation(source_excitation, t_direct, t_time_reversal, delay_time_reversal_metric,
                   size_in_wavelengths, resolution=4, plot=False, max_energy=1e-3, verbosity=1,
                   save_gif=False):
    # t = source_excitation.standard_time
    # plt.plot(t, source_excitation.shape(t))
    # plt.show()
    t_direct = round_to(t_direct)
    t_time_reversal = round_to(t_time_reversal)
    size_in_wavelengths = round_to(size_in_wavelengths)
    signature = f"{str(source_excitation)}X{t_direct:02.2f}X{t_time_reversal:02.2f}X{delay_time_reversal_metric:02.2f}"\
                f"X{size_in_wavelengths:02.2f}X{resolution:02.2f}"
    signature = "".join(x for x in signature if x.isalnum())
    print(f"The signature of this run is {signature}")
    file_path = os.path.join("data", f"{signature}.pickle")
    try:
        with open(file_path, "rb") as fd:
            print("Found saved result, returning")
            return pickle.load(fd)
    except FileNotFoundError:
        pass

    # if plot:
    #     plt.figure()
    #     t = np.linspace(-1, 5, 300)
    #     plt.plot(t, source_excitation.shape(t))
    #     plt.show()
    print(f"Starting {source_excitation}")
    c = 1
    f = source_excitation.frequency
    wavelength = c / f
    dt = 1 / f / 10
    eps0_si = 8.854e-12
    mu0_si = 4 * np.pi * 1e-7
    c_si = 3e8

    mp.verbosity(verbosity)

    r_omega = size_in_wavelengths * wavelength
    resolution = resolution / wavelength

    cell_size = mp.Vector3(r_omega, 0, 2 * r_omega)

    geometry = [mp.Block(mp.Vector3(r_omega, 0, 2 * r_omega),
                         material=mp.Medium()),
                ]

    sources = [mp.Source(mp.CustomSource(source_excitation.shape),
                         mp.Ez,
                         center=mp.Vector3(),
                         size=mp.Vector3(0, 0, 0))]

    # pml_layers = [mp.PML(wavelength / 2)]
    # pec_layers = [Al]
    sim = mp.Simulation(cell_size=cell_size,
                        geometry=geometry,
                        # symmetries=[mp.Symmetry("Z"), ],
                        dimensions=mp.CYLINDRICAL,
                        sources=sources,
                        k_point=False,
                        resolution=resolution)
    if plot:
        # plt.figure(figsize=(12, 4))
        # plt.ion()
        # p1 = plt.subplot(1, 3, 1)
        # sim.plot2D(output_plane=mp.Volume(size=mp.Vector3(2.1 * r_omega, 2.1 * r_omega, 0)))
        # p2 = plt.subplot(1, 3, 2)
        # sim.plot2D(output_plane=mp.Volume(size=mp.Vector3(2.1 * r_omega, 0, 2.1 * r_omega)))
        # p3 = plt.subplot(1, 3, 3)
        # sim.plot2D(output_plane=mp.Volume(size=mp.Vector3(0, 2.1 * r_omega, 2.1 * r_omega)), )
        # ticks = np.arange(-size_in_wavelengths, size_in_wavelengths, .5)
        # plt.scatter(ticks, np.zeros((ticks.size, )), marker="+", color="red")

        fig = pplt.figure()
        ax = fig.subplot()
        pplt.show(block=False)
        ax.format(xlabel="x (m)", ylabel="y (m)")

    contours = (None,) * 3
    colorbar = None
    modifier = ""
    t_sim, filenames, mean_locs, var_locs, total_energies, entropies = [], [], [], [], [], []

    def update_plots(current_sim, update_data=False):
        nonlocal contours, modifier, filenames, mean_locs, var_locs, t_sim, entropies, colorbar

        e_field = current_sim.get_efield()
        e2 = np.sum(e_field ** 2, axis=-1) * (1 / eps0_si / c_si) ** 2
        b2 = np.sum(current_sim.get_bfield() ** 2, axis=-1) * (1 / eps0_si / c_si ** 2) ** 2
        energy_density = eps0_si / mu0_si * e2 + 1 / mu0_si ** 2 * b2
        r = np.linspace(0, r_omega, energy_density.shape[1])
        z = np.linspace(-r_omega, r_omega, energy_density.shape[0])
        # energy_density[:, r < .1] = 0.
        total_energy = np.trapz(np.trapz(2 * np.pi * r[np.newaxis, :] * energy_density))

        # index_mid = int(e2.shape[0] / 2)
        # slices = ((slice(None), slice(None), index_mid),
        #           (slice(None), index_mid, slice(None)),
        #           (index_mid, slice(None), slice(None)))
        plt_args = {"alpha": 1,
                    "levels": np.linspace(0, max_energy, 9),
                    "cmap": "Spectral_r"
                    }

        if plot:
            if contours[0] is not None:
                for contour_i in contours:
                    for coll in contour_i.collections:
                        coll.remove()
            # contours = (p1.contourf(x1, x2, energy_density[slices[0]].T, **plt_args),
            #             p2.contourf(x1, x3, energy_density[slices[1]].T, **plt_args),
            #             p3.contourf(x2, x3, energy_density[slices[2]].T, **plt_args))
            # p2.set_title(f"t={current_sim.meep_time():.3f}{modifier}")
            contours = (ax.contourf(
                r, z, energy_density, **plt_args
            ), )
            if colorbar is None:
                colorbar = ax.colorbar(contours[0], loc='r', label=f'Energy density')
            plt.pause(.000001)
            if save_gif:
                filename = f"figs/tr/t{modifier}-{current_sim.meep_time():.5f}.png"
                fig.save(filename)
                filenames.append(filename)

        if update_data and current_sim.meep_time() > t_direct + delay_time_reversal_metric:
            mean_loc = expected_energy_location(r, z, energy_density/total_energy)
            var_loc = energy_location_variance(r, z, energy_density/total_energy, mean_loc)
            # entropies.append(compute_entropy(*tuple(
            #     [e_field[:, :, :, i] for i in range(3)]
            # )))
            entropies.append(compute_entropy(energy_density))
            # if plot:
            #     p1.set_title(f"{mean_loc[0]:.1f} {mean_loc[1]:.1f} {mean_loc[2]:.1f}")
            #     p3.set_title(f"{var_loc[0]:.1f} {var_loc[1]:.1f} {var_loc[2]:.1f}")

            t_sim.append(current_sim.meep_time())
            mean_locs.append(mean_loc)
            var_locs.append(var_loc)
            total_energies.append(total_energy)

    if plot:
        sim.run(mp.at_every(dt, lambda arg: update_plots(arg, update_data=False)), until=t_direct)
        if save_gif:
            write_gif(f"figs/direct-source.gif", filenames)
            filenames = []
    else:
        sim.run(until=t_direct)
    print(f"Starting TR {source_excitation}")
    sim.fields.scale_magnetic_fields(-1.)
    # for component in [mp.Hr, mp.Hp, mp.Hz, mp.Br, mp.Bp, mp.Bz]:
    #     sim.initialize_field(
    #         component,
    #         lambda pt: -sim.get_field_point(component, pt)
    #     )
    modifier = "tr"

    sim.run(mp.at_every(dt, lambda arg: update_plots(arg, update_data=True)), until=t_time_reversal)

    if plot and save_gif:
        write_gif(f"figs/tr-source.gif", filenames)

    total_energies, mean_locs, var_locs = np.array(total_energies), np.array(mean_locs), np.array(var_locs)
    entropies = np.array(entropies)
    min_t, max_t = np.min(t_sim), np.max(t_sim)
    duration = max_t - min_t
    t_sim = np.array(t_sim)
    keep_indices = (t_sim - min_t < duration/4) + (t_sim - min_t > duration*3/4)
    polynomials = np.polyfit(t_sim[keep_indices], var_locs[keep_indices, :], 2)

    minimums = []
    for coeff in np.transpose(polynomials):
        t_min, val_min = min_of_quadratic(coeff)
        minimums.append((2 * t_direct - t_min, val_min))

    if plot:
        fig, ax2 = pplt.subplots()

        # for data, color, axis_name in zip(mean_locs.T, colors, ("x", "y", "z")):
        #     ax1.plot(t_sim, data, color=color, linestyle="dashed",
        #              label=f"$\sigma_{axis_name}^t$")
        # ax1.legend()
        # ax1.format(ylabel="Expected position", ylim=(-1, 1))

        # ax2 = ax1.twinx()
        axis_names = ("PLCLDR", "PLCLDR", "PLCLDR", )
        line_styles_data = ("-", "--", ":", )
        fit_markers = ("o", "s", "d", )
        optimum_markers = ("x", "+", "2", )
        for index, (data, coeff, line_style_data, fit_marker, optimum_marker, axis_name) in enumerate(
                zip(
                    np.transpose(var_locs), np.transpose(polynomials),
                    line_styles_data, fit_markers, optimum_markers, axis_names)
                ):
            color = ("Spectral", index/3)
            n_down_sample = 1
            ax2.plot(t_sim[::n_down_sample], data[::n_down_sample], line_style_data, marker=fit_marker,
                     facecolor="none", edgecolor=color, linewidth=0,
                     label=f"$(\sigma_{axis_name}^t)^2$", color=color)
            ax2.plot(t_sim, np.polyval(coeff, t_sim), line_style_data, label=f"($\sigma_{axis_name}^t)^2$, fit",
                     color=color)
            # ax2.plot(t_sim, entropy / np.max(entropy) * np.max(data))
            t_min, val_min = min_of_quadratic(coeff)
            ax2.plot((t_min, ), (val_min, ), fit_marker, label=f"$(\sigma_{axis_name}^\star)^2$", color=color)
            ax2.legend(ncol=3)

        # ax2.plot(t_sim, total_energies, "b*-")

        # plt.legend((
        #     "Energy location variance, $x$",
        #     "..., 2nd order fit",
        #     "..., minimum",
        #     "Energy location variance, $y$",
        #     "..., 2nd order fit",
        #     "..., minimum",
        #     "Energy location variance, $z$",
        #     "..., 2nd order fit",
        #     "..., minimum",
        #     "Total energy"
        # ), loc="upper right"
        # )

        ax2.format(xlabel="Normalized time", ylabel="(wavelength$^2$)")
        fig.set_size_inches(5, 3)
        fig.save("figs/lsq_fit.pdf")
        pplt.show(block=True)
    print(f"Done {type(source_excitation)}")
    with open(file_path, "wb") as fd:
        pickle.dump(
            minimums, fd
        )
    return minimums


SRC_NAMES = [
    # "Windowed sine (2 periods)",
    # "Windowed sine (4 periods)",
    # "Windowed sine (1 period, 1 repeat)",
    "Asymmetric Gaussian",
]
SRC_NAMES_2_LINES = [
    # "Windowed sine\n(2 periods)",
    # "Windowed sine\n(4 periods)",
    # "Windowed sine\n(1 period, 1 repeat)",
    "Asymmetric\nGaussian",
]
N_SRC_TYPE = len(SRC_NAMES)


def run_parallel():
    fs = np.arange(.5, 2, .1)
    res = 2001
    # func_args = [(WindowedSineExcitation(fi, 2), 2.5 / fi, 3 / fi, 0 / fi, 4, res) for fi in fs] + \
    #        [(WindowedSineExcitation(fi, 4), 4.5 / fi, 5 / fi, 0 / fi, 5.5, res) for fi in fs] + \
    #        [(AsymmetricExcitation(fi / 2, fi, 1 / fi), 6 / fi, 6 / fi, 0., 6., res) for fi in fs] + \
    #        [(DerivedGaussianExcitation(fi), 2 / fi, 2 / fi, 0., 2, res) for fi in fs]
    func_args = [(AsymmetricGaussianExcitation(fi), 3 / fi, 3 / fi, 0 / fi, 3, res) for fi in fs]
    with multiprocessing.Pool(3) as pool:
        result = pool.starmap(run_simulation, func_args)

    means, variances = np.zeros((len(result), 2)), np.zeros((len(result), 2))
    for ind_exp, dim in itertools.product(range(len(result)), range(2)):
        means[ind_exp, dim], variances[ind_exp, dim] = tuple(result[ind_exp][dim])

    with open("stats.pickle", "wb") as fd:
        pickle.dump((fs, func_args, means, variances), fd)


def split_list(arr, n: int):
    if len(arr) % n != 0:
        raise ValueError("The length of the list is not a multiple of n")
    len_chunk = len(arr) // n
    if isinstance(arr, np.ndarray):
        return [arr[shift:shift + len_chunk, :] for shift in range(0, len(arr), len_chunk)]
    elif isinstance(arr, list):
        return [arr[shift:shift + len_chunk] for shift in range(0, len(arr), len_chunk)]
    else:
        raise ValueError(f"Unknown type of arr: {type(arr)}")


def post_processing():
    with open("stats.pickle", "rb") as fd:
        data = pickle.load(fd)
        fs, args, all_means, all_variances = data
    fs = np.array(fs)
    all_excitations = [arg[0] for arg in args]
    all_means, all_variances, all_excitations = tuple([split_list(arr, N_SRC_TYPE) for arr in (
        all_means, all_variances, all_excitations
    )])

    fig = plt.figure(figsize=(10, 3), constrained_layout=True)
    gs = GridSpec(1, 8, figure=fig)

    colors = ("k", )  # plt.get_cmap('accent')(np.linspace(0, 1, N_SRC_TYPE))"
    markers = [
        "o",
        "*",
        "v",
        "+",
        "."
    ]
    lines = [
        "-",
        "--",
        "-.",
        ":"
    ]

    spec_data = [{"mfc": color, "mec": color, "marker": marker, "linestyle": ""} for color, marker in zip(
        colors, markers
    )]
    spec_fit = [{"color": color, "marker": "", "linestyle": line} for color, line in zip(colors, lines)]

    ax1 = fig.add_subplot(gs[0, :3])
    ax3 = fig.add_subplot(gs[0, 3:6])
    # ax2 = fig.add_subplot(gs[0, 6:7])
    ax4 = fig.add_subplot(gs[0, 6:])

    for exp_index, (means, variances, excitations) in enumerate(zip(all_means, all_variances, all_excitations)):
        means, variances = np.median(means, axis=1), np.sum(variances, axis=1)
        std_devs = np.sqrt(variances)
        theoretical_stats = np.array([excitation.get_stats(use_derivative=True) for excitation in excitations])

        # res = scipy.optimize.curve_fit(fit_means, fs, means)
        # a_opt = res[0]
        # fitted_means = fit_means(fs, a_opt)
        # error_mean = np.sum((fitted_means - means) ** 2) / np.sum(means ** 2)
        #
        # res = scipy.optimize.curve_fit(fit_vars, fs, variances)
        # c_opt = res[0]
        # fitted_variances = fit_vars(fs, c_opt)
        # error_variance = np.sum((fitted_variances - variances) ** 2) / np.sum(variances ** 2)

        ax1.plot(fs, means, **spec_data[exp_index])
        ax1.plot(fs, theoretical_stats[:, 0], **spec_fit[exp_index])
        # plt.fill_between(fs, means - std_devs / 2, means + std_devs / 2, color=spec_fit[exp_index]["color"]+"10")

        predicted_std_devs = np.sqrt(theoretical_stats[:, 1])
        # ax2.plot(fs, std_devs / predicted_std_devs, **spec_data[exp_index])
        ax3.plot(fs, predicted_std_devs, **spec_fit[exp_index])
        ax3.plot(fs, std_devs, **spec_data[exp_index])
    ax1.set_ylabel("Normalized time")
    ax1.set_title("Optimal focusing time $t^\star$\nand source power expected value $t_s$")

    # ax2.set_title("Ratio of standard deviations:\nsimulation $\sigma^\star$ over source $\sigma_s$ ")
    # ax2.set_ylim((0, 1))
    # ax2.set_ylabel("$\sigma^\star/\sigma_s$")

    ax3.set_title("Energy location standard deviation $\sigma^\star$\nand source power standard deviation $\sigma_s$")
    ax3.set_ylabel("(wavelength)")

    for ax in [ax1, ax3]:
        ax.set_xlabel("Normalized main frequency component")

    ax4.axis("off")
    x_meas = .7
    x_pred = .9
    len_pred = .1
    y_meas = np.flip(np.linspace(.1, .8, N_SRC_TYPE))
    txt_shift = .05
    txt_centering = 0.02
    for y, marker, line, color, src_name in zip(y_meas, markers, lines, colors, SRC_NAMES_2_LINES):
        plt.scatter(x_meas, y, marker=marker, color=color)
        plt.plot([x_pred - len_pred / 2, x_pred + len_pred / 2], [y, y],
                 linestyle=line, color=color)
        plt.text(0, y - txt_shift, src_name, fontdict=dict())

    plt.text(x_meas - txt_centering, 1, "$t^\star$ or $\sigma^\star$", rotation=90)
    plt.text(x_pred - txt_centering, 1, "$t_s$ or $\sigma_s$", rotation=90)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("figs/stats_res.pdf")

    fig, axes = plt.subplots(N_SRC_TYPE, 1, figsize=(6, 4))
    if N_SRC_TYPE == 1:
        axes = [axes, ]

    try:
        exp_index = np.where(fs >= 1.)[0][0]
    except IndexError:
        raise RuntimeError("There might be no data for f=1")

    t_max = max(max(src_type[exp_index].standard_time) for src_type in all_excitations)
    t = np.linspace(-.1, 3, 1000)
    dt = t[1] - t[0]

    for plt_index, (ax, src_type) in enumerate(zip(axes, all_excitations)):
        e = src_type[exp_index]
        y = e.shape(t)
        y = y / np.max(np.abs(y))
        dy = np.gradient(y) / dt
        pdf = dy**2
        pdf = pdf / np.sum(pdf) / dt
        ax2 = ax.twinx()
        color_pdf = "seagreen"
        color_f = "red"
        l2 = ax.fill_between(t, pdf, color=color_pdf, alpha=.4, label="PDF $\propto ($d$p/$d$t)^2$")

        l1 = ax2.plot(t, y, color=color_f, label="$p$")

        mean, var = e.get_stats()
        std_dev = np.sqrt(var)
        n_std_devs = 3
        lower, upper = mean - std_dev * n_std_devs / 2, mean + std_dev * n_std_devs / 2
        y_dev = (max(y) + min(y)) / 2
        ax2.plot((lower, upper), (y_dev, y_dev),  'k|-', linewidth=2, markersize=10)
        ax2.plot((mean, ), (y_dev, ),  'k|-', linewidth=2, markersize=20)

        ax.tick_params(axis="y", colors=color_pdf)
        ax2.tick_params(axis="y", colors=color_f)
        ax.set_xlim(min(t), max(t))
        props = dict(boxstyle='square', facecolor='lightblue', alpha=0.8)
        ax2.text(.90 - len(SRC_NAMES[plt_index]) * .014, 0.9, SRC_NAMES[plt_index], transform=ax.transAxes,
                 verticalalignment='top', bbox=props)
        ax.grid(visible=True, axis="x", which="both")
        if plt_index == 0:
            ax.set_ylabel("Energy density", color=color_pdf)
            ax2.set_ylabel("Normalized dipole moment", color=color_f)
            # ax.yaxis.set_label_coords(-0.15, 1.02)
            # ax2.yaxis.set_label_coords(1.15, 1.02)

        match plt_index:
            case 0:  # n if n == N_SRC_TYPE - 1:
                ax.set_xlabel("Normalized time", loc="center")
                lines = [l2] + l1
                labels = [line.get_label() for line in lines]
                ax.legend(lines, labels,
                          bbox_to_anchor=(.8, -.3),
                          ncol=2)
            case 0:  # n if n < N_SRC_TYPE - 1:
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    labelbottom=False
                )
            case 0:
                ax.set_ylabel("Energy density", color=color_pdf)
                ax2.set_ylabel("Amplitude", color=color_f)
                for child in fig.gca().get_children():
                    if isinstance(child, matplotlib.spines.Spine):
                        child.set_color(color_pdf)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.02)
    plt.savefig("figs/excitations.pdf")

    plt.show()


def show_min_var_x_vs_resolution_for_gaussian_excitation():
    resolutions = np.array((10, 15, 20, 25, 30, 35, 40, 50, ))
    min_vars = np.array((0.0671571860915785,
                         0.0485454155237508,
                         0.04246576948854175,
                         0.03995087858150548,
                         0.038396992219154935,
                         0.03762829362390496,
                         0.036972621676337525,
                         0.036300663962869795
                         ))
    plt.figure()
    plt.plot(resolutions, min_vars)
    plt.show()


if __name__ == '__main__':
    # run_simulation(
    #     AsymmetricGaussianExcitation(1),
    #     3, 3, 0., 3, resolution=2002,
    #     plot=True, max_energy=1e-5
    # )
    # import sys
    # sys.exit(0)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_excitation", nargs=1, type=str
    )
    parser.add_argument(
        "t_direct", nargs=1, type=float
    )
    parser.add_argument(
        "t_time_reversal", nargs=1, type=float
    )
    parser.add_argument(
        "size_in_wavelengths", nargs=1, type=float
    )
    parser.add_argument(
        "resolution", nargs=1, type=int
    )
    parser.add_argument(
        "args", nargs="+", type=float
    )
    try:
        args = parser.parse_args()
    except SystemExit:
        run_parallel()
        post_processing()
        import sys
        sys.exit(0)
    name = args.source_excitation[0]
    cls = None
    if name == "WindowedSineExcitation":
        cls = WindowedSineExcitation
    elif name == "AsymmetricExcitation":
        cls = AsymmetricExcitation
    elif name == "DerivedGaussianExcitation":
        cls = DerivedGaussianExcitation
    else:
        raise ValueError(f"Unknown class: {name}")

    source = cls(*tuple(args.args))
    print(args)
    run_simulation(
        source_excitation=source,
        t_direct=args.t_direct[0],
        t_time_reversal=args.t_time_reversal[0],
        size_in_wavelengths=args.size_in_wavelengths[0],
        resolution=args.resolution[0],
        delay_time_reversal_metric=0.
    )
