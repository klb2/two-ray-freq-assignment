import logging

import numpy as np
from scipy import constants
from scipy import optimize
import matplotlib.pyplot as plt

from util import export_results, to_decibel

from model import length_los, length_ref


LOGGER = logging.getLogger(__name__)


def delta_phi(distance, freq, h_tx, h_rx, c=constants.speed_of_light):
    omega = 2 * np.pi * freq
    _d_phi = (
        omega
        / c
        * (length_ref(distance, h_tx, h_rx) - length_los(distance, h_tx, h_rx))
    )
    return _d_phi


def rec_power(distance, freq, h_tx, h_rx, G_los=1, G_ref=1, c=constants.c, power_tx=1):
    d_los = length_los(distance, h_tx, h_rx)
    d_ref = length_ref(distance, h_tx, h_rx)
    omega = 2 * np.pi * freq
    phi = omega / c * (d_ref - d_los)
    _factor = power_tx * (c / (2 * omega)) ** 2
    _part1 = G_los / (d_los**2)
    _part2 = G_ref / (d_ref**2)
    _part3 = -2 * np.sqrt(G_los * G_ref) / (d_los * d_ref) * np.cos(phi)
    power_rx = _factor * (_part1 + _part2 + _part3)
    return power_rx


def rec_power_lower_envelope(
    distance, freq, h_tx, h_rx, G_los=1, G_ref=1, c=constants.c, power_tx=1
):
    d_los = length_los(distance, h_tx, h_rx)
    d_ref = length_ref(distance, h_tx, h_rx)
    omega = 2 * np.pi * freq
    _factor = power_tx * (c / (2 * omega)) ** 2
    _part1 = G_los / (d_los**2)
    _part2 = G_ref / (d_ref**2)
    _part3 = -2 * np.sqrt(G_los * G_ref) / (d_los * d_ref)
    power_rx = _factor * (_part1 + _part2 + _part3)
    return power_rx


def crit_dist(freq, h_tx, h_rx, c=constants.c, k=None):
    a = h_tx - h_rx
    b = h_tx + h_rx
    max_phi = 2 * np.pi * freq / c * (b - a)
    max_k = np.divmod(max_phi, 2 * np.pi)[0]
    if k is not None:
        if k > max_k:
            raise ValueError(
                f"Your provided k is too large. The maximum k is {max_k:d}"
            )
        k = k + 0j
    else:
        k = np.arange(max_k) + 1 + 0j
    _d = (
        -1
        / (2 * c * freq * k)
        * np.sqrt(c**2 * k**2 - 4 * freq**2 * h_rx**2)
        * np.sqrt(c**2 * k**2 - 4 * freq**2 * h_tx**2)
    )
    _d = np.real(_d)
    return _d


def min_rec_power_single_freq(
    d_min: float, d_max: float, freq, h_tx, h_rx, c=constants.c
):
    _crit_dist = crit_dist(freq, h_tx, h_rx)
    idx_dk_range = np.where(np.logical_and(_crit_dist >= d_min, _crit_dist <= d_max))
    if len(idx_dk_range[0]) == 0:
        _pow_dk = np.infty
    else:
        dk_worst = np.max(_crit_dist[idx_dk_range])
        _pow_dk = rec_power(dk_worst, freq, h_tx, h_rx)
    _pow_dmin = rec_power(d_min, freq, h_tx, h_rx)
    _pow_dmax = rec_power(d_max, freq, h_tx, h_rx)
    return np.min([_pow_dmin, _pow_dk, _pow_dmax])


def main_phi(freq, h_tx, h_rx, plot=False, export=False, rho=1.0):
    distance = np.logspace(0, 3, 1000)
    d_phi = delta_phi(distance, freq, h_tx, h_rx)
    results = {"distance": distance, "dPhi": d_phi}
    if plot:
        fig, axs = plt.subplots()
        axs.semilogx(distance, d_phi)
    if export:
        LOGGER.debug("Exporting delta phi results.")
        export_results(results, f"delta_phi-{freq:E}-t{h_tx:.1f}-r{h_rx:.1f}.dat")
    return results


def main_power_single_freq(freq, h_tx, h_rx, rho=1.0, plot=False, export=False):
    if not 0 < rho <= 1:
        raise ValueError("Rho needs to be between 0 and 1")
    G_ref = rho**2

    if freq > 1e9:
        _num_steps = 5000
    else:
        _num_steps = 2000
    distance = np.logspace(0, 3, _num_steps)
    power_rx = rec_power(distance, freq, h_tx, h_rx, G_ref=G_ref)
    power_rx_db = to_decibel(power_rx)
    results = {"distance": distance, "power": power_rx_db}
    crit_distances = crit_dist(freq, h_tx, h_rx)
    LOGGER.info("Critical distances: %s", crit_distances)

    dmin = 30
    dmax = 100
    LOGGER.info(f"Normalized gain: rho={rho:.2f}")
    LOGGER.info(f"Numerical Example with dmin={dmin:.1f} and dmax={dmax:.1f}")
    _crit_dist_interval = crit_distances[
        np.where(np.logical_and(crit_distances < dmax, crit_distances > dmin))
    ]
    LOGGER.info(f"Critical distances in [dmin, dmax]: {_crit_dist_interval}")
    _pr_min = to_decibel(rec_power(dmin, freq, h_tx, h_rx, G_ref=G_ref))
    _pr_max = to_decibel(rec_power(dmax, freq, h_tx, h_rx, G_ref=G_ref))
    _pr_d1 = to_decibel(
        rec_power(np.max(_crit_dist_interval), freq, h_tx, h_rx, G_ref=G_ref)
    )
    LOGGER.info(f"Power at dmin = {_pr_min:.1f}")
    LOGGER.info(f"Power at dmax = {_pr_max:.1f}")
    LOGGER.info(f"Power at max d_k = {_pr_d1:.1f}")

    if plot:
        fig, axs = plt.subplots()
        axs.semilogx(distance, power_rx_db)
        axs.vlines(
            crit_distances,
            min(power_rx_db),
            max(power_rx_db),
            colors="k",
            linestyles="dashed",
        )
    if export:
        LOGGER.debug("Exporting single frequency power results.")
        export_results(
            results, f"power_single-{freq:E}-t{h_tx:.1f}-r{h_rx:.1f}-rho{rho:.2f}.dat"
        )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--h_tx", type=float, default=10.0)
    parser.add_argument("-r", "--h_rx", type=float, default=1.0)
    parser.add_argument("-f", "--freq", type=float, default=2.4e9)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument(
        "-v", "--verbosity", action="count", default=0, help="Increase output verbosity"
    )
    args = vars(parser.parse_args())
    verb = args.pop("verbosity")
    logging.basicConfig(
        format="%(asctime)s - %(module)s -- [%(levelname)8s]: %(message)s",
        handlers=[
            logging.FileHandler("main.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    loglevel = logging.WARNING - verb * 10
    LOGGER.setLevel(loglevel)
    main_phi(**args)
    main_power_single_freq(**args)
    plt.show()
