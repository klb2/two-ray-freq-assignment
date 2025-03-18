import logging

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

from model import length_los, length_ref, get_distance_from_delta_len
from multiple_frequencies import sum_power_lower_envelope, sum_power
from roots_fourier_series import (
    calc_fourier_coefficients,
    construct_frobenius_matrix_fourier_series,
    roots_from_frobenius_matrix,
)
from util import to_decibel, export_results

LOGGER = logging.getLogger(__name__)


def main_crit_distances(
    d_min: float,
    d_max: float,
    freq: float,
    delta_freq: float,
    num_freq: int,
    h_tx: float,
    h_rx: float,
    c: float = constants.c,
    power_tx: float = 1,
    plot: bool = False,
    export: bool = False,
):
    freq = freq + np.arange(num_freq) * delta_freq
    coeff_a, coeff_b = calc_fourier_coefficients(freq)
    coeff_b_deriv = -coeff_a[1:] * (np.arange(num_freq - 1) + 1)
    frob_matrix = construct_frobenius_matrix_fourier_series(
        a=np.zeros(num_freq), b=coeff_b_deriv
    )
    roots_t = roots_from_frobenius_matrix(frob_matrix)
    bounds_t = (
        2
        * np.pi
        * delta_freq
        * (
            length_ref([d_max, d_min], h_tx, h_rx)
            - length_los([d_max, d_min], h_tx, h_rx)
        )
        / c
    )
    _m = (bounds_t + np.pi) / (2 * np.pi)
    roots_t = np.concatenate(
        [roots_t + 2 * np.pi * k for k in np.arange(int(_m[0]), int(_m[1]) + 1)]
    )
    roots_t = roots_t[roots_t > 0]
    delta_len = c * roots_t / (2 * np.pi * delta_freq)
    distance_roots = get_distance_from_delta_len(delta_len, h_tx, h_rx)
    distance_roots = distance_roots[~np.isnan(distance_roots)]
    distance_roots = distance_roots[
        np.logical_and(d_min <= distance_roots, distance_roots <= d_max)
    ]
    _crit_dist = np.concatenate((distance_roots, [d_min, d_max]))
    _sum_power_crit_dist = sum_power_lower_envelope(_crit_dist, freq, h_tx, h_rx)
    minimum_sum_power = np.min(_sum_power_crit_dist)
    _actual_sum_power_crit_dist = sum_power(_crit_dist, freq, h_tx, h_rx)
    _min_actual_power = np.min(_actual_sum_power_crit_dist)

    LOGGER.info(
        f"Crit distances with following parameters: dmin={d_min:.1f}, dmax={d_max:.1f}, h_tx={h_tx:.1f}, h_rx={h_rx:.1f}, f0={freq[0]:E}, df={delta_freq:E}, N={num_freq:d}"
    )
    LOGGER.info(f"Crit distances: {distance_roots}")
    LOGGER.info(
        f"Power at boundaries: P(dmin) = {to_decibel(_sum_power_crit_dist[-2]):.1f},\tP(dmax) = {to_decibel(_sum_power_crit_dist[-1]):.1f}"
    )
    LOGGER.info(f"Min receive power: {to_decibel(minimum_sum_power):.1f}")
    LOGGER.info(f"Actual power at minimum: {to_decibel(_min_actual_power):.1f}")

    distance = np.logspace(np.log10(d_min) - 0.3, np.log10(d_max) + 0.3, 2000)

    power_rx = sum_power(distance, freq, h_tx, h_rx)
    power_rx_db = to_decibel(power_rx)
    power_sum_lower = sum_power_lower_envelope(distance, freq, h_tx, h_rx)
    power_sum_lower_db = to_decibel(power_sum_lower)
    results = {
        "distance": distance,
        "powerSum": power_rx_db,
        "envelope": power_sum_lower_db,
    }

    if plot:
        fig, axs = plt.subplots()
        axs.semilogx(distance, power_rx_db, label="Receive Power")
        axs.semilogx(distance, power_sum_lower_db, label="Lower Bound")
        axs.vlines(distance_roots, -140, -60, ls="--", color="r")
        axs.vlines([d_min, d_max], -140, -60, ls="-", color="k", lw=2)
        axs.legend()
        axs.set_xlim([min(distance), max(distance)])
        axs.set_ylim([1.05 * min(power_sum_lower_db), 0.95 * max(power_rx_db)])
    if export:
        LOGGER.debug("Exporting results...")
        _fname = "power_sum-{freqs}-t{h_tx:.1f}-r{h_rx:.1f}.dat".format(
            h_rx=h_rx, h_tx=h_tx, freqs="-".join([f"{_f:E}" for _f in freq])
        )
        export_results(results, _fname)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--h_tx", type=float, default=10.0)
    parser.add_argument("-r", "--h_rx", type=float, default=1.5)
    parser.add_argument("-f", "--freq", type=float, default=2e9)
    parser.add_argument("-df", "--delta_freq", type=float, default=500e6)
    parser.add_argument("-n", "--num_freq", type=int, default=3)
    parser.add_argument("-dmin", "--d_min", type=float, default=10.0)
    parser.add_argument("-dmax", "--d_max", type=float, default=100.0)
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
    main_crit_distances(**args)
    plt.show()
