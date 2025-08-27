import logging
import itertools

import numpy as np
from scipy import constants
from scipy import optimize
import matplotlib.pyplot as plt

from util import export_results, to_decibel

from model import length_los, length_ref, get_distance_from_delta_len
from single_frequency import rec_power, crit_dist
from roots_fourier_series import (
    calc_fourier_coefficients,
    construct_frobenius_matrix_fourier_series,
    roots_from_frobenius_matrix,
)

LOGGER = logging.getLogger(__name__)


def sum_power(distance, freq, h_tx, h_rx, c=constants.c, power_tx=1.0):
    power_rx = np.zeros_like(distance)
    for f in freq:
        power_rx += rec_power(distance, f, h_tx, h_rx, power_tx=power_tx)
    power_rx = power_rx / len(freq)
    return power_rx


def sc_power(distance, freq, h_tx, h_rx, c=constants.c, power_tx=1.0):
    _power_sc = rec_power(distance, np.expand_dims(freq, -1), h_tx, h_rx)
    power_sc = np.max(_power_sc, axis=0)
    return power_sc


def sum_power_lower_envelope(
    distance,
    freq,
    h_tx,
    h_rx,
    G_los=1,
    G_ref=1,
    c=constants.c,
    power_tx=1,
    return_parts=False,
):
    freq = np.array(freq)
    d_los = length_los(distance, h_tx, h_rx)
    d_ref = length_ref(distance, h_tx, h_rx)
    omega = 2 * np.pi * freq
    _factor = power_tx / len(freq) * (c / 2) ** 2
    _part1 = (1 / d_los**2 + 1 / d_ref**2) * np.sum(1 / omega**2)
    _part22 = np.zeros_like(d_los)
    for wj, wk in itertools.combinations(omega, 2):
        _part22 += 1 / (wj**2 * wk**2) * np.cos((wj - wk) * (d_ref - d_los) / c)
    _part2 = np.sqrt(np.sum(1 / omega**4) + 2 * _part22)
    _part2 = -2 / (d_los * d_ref) * _part2
    if return_parts:
        return _factor * _part1, _factor * _part2
    else:
        power_rx = _factor * (_part1 + _part2)
        return power_rx


def _max_cosine_interval(d_min, d_max, freq, h_tx, h_rx, c=constants.c):
    freq = np.abs(freq)
    _crit_dist = crit_dist(freq, h_tx, h_rx)
    idx_dk_range = np.where(np.logical_and(_crit_dist >= d_min, _crit_dist <= d_max))
    if len(idx_dk_range[0]) == 0:
        _cos_dk = -1.0
    else:
        dk_worst = np.max(_crit_dist[idx_dk_range])
        _d_los_dk = length_los(dk_worst, h_tx, h_rx)
        _d_ref_dk = length_ref(dk_worst, h_tx, h_rx)
        _cos_dk = np.cos(2 * np.pi * freq / c * (_d_ref_dk - _d_los_dk))
    _cos_dmin = np.cos(
        2
        * np.pi
        * freq
        / c
        * (length_ref(d_min, h_tx, h_rx) - length_los(d_min, h_tx, h_rx))
    )
    _cos_dmax = np.cos(
        2
        * np.pi
        * freq
        / c
        * (length_ref(d_max, h_tx, h_rx) - length_los(d_max, h_tx, h_rx))
    )
    _cos_max = np.max((_cos_dk, _cos_dmin, _cos_dmax))
    return _cos_max


def lower_bound_min_sum_power_equal_df(
    d_min: float,
    d_max: float,
    freq: float,
    delta_freq: float,
    num_freq: int,
    h_tx: float,
    h_rx: float,
    c: float = constants.c,
    power_tx: float = 1,
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
    return minimum_sum_power


def lower_bound_min_sum_power(
    distance, d_min: float, d_max: float, freq, h_tx, h_rx, c=constants.c, power_tx=1
):
    freq = np.array(freq)
    num_freqs = len(freq)
    d_los = length_los(distance, h_tx, h_rx)
    d_ref = length_ref(distance, h_tx, h_rx)

    omega = 2 * np.pi * freq
    _factor = power_tx / len(freq) * (c / 2) ** 2

    _freq_t = np.tile(freq, (num_freqs, 1))
    _freq_diff = _freq_t - _freq_t.T
    _freq_diff = _freq_diff[_freq_diff > 0]
    _base_freq = np.abs(np.min(_freq_diff))
    _part22 = 0.0
    for fj, fk in itertools.combinations(freq, 2):
        _df = abs(fj - fk)
        if np.isclose(_df, _base_freq):
            _cos_part = np.cos(2 * np.pi * _df * (d_ref - d_los) / c)
        else:
            _cos_part = _max_cosine_interval(d_min, d_max, _df, h_tx, h_rx)
        _part22 += 1 / ((2 * np.pi * fj) ** 2 * (2 * np.pi * fk) ** 2) * _cos_part
    _part22 = np.sqrt(np.sum(1.0 / omega**4) + 2 * _part22)
    _part1 = (1 / d_los**2 + 1 / d_ref**2) * np.sum(1 / omega**2)
    _part21 = -2 / (d_los * d_ref)
    _part2 = _part21 * _part22
    bound_min_power = _factor * (_part1 + _part2)
    return bound_min_power


def main_power_multiple_freq(
    d_min, d_max, freq, h_tx, h_rx, plot=False, export=False, **kwargs
):
    distance = np.logspace(np.log10(d_min), np.log10(d_max), 2000)

    freq = np.array(freq)
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
        axs.legend()
    if export:
        LOGGER.debug("Exporting single frequency power results.")
        _fname = "power_sum-{freqs}-t{h_tx:.1f}-r{h_rx:.1f}.dat".format(
            h_rx=h_rx, h_tx=h_tx, freqs="-".join([f"{_f:E}" for _f in freq])
        )
        export_results(results, _fname)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--h_tx", type=float, default=10.0)
    parser.add_argument("-r", "--h_rx", type=float, default=1.0)
    parser.add_argument("-f", "--freq", type=float, nargs="+", default=[2.4e9])
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
    main_power_multiple_freq(**args)
    plt.show()
