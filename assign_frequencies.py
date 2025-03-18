import logging

import numpy as np
from scipy import constants

from model import length_los, length_ref, get_distance_from_delta_len
from util import to_decibel
from single_frequency import rec_power, crit_dist, min_rec_power_single_freq
from multiple_frequencies import sum_power_lower_envelope
from roots_fourier_series import (
    calc_fourier_coefficients,
    construct_frobenius_matrix_fourier_series,
    roots_from_frobenius_matrix,
)

LOGGER = logging.getLogger(__name__)


def calc_worst_case_power_from_assignment(
    assignment,
    d_min: float,
    d_max: float,
    freq: float,
    delta_freq: float,
    num_freq: int,
    max_freq: int,
    h_tx: float,
    h_rx: float,
    c: float = constants.c,
):
    freq = freq + np.arange(num_freq) * delta_freq
    idx_assigned_freq = np.where(assignment)[0]
    freq = freq[idx_assigned_freq]
    num_freq = len(freq)
    if num_freq == 0:
        return 0
    elif num_freq == 1:
        _min_power = min_rec_power_single_freq(d_min, d_max, freq[0], h_tx, h_rx)
        # _min_power_db = to_decibel(_min_power)
        return _min_power

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
    # min_power_db = to_decibel(minimum_sum_power)
    return minimum_sum_power


def greedy_frequency_assignment(
    d_min: float,
    d_max: float,
    freq: float,
    delta_freq: float,
    num_freq: int,
    max_freq: int,
    h_tx: float,
    h_rx: float,
    c: float = constants.c,
):
    assignment = np.zeros(num_freq)
    while np.count_nonzero(assignment) < max_freq:
        idx_unassigned_freq = np.where(assignment == 0)[0]
        _new_assigns = np.eye(num_freq)[idx_unassigned_freq]
        _new_freqs = assignment + _new_assigns
        _new_worst_power = [
            calc_worst_case_power_from_assignment(
                _assign, d_min, d_max, freq, delta_freq, num_freq, max_freq, h_tx, h_rx
            )
            for _assign in _new_freqs
        ]
        best_freq = np.argmax(_new_worst_power)
        idx_best_freq = idx_unassigned_freq[best_freq]
        assignment[idx_best_freq] = 1
    return assignment


def uniform_frequency_assignment(
    d_min: float,
    d_max: float,
    freq: float,
    delta_freq: float,
    num_freq: int,
    max_freq: int,
    h_tx: float,
    h_rx: float,
    c: float = constants.c,
):
    unif = np.zeros(num_freq)
    unif[np.arange(0, num_freq, int(np.ceil(num_freq / max_freq)))] = 1
    return unif


def random_frequency_assignment(
    d_min: float,
    d_max: float,
    freq: float,
    delta_freq: float,
    num_freq: int,
    max_freq: int,
    h_tx: float,
    h_rx: float,
    c: float = constants.c,
):
    idx_random_freq = np.random.choice(
        np.arange(num_freq), size=max_freq, replace=False
    )
    random_freq = np.zeros(num_freq)
    random_freq[idx_random_freq] = 1
    return random_freq


def consecutive_frequency_assignment(
    d_min: float,
    d_max: float,
    freq: float,
    delta_freq: float,
    num_freq: int,
    max_freq: int,
    h_tx: float,
    h_rx: float,
    c: float = constants.c,
):
    consec = np.zeros(num_freq)
    consec[:max_freq] = 1
    return consec


def single_frequency_assignment(
    d_min: float,
    d_max: float,
    freq: float,
    delta_freq: float,
    num_freq: int,
    max_freq: int,
    h_tx: float,
    h_rx: float,
    c: float = constants.c,
):
    assignment = np.zeros(num_freq)
    assignment[0] = 1
    return assignment


def main(
    d_min: float,
    d_max: float,
    freq: float,
    delta_freq: float,
    num_freq: int,
    max_freq: int,
    h_tx: float,
    h_rx: float,
    c: float = constants.c,
    num_runs: int = 100,
    export: bool = False,
):
    if max_freq is None:
        max_freq = num_freq

    LOGGER.info(
        f"Assigning Frequencies with following parameters: dmin={d_min:.1f}, dmax={d_max:.1f}, h_tx={h_tx:.1f}, h_rx={h_rx:.1f}, f0={freq:E}, df={delta_freq:E}, N={num_freq:d}, K={max_freq:d}"
    )

    ALGORITHMS = {
        "Greedy": (greedy_frequency_assignment, 1),
        "Random": (random_frequency_assignment, num_runs),
        "Consecutive": (consecutive_frequency_assignment, 1),
        "Uniform": (uniform_frequency_assignment, 1),
        "Single": (single_frequency_assignment, 1),
    }

    results = {k: [] for k in ALGORITHMS}

    for name_alg, (func_alg, num_runs_alg) in ALGORITHMS.items():
        LOGGER.info(f"Working on algorithm: {name_alg}")
        for t in range(num_runs_alg):
            LOGGER.info(f"Run {t+1:d}/{num_runs_alg:d}")
            _assignment = func_alg(
                d_min, d_max, freq, delta_freq, num_freq, max_freq, h_tx, h_rx
            )
            assert np.count_nonzero(_assignment) <= max_freq
            _worst_case_power = calc_worst_case_power_from_assignment(
                _assignment,
                d_min,
                d_max,
                freq,
                delta_freq,
                num_freq,
                max_freq,
                h_tx,
                h_rx,
            )
            _worst_case_power_db = to_decibel(_worst_case_power)
            results[name_alg].append(_worst_case_power_db)

    results = {k: np.mean(v) for k, v in results.items()}

    for _name, _results in results.items():
        LOGGER.info(f"{_name}:\t{_results:.1f}")

    _line_results = " & ".join([f"\\num{{{v:.1f}}}" for k, v in results.items()])
    line = f"\\num{{{num_freq:d}}} & \\num{{{delta_freq//1e6:.0f}}} & \\num{{{max_freq:d}}} & \\num{{{d_min:.0f}}} & \\num{{{d_max:.0f}}} & {_line_results} \\\\"
    LOGGER.info(f"LaTeX Table line: {line}")
    if export:
        with open("latex-output.txt", "a") as out_file:
            out_file.write(f"{line}\n")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--h_tx", type=float, default=10.0)
    parser.add_argument("-r", "--h_rx", type=float, default=1.0)
    parser.add_argument("-f", "--freq", type=float, default=2.4e9)
    parser.add_argument("-df", "--delta_freq", type=float, default=10e6)
    parser.add_argument("-dmin", "--d_min", type=float, default=10.0)
    parser.add_argument("-dmax", "--d_max", type=float, default=100.0)
    parser.add_argument("-n", "--num_freq", type=int, default=100)
    parser.add_argument("-m", "--max_freq", type=int)
    parser.add_argument("--num_runs", type=int, default=100)
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

    np.random.seed(20230824)
    main(**args)
