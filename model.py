import numpy as np


def length_los(distance, h_tx, h_rx):
    distance = np.array(distance)
    return np.sqrt(distance**2 + (h_tx - h_rx) ** 2)


def length_ref(distance, h_tx, h_rx):
    distance = np.array(distance)
    return np.sqrt(distance**2 + (h_tx + h_rx) ** 2)


def get_distance_from_delta_len(delta_len, h_tx, h_rx):
    delta_len = np.array(delta_len)
    _arg = (4 * h_rx**2 - delta_len**2) * (4 * h_tx**2 - delta_len**2)
    _solvable = np.where(_arg >= 0)
    _distance = np.sqrt(_arg[_solvable]) / (2 * delta_len[_solvable])
    distance = np.zeros_like(delta_len) * np.nan
    distance[_solvable] = _distance
    distance = np.real(distance)
    return distance
