import numpy as np
from scipy import linalg
from scipy import constants


@np.vectorize(excluded=["a", "b"])
def coeff_h(j: int, a: list, b: list):
    N = len(a) - 1
    if 0 <= j <= N - 1:
        hj = a[N - j] + 1j * b[N - j - 1]
    elif j == N:
        hj = 2 * a[0]
    elif N + 1 <= j <= 2 * N:
        hj = a[j - N] - 1j * b[j - N - 1]
    else:
        raise ValueError(f"j needs to be within [0, 2N]. Got j={j} and N={N}")
    return hj


def construct_frobenius_matrix_fourier_series(a: list, b: list):
    N = len(a) - 1
    assert len(b) == N
    frob_matrix = np.eye(2 * N, k=1, dtype=complex)
    frob_matrix[-1] = -1.0 * coeff_h(np.arange(2 * N), a=a, b=b) / (a[-1] - 1j * b[-1])
    return frob_matrix


def roots_from_frobenius_matrix(frob_matrix):
    eig_frob = linalg.eigvals(frob_matrix)
    roots = np.angle(eig_frob) - 1j * np.log(np.abs(eig_frob))
    roots = roots[np.isclose(np.imag(roots), 0)]
    roots = np.real_if_close(roots)
    return roots


def calc_fourier_coefficients(freqs):
    freqs = np.array(freqs)
    num_freqs = len(freqs)
    omega = 2 * np.pi * freqs
    _omega_t = np.tile(omega, (num_freqs, 1))
    omega_combinations = 1.0 / (_omega_t * _omega_t.T) ** 2
    coeff_a = [2 * np.trace(omega_combinations, offset=k) for k in range(num_freqs)]
    coeff_a[0] *= 0.5
    coeff_a = np.array(coeff_a)
    normalization_factor = np.max(np.abs(coeff_a))
    coeff_a = coeff_a / normalization_factor  # for numerical stability
    coeff_b = np.zeros(len(coeff_a) - 1)
    return coeff_a, coeff_b


def test_frob_matrix_2(a: list):
    if len(a) != 3:
        raise ValueError(
            "This function is only used to test the construction of the Frobenius matrix for N=2"
        )

    b = np.zeros(len(a) - 1)
    frob_matrix = construct_frobenius_matrix_fourier_series(a, b)
    expected = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [
                -(a[2] + 1j * b[1]) / (a[2] - 1j * b[1]),
                -(a[1] + 1j * b[0]) / (a[2] - 1j * b[1]),
                -2 * a[0] / (a[2] - 1j * b[1]),
                -(a[1] - 1j * b[0]) / (a[2] - 1j * b[1]),
            ],
        ]
    )
    assert np.allclose(frob_matrix, expected)


def main(a: list, b: list = None):
    num_components = len(a)
    if b is None:
        b = np.zeros(num_components - 1)
    b_deriv = a[1:] * -(np.arange(num_components - 1) + 1)
    frob_matrix = construct_frobenius_matrix_fourier_series(
        a=np.zeros(num_components), b=b_deriv
    )
    roots = roots_from_frobenius_matrix(frob_matrix)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots()
    x = np.linspace(-np.pi, np.pi, 1000)
    f = np.sum([_a * np.cos(j * x) for j, _a in enumerate(a)], axis=0) + np.sum(
        [_b * np.sin((j + 1) * x) for j, _b in enumerate(b)], axis=0
    )
    axs.hlines(0, min(x), max(x), "k")
    axs.vlines(roots, min(f), max(f), "r", ls="--")
    axs.plot(x, f)
    plt.show()


if __name__ == "__main__":
    test_frob_matrix_2(a=[0.25, 1.0, 0.5])
    a = [-0.25, 0.75, 0.5, 0.25, 0.5, 0.5]
    main(a)
