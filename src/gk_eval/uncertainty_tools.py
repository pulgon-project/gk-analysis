# Copyright 2025 The PULGON Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import numpy as np
from scipy.signal import correlate
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid
import numba


eV2J = 1.60218e-19
kB = 1.38064852e-23
A2m = 1.0e-10
ps2s = 1.0e-12
fs2s = 1.0e-15
flux2SI = eV2J / A2m**2 / ps2s


SI_PREFACTOR = flux2SI**2 * A2m**3 * fs2s
HCACF_UNIT = flux2SI**2
time_factor = 1.0


def calc_correlation_old(data, num_pieces):
    """
    Function to compute the autocorrelation out of a set of data by first splitting it into individual pieces.
    Evaluates the uncertainty solely based on the differences between the pieces.
    The differences of the correlation function for each piece and its mean are also given.
    """
    pieced_data = np.split(data, num_pieces)
    corrfuncs = []
    plen = len(pieced_data[0])
    contributions = np.array([plen - k for k in range(plen - 1)])
    # this is BIG - better to just evaluate the required sum
    # covariance = np.zeros((len(pieced_data[0])-1, len(pieced_data[0])-1))
    caf2 = np.zeros(plen - 1)
    for piece in pieced_data:

        corrfuncs.append(
            correlate(piece, piece, method="fft")[len(pieced_data[0]) :] / contributions
        )
        caf2 += correlate(piece**2, piece**2)[len(pieced_data[0]) :]

    corrfunc = np.mean(corrfuncs, axis=0)
    caf2 /= num_pieces * contributions

    corrdiff = np.zeros((num_pieces, plen - 1))
    for cid, cf in enumerate(corrfuncs):
        corrdiff[cid] = cf - corrfunc
    uncertainty = np.sqrt(
        1 / num_pieces * np.sum(corrdiff**2, axis=0)
    )  # / np.sqrt(num_pieces)
    unc_caf = np.sqrt(caf2 - corrfunc**2) / np.sqrt(num_pieces * contributions - 1)
    return corrfunc, uncertainty, unc_caf, corrdiff


def calc_correlation(data, num_pieces=None):
    """
    Function to compute the autocorrelation out of a set of data by first splitting it into individual pieces.
    Evaluates the uncertainty solely based on the differences between the pieces.
    The differences of the correlation function for each piece and its mean are also given.
    If num_pieces is None, assumes to get a 2 dimensional array for data that already corresponds to the pieces
    """
    if num_pieces is None:
        pieced_data = data
        num_pieces = data.shape[0]
    else:
        pieced_data = np.split(data, num_pieces)
    corrfunc = np.zeros(len(pieced_data[0]))
    corrfuncs = []
    plen = len(pieced_data[0])
    contributions = np.array([plen - k for k in range(plen - 1)])
    contributions = np.flip(np.array(range(len(pieced_data[0]))) + 1)
    # this is BIG - better to just evaluate the required sum
    # covariance = np.zeros((len(pieced_data[0])-1, len(pieced_data[0])-1))
    caf2 = np.zeros(plen)
    for piece in pieced_data:
        corrfuncs.append(
            correlate(piece, piece, method="fft")[len(pieced_data[0]) - 1 :]
            / contributions
        )
        corrfunc += corrfuncs[-1]
        caf2 += correlate(piece**2, piece**2)[len(pieced_data[0]) - 1 :]

    corrfunc = corrfunc / len(pieced_data)
    caf2 /= num_pieces * contributions

    corrdiff = np.zeros((num_pieces, plen))
    for cid, cf in enumerate(corrfuncs):
        corrdiff[cid] = cf - corrfunc
    uncertainty = np.sqrt(
        1 / (num_pieces) * np.sum(corrdiff**2, axis=0)
    )  # / np.sqrt(num_pieces)
    unc_caf = np.sqrt(caf2 - corrfunc**2) / np.sqrt(num_pieces * contributions - 1)
    return corrfunc, uncertainty, unc_caf, corrdiff


@numba.njit(
    parallel=False
)  # parallelization is suboptimal as the next value reuses the previous
def compute_cov_contrib(corrdiff: np.ndarray):
    """
    Compute the uncertainty contribution for the Euler integral.
    This does not store the covariance matrix in the memory and only computes the lower triangular matrix
    """
    M = corrdiff.shape[0]
    k = corrdiff.shape[1]
    covariance_contrib = np.zeros(k)

    for l in range(1, k):
        covariance_contrib[l] = covariance_contrib[l - 1]
        for j in range(M):
            for i in range(0, l):
                covariance_contrib[l] += corrdiff[j, l] * corrdiff[j, i]

    return covariance_contrib / (M - 1) / M


def calc_euler_integral(
    data,
    dt,
    unc_corr,
    corrdiff,
    include_cov=False,
    fast_cov=True,
    SI_PREFACTOR=SI_PREFACTOR,
    volume=1.0,
    temperature=300,
):
    """
    Computes a cumulative integral over a correlation function using an Euler integral.
    Can evaluate the uncertainties based on the covariance matrix, which is computed in the process.
    """
    cumul = dt * (np.cumsum(data))
    k = len(cumul)
    M = len(corrdiff)
    if include_cov:
        # it is excruciatingly slow if I do it in the memory sparse way
        covariance_contrib = np.zeros(k)
        if fast_cov:
            # this is somewhat faster and memory light
            covariance_contrib = compute_cov_contrib(corrdiff)
        else:
            # this eats a ton of RAM but is acceptably quick
            # is there an easy way to only compute the lower triangular matrix?
            cov = np.einsum("ij,ik->jk", corrdiff, corrdiff) / M
            # covariance_contrib[0] = np.sum(cov[: - 1, 1:0])
            for l in tqdm(range(1, k)):
                # covariance_contrib[l] = np.sum(cov[: l - 1, 1:l])
                # covariance_contrib[l] = covariance_contrib[l-1] + cov[l-1,l-1] + np.sum(cov[:l-1,l])
                # previous definition was not correct (but not far off - numbers are similar)
                covariance_contrib[l] = covariance_contrib[l - 1] + np.sum(cov[l, 0:l])
        # else:
        #     for l in tqdm(range(0, k)):
        #         for i in range(0, l - 1):
        #             for j in range(i, l):
        #                 covariance_contrib[l] += np.mean(
        #                     corrdiff[:, i] * corrdiff[:, j]
        #                 )

    else:
        covariance_contrib = 0
    n_contrib = dt**2 * np.cumsum(unc_corr**2)
    cov_contrib = dt**2 * 2 * covariance_contrib
    unc_cumul = n_contrib + cov_contrib
    pref = volume / (kB * temperature**2)
    return (
        cumul * SI_PREFACTOR * pref,
        np.sqrt(unc_cumul) * SI_PREFACTOR * pref,
        n_contrib * SI_PREFACTOR**2 * pref**2,
        cov_contrib * SI_PREFACTOR**2 * pref**2,
    )


def calc_cumtrapz_integral(
    data, dt, unc_caf, SI_PREFACTOR=SI_PREFACTOR, volume=1.0, temperature=300
):

    cumul = cumulative_trapezoid(data, np.arange(len(data)), initial=0) * dt
    unc_trapzs = np.zeros(len(data))
    unc_trapzs[1:] = np.array(
        [unc_caf[i] ** 2 + unc_caf[i - 1] ** 2 for i in range(1, len(cumul))]
    )
    unc_cumul = (dt / 2) * np.sqrt(np.cumsum(unc_trapzs))
    pref = volume / (kB * temperature**2)
    return cumul * SI_PREFACTOR * pref, unc_cumul * SI_PREFACTOR * pref
