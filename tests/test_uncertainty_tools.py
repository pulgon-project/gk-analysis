import numpy as np
import pytest

from gk_analysis import uncertainty_tools as ut


# ---------------------------------------------------------------------------
# calc_correlation
# ---------------------------------------------------------------------------


def test_calc_correlation_constant_series_is_flat():
    """For a constant series, the ACF must equal series**2 at every lag."""
    c = 3.5
    num_pieces = 4
    plen = 10
    data = np.full(num_pieces * plen, c)

    with np.errstate(invalid="ignore"):
        # unc_caf involves sqrt(caf2 - corrfunc**2), which is ~0 here and can
        # go slightly negative from floating-point roundoff for a constant
        # series -- not something this test is exercising.
        corrfunc, uncertainty, unc_caf, corrdiff = ut.calc_correlation(
            data, num_pieces=num_pieces
        )

    assert corrfunc.shape == (plen,)
    assert np.allclose(corrfunc, c**2)
    # no piece-to-piece variation for a constant series
    assert np.allclose(uncertainty, 0.0)


def test_calc_correlation_shapes_with_num_pieces():
    num_pieces = 5
    plen = 8
    rng = np.random.default_rng(0)
    data = rng.normal(size=num_pieces * plen)

    corrfunc, uncertainty, unc_caf, corrdiff = ut.calc_correlation(
        data, num_pieces=num_pieces
    )

    assert corrfunc.shape == (plen,)
    assert uncertainty.shape == (plen,)
    assert unc_caf.shape == (plen,)
    assert corrdiff.shape == (num_pieces, plen)


def test_calc_correlation_pre_pieced_2d_input_matches_1d_split():
    num_pieces = 4
    plen = 6
    rng = np.random.default_rng(1)
    data = rng.normal(size=num_pieces * plen)

    ref = ut.calc_correlation(data, num_pieces=num_pieces)
    pieced = data.reshape(num_pieces, plen)
    out = ut.calc_correlation(pieced, num_pieces=None)

    for a, b in zip(ref, out):
        assert np.allclose(a, b)


def test_calc_correlation_corrdiff_sums_to_zero_across_pieces():
    num_pieces = 6
    plen = 9
    rng = np.random.default_rng(2)
    data = rng.normal(size=num_pieces * plen)

    _, _, _, corrdiff = ut.calc_correlation(data, num_pieces=num_pieces)

    # corrdiff[i] = corrfuncs[i] - mean(corrfuncs); by construction the
    # per-lag deviations from the mean must sum to zero across pieces.
    assert np.allclose(np.sum(corrdiff, axis=0), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_cov_contrib
# ---------------------------------------------------------------------------


def _brute_force_cov_contrib(corrdiff):
    """Independent, unoptimized re-implementation of the documented formula:
    the lower-triangular sum of the (M x M averaged) covariance matrix,
    accumulated cumulatively over lag and normalized once by M*(M-1)."""
    M, k = corrdiff.shape
    contrib = np.zeros(k)
    for l in range(1, k):
        s = 0.0
        for j in range(M):
            for i in range(l):
                s += corrdiff[j, l] * corrdiff[j, i]
        contrib[l] = contrib[l - 1] + s
    return contrib / (M - 1) / M


def test_compute_cov_contrib_matches_brute_force():
    rng = np.random.default_rng(4)
    corrdiff = rng.normal(size=(5, 7))

    fast = ut.compute_cov_contrib(corrdiff)
    brute = _brute_force_cov_contrib(corrdiff)

    assert fast.shape == (7,)
    assert fast[0] == 0.0
    assert np.allclose(fast, brute)


# ---------------------------------------------------------------------------
# calc_euler_integral
# ---------------------------------------------------------------------------


def test_calc_euler_integral_include_cov_false_ignores_corrdiff():
    rng = np.random.default_rng(5)
    n = 10
    data = rng.normal(size=n)
    unc_corr = np.abs(rng.normal(size=n))
    corrdiff = rng.normal(size=(4, n))

    cumul, unc_cumul, n_contrib, cov_contrib = ut.calc_euler_integral(
        data, dt=0.5, unc_corr=unc_corr, corrdiff=corrdiff, include_cov=False
    )
    assert np.all(cov_contrib == 0)
    # without covariance, uncertainty is purely from the n_contrib term
    assert np.allclose(unc_cumul**2, n_contrib)


def test_calc_euler_integral_fast_cov_matches_compute_cov_contrib():
    """fast_cov=True must be consistent with compute_cov_contrib itself."""
    rng = np.random.default_rng(6)
    n = 8
    data = rng.normal(size=n)
    unc_corr = np.abs(rng.normal(size=n))
    corrdiff = rng.normal(size=(4, n))
    dt = 0.25
    volume = 1.0
    temperature = 300.0

    cumul, unc_cumul, n_contrib, cov_contrib = ut.calc_euler_integral(
        data,
        dt,
        unc_corr,
        corrdiff,
        include_cov=True,
        fast_cov=True,
        SI_PREFACTOR=1.0,
        volume=volume,
        temperature=temperature,
    )

    expected_cov_contrib_raw = ut.compute_cov_contrib(corrdiff)
    expected_cov_contrib = dt**2 * 2 * expected_cov_contrib_raw
    pref = volume / (ut.kB * temperature**2)
    assert np.allclose(cov_contrib, expected_cov_contrib * pref**2)


def test_calc_euler_integral_fast_and_slow_cov_agree():
    """fast_cov=True (compute_cov_contrib) and fast_cov=False (the plain
    einsum + cumulative-sum path) must produce equivalent uncertainties."""
    rng = np.random.default_rng(7)
    n = 6
    data = rng.normal(size=n)
    unc_corr = np.abs(rng.normal(size=n))
    corrdiff = rng.normal(size=(5, n))

    fast = ut.calc_euler_integral(
        data, dt=0.1, unc_corr=unc_corr, corrdiff=corrdiff,
        include_cov=True, fast_cov=True,
    )
    slow = ut.calc_euler_integral(
        data, dt=0.1, unc_corr=unc_corr, corrdiff=corrdiff,
        include_cov=True, fast_cov=False,
    )

    assert np.allclose(fast[1], slow[1])


def test_calc_euler_integral_analytic_constant_data():
    n = 20
    dt = 0.3
    value = 2.0
    data = np.full(n, value)
    unc_corr = np.zeros(n)
    corrdiff = np.zeros((3, n))

    cumul, unc_cumul, n_contrib, cov_contrib = ut.calc_euler_integral(
        data,
        dt,
        unc_corr,
        corrdiff,
        include_cov=False,
        SI_PREFACTOR=1.0,
        volume=1.0,
        temperature=1.0 / ut.kB,  # makes pref = volume / (kB * T**2) = kB
    )

    expected = dt * np.cumsum(data) * ut.kB
    assert np.allclose(cumul, expected)
    assert np.allclose(unc_cumul, 0.0)


def test_calc_euler_integral_unit_scaling():
    rng = np.random.default_rng(8)
    n = 5
    data = rng.normal(size=n)
    unc_corr = np.abs(rng.normal(size=n))
    corrdiff = rng.normal(size=(3, n))
    dt = 1.0
    SI_PREFACTOR = 7.0
    volume = 2.0
    temperature = 50.0
    pref = volume / (ut.kB * temperature**2)

    scaled = ut.calc_euler_integral(
        data, dt, unc_corr, corrdiff, include_cov=True,
        SI_PREFACTOR=SI_PREFACTOR, volume=volume, temperature=temperature,
    )

    assert np.allclose(scaled[0], dt * np.cumsum(data) * SI_PREFACTOR * pref)
    assert np.allclose(scaled[2], dt**2 * np.cumsum(unc_corr**2) * SI_PREFACTOR**2 * pref**2)


# ---------------------------------------------------------------------------
# calc_cumtrapz_integral
# ---------------------------------------------------------------------------


def test_calc_cumtrapz_integral_matches_scipy_and_unc_propagation():
    from scipy.integrate import cumulative_trapezoid

    rng = np.random.default_rng(9)
    n = 12
    data = rng.normal(size=n)
    unc_caf = np.abs(rng.normal(size=n))
    dt = 0.4
    SI_PREFACTOR = 3.0
    volume = 1.5
    temperature = 100.0
    pref = volume / (ut.kB * temperature**2)

    cumul, unc_cumul = ut.calc_cumtrapz_integral(
        data, dt, unc_caf, SI_PREFACTOR=SI_PREFACTOR, volume=volume, temperature=temperature
    )

    expected_cumul = (
        cumulative_trapezoid(data, np.arange(n), initial=0) * dt * SI_PREFACTOR * pref
    )
    assert np.allclose(cumul, expected_cumul)

    unc_trapzs = np.zeros(n)
    unc_trapzs[1:] = unc_caf[1:] ** 2 + unc_caf[:-1] ** 2
    expected_unc = (dt / 2) * np.sqrt(np.cumsum(unc_trapzs)) * SI_PREFACTOR * pref
    assert np.allclose(unc_cumul, expected_unc)
