import numpy as np
import pytest

from gk_analysis.systematic_GK_analysis import extract_direct


def test_extract_direct_fixed_fraction_no_running_average():
    kappa = np.linspace(0.0, 10.0, 21)
    kappa_err = np.full(21, 0.5)

    kappas, errs = extract_direct(
        [0.0, 0.25, 0.5, 0.75, 1.0], kappa, kappa_err, hfacf_ravg=False, N=5
    )

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        idx = int((len(kappa) - 1) * frac)
        assert kappas[frac] == kappa[idx]
        assert errs[frac] == kappa_err[idx]

    # fraction 1.0 now resolves to the last valid index instead of raising
    assert kappas[1.0] == kappa[-1]
    assert errs[1.0] == kappa_err[-1]


def test_extract_direct_running_average_hand_checked():
    rng = np.random.default_rng(0)
    n = 50
    N = 10
    kappa = rng.normal(loc=5.0, scale=0.1, size=n)
    kappa_err = np.full(n, 0.2)

    kappas, errs = extract_direct([0.5], kappa.copy(), kappa_err.copy(), hfacf_ravg=True, N=N)

    mean = np.convolve(kappa, np.ones(N) / N, mode="valid")
    mean_sq = np.convolve(kappa**2, np.ones(N) / N, mode="valid")
    var = np.clip(mean_sq - mean**2, 0, None)
    expected_err = np.sqrt(var) / np.sqrt(N)

    idx = int((len(mean) - 1) * 0.5)
    assert kappas[0.5] == pytest.approx(mean[idx])
    assert errs[0.5] == pytest.approx(expected_err[idx])


def test_extract_direct_running_average_warns_when_not_enough_data(capsys):
    n = 3
    N = 10  # window larger than the data -> xvals is empty
    kappa = np.zeros(n)
    kappa_err = np.zeros(n)

    kappas, errs = extract_direct([0.5], kappa.copy(), kappa_err.copy(), hfacf_ravg=True, N=N)

    captured = capsys.readouterr()
    assert "cannot compute running average" in captured.out
    # falls back to using the (unmodified) input kappa/kappa_err directly
    idx = int((len(kappa) - 1) * 0.5)
    assert kappas[0.5] == kappa[idx]
    assert errs[0.5] == kappa_err[idx]


def test_extract_direct_hfacf_ravg_over_smooths_later_fractions_in_same_call():
    """Known bug: inside the loop over hcacf_extract_values, `kappa`/`kappa_err`
    are reassigned to the smoothed mean/uncertainty (`kappa = mean`,
    `kappa_err = ...`). With hfacf_ravg=True and more than one requested
    fraction, every fraction after the first gets the running average
    applied AGAIN on top of the already-smoothed array from the previous
    iteration, instead of each fraction being computed independently from
    the original raw data. This test pins that current, order-dependent
    behaviour rather than asserting the (presumably intended) independence."""
    rng = np.random.default_rng(0)
    n = 50
    N = 10
    kappa = rng.normal(loc=5.0, scale=0.1, size=n)
    kappa_err = np.full(n, 0.2)

    multi, _ = extract_direct([0.25, 0.75], kappa.copy(), kappa_err.copy(), hfacf_ravg=True, N=N)
    single_75, _ = extract_direct([0.75], kappa.copy(), kappa_err.copy(), hfacf_ravg=True, N=N)

    # the second fraction in a multi-fraction call does NOT match the same
    # fraction computed on its own, because it was smoothed twice
    assert multi[0.75] != pytest.approx(single_75[0.75])
