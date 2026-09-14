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


def test_extract_direct_hfacf_ravg_fractions_are_independent_of_call_order():
    """Regression test: the running average used to be recomputed inside
    the loop over hcacf_extract_values, reassigning `kappa`/`kappa_err` to
    the smoothed mean/uncertainty on every iteration. With hfacf_ravg=True
    and more than one requested fraction, this meant every fraction after
    the first got smoothed again on top of the already-smoothed array.
    Fixed by computing the running average once, before the loop."""
    rng = np.random.default_rng(0)
    n = 50
    N = 10
    kappa = rng.normal(loc=5.0, scale=0.1, size=n)
    kappa_err = np.full(n, 0.2)

    multi, multi_errs = extract_direct(
        [0.25, 0.75], kappa.copy(), kappa_err.copy(), hfacf_ravg=True, N=N
    )
    single_25, single_25_errs = extract_direct(
        [0.25], kappa.copy(), kappa_err.copy(), hfacf_ravg=True, N=N
    )
    single_75, single_75_errs = extract_direct(
        [0.75], kappa.copy(), kappa_err.copy(), hfacf_ravg=True, N=N
    )

    assert multi[0.25] == pytest.approx(single_25[0.25])
    assert multi[0.75] == pytest.approx(single_75[0.75])
    assert multi_errs[0.25] == pytest.approx(single_25_errs[0.25])
    assert multi_errs[0.75] == pytest.approx(single_75_errs[0.75])
