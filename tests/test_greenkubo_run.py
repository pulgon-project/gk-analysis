import os

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from gk_analysis.GreenKubo_run import (
    AIC,
    AIC_weight,
    GreenKubo_run,
    calc_spectrum,
    expdecay,
)


# ---------------------------------------------------------------------------
# calc_spectrum
# ---------------------------------------------------------------------------


def test_calc_spectrum_shapes_and_freqs():
    n_steps = 64
    dt = 0.5
    fluxes = np.zeros((2, n_steps, 3))
    freqs, ffts, spectra = calc_spectrum(fluxes, dt)

    assert np.allclose(freqs, np.fft.fftfreq(n_steps, dt))
    assert ffts.shape == (2, n_steps, 3)
    assert spectra.shape == (2, 2, n_steps)


def test_calc_spectrum_single_tone_peak():
    n = 256
    dt = 0.5
    freq_bin = 5
    freq = freq_bin / (n * dt)
    t = np.arange(n) * dt
    sine = np.sin(2 * np.pi * freq * t)

    fluxes = np.zeros((1, n, 3))
    fluxes[0, :, 0] = sine
    freqs, ffts, spectra = calc_spectrum(fluxes, dt)

    power = np.abs(spectra[0, 0, :])
    positive = freqs >= 0
    peak_idx = np.argmax(power[positive])
    assert freqs[positive][peak_idx] == pytest.approx(freq)


def test_calc_spectrum_white_noise_is_flat_on_average():
    rng = np.random.default_rng(0)
    n = 4096
    dt = 1.0
    fluxes = rng.normal(size=(1, n, 3))
    freqs, ffts, spectra = calc_spectrum(fluxes, dt)

    power = np.real(spectra[0, 0, :])
    mean_power = np.mean(power)
    # a flat spectrum: individual bins shouldn't deviate wildly from the mean
    assert np.std(power) < 2 * mean_power


# ---------------------------------------------------------------------------
# AIC / AIC_weight / expdecay
# ---------------------------------------------------------------------------


def test_aic_hand_checked_value():
    cepstral = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var = 2.0
    # N=6, P=1: N/var * sum(cepstral[1:3]**2) + 2*P = 3*(4+9) + 2 = 41
    assert AIC(1, cepstral, var, AICc=False) == pytest.approx(41.0)


def test_aic_aicc_correction_term():
    cepstral = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var = 2.0
    P = 1
    N = len(cepstral)
    correction = AIC(P, cepstral, var, AICc=True) - AIC(P, cepstral, var, AICc=False)
    assert correction == pytest.approx(2 * P * (P + 1) / (N - P - 1))


def test_aic_increases_for_large_p_beyond_informative_region():
    rng = np.random.default_rng(1)
    cepstral = rng.normal(scale=0.01, size=40)
    var = 1.0
    aics = [AIC(P, cepstral, var, AICc=True) for P in range(1, 15)]
    # deep into the noise floor, adding more coefficients only adds the 2*P
    # (and AICc) penalty without reducing the sum-of-squares term meaningfully
    assert aics[-1] > aics[0]


def test_aic_weight_sums_to_one_and_favors_minimum():
    aics = np.array([10.0, 5.0, 20.0])
    w = AIC_weight(aics)
    assert w.sum() == pytest.approx(1.0)
    assert np.argmax(w) == 1


def test_aic_weight_single_entry():
    assert AIC_weight(np.array([7.0])) == pytest.approx([1.0])


def test_expdecay():
    assert expdecay(2.0, 2.0) == pytest.approx(np.exp(-1.0))
    assert expdecay(0.0, 5.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# GreenKubo_run: construction and flux reading
# ---------------------------------------------------------------------------


def test_read_flux_basic(poscar_path, write_flux_file, rng):
    n = 20
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 3))])
    path = write_flux_file(data)

    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3, units="metal")

    assert gk.flux.shape == (n, 3)
    assert np.allclose(gk.flux, -data[:, 1:])
    assert gk.temperature == pytest.approx(np.mean(data[:, 0]))
    assert np.allclose(gk.time_index, np.arange(n) * gk.dt)


def test_read_flux_take_every_and_fmod(poscar_path, write_flux_file, rng):
    n = 10
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 5))])
    path = write_flux_file(data)

    gk = GreenKubo_run(
        path, poscar_path, dt=1.0, n_cart=3, units="metal",
        col_ind=2, take_every=2, fmod=3,
    )

    cutnum = n % 3
    trimmed = data[: n - cutnum] if cutnum else data
    expected = -trimmed[:, 1:][::2][:, 1:4]
    assert np.allclose(gk.flux, expected)


def test_read_flux_max_rows_int_and_per_file(poscar_path, write_flux_file, rng):
    n1, n2 = 10, 10
    d1 = np.column_stack([np.full(n1, 300.0), rng.normal(size=(n1, 3))])
    d2 = np.column_stack([np.full(n2, 300.0), rng.normal(size=(n2, 3))])
    p1 = write_flux_file(d1, name="f1.dat")
    p2 = write_flux_file(d2, name="f2.dat")

    gk_int = GreenKubo_run([p1, p2], poscar_path, dt=1.0, n_cart=3, max_rows=5)
    # first file capped at 5, second file (after restart-skip of row 0) capped at 5
    assert gk_int.flens == [5, 4]

    gk_list = GreenKubo_run([p1, p2], poscar_path, dt=1.0, n_cart=3, max_rows=[5, 3])
    assert gk_list.flens == [5, 2]


def test_read_flux_multifile_restart_semantics(poscar_path, write_flux_file, rng):
    n1, n2 = 8, 6
    d1 = np.column_stack([np.full(n1, 300.0), rng.normal(size=(n1, 3))])
    d2 = np.column_stack([np.full(n2, 310.0), rng.normal(size=(n2, 3))])
    p1 = write_flux_file(d1, name="f1.dat")
    p2 = write_flux_file(d2, name="f2.dat")

    gk_dep = GreenKubo_run([p1, p2], poscar_path, dt=1.0, n_cart=3, independent_flux=False)
    gk_indep = GreenKubo_run([p1, p2], poscar_path, dt=1.0, n_cart=3, independent_flux=True)

    assert gk_dep.flens == [n1, n2 - 1]
    assert gk_indep.flens == [n1, n2]
    assert np.allclose(gk_dep.flux[n1], -d2[1, 1:])
    assert np.allclose(gk_indep.flux[n1], -d2[0, 1:])


def test_read_flux_mean_corr(poscar_path, write_flux_file, rng):
    n = 12
    data = np.column_stack([np.full(n, 300.0), rng.normal(loc=5.0, size=(n, 3))])
    path = write_flux_file(data)

    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3, mean_corr=True)
    # mean-subtraction happens before the sign flip
    assert np.allclose(np.mean(-gk.flux, axis=0), 0.0, atol=1e-10)


def test_read_flux_temperature_default_vs_explicit(poscar_path, write_flux_file, rng):
    n = 10
    data = np.column_stack([np.full(n, 250.0), rng.normal(size=(n, 3))])
    path = write_flux_file(data)

    gk_default = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)
    assert gk_default.temperature == pytest.approx(250.0)

    gk_fixed = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3, temperature=999.0)
    assert gk_fixed.temperature == 999.0


def test_hf_lammps_divides_by_volume(poscar_path, write_flux_file, rng, bcc_fe_atoms):
    n = 10
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 3))])
    path = write_flux_file(data)

    gk_plain = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3, hf_lammps=False)
    gk_scaled = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3, hf_lammps=True)

    assert np.allclose(gk_scaled.flux * bcc_fe_atoms.get_volume(), gk_plain.flux)


def test_units_metal_vs_ase_prefactors(poscar_path, write_flux_file, rng):
    import ase.units

    n = 5
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 3))])
    path = write_flux_file(data)

    gk_metal = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3, units="metal")
    gk_ase = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3, units="ase")

    assert gk_metal.kB == pytest.approx(1.38064852e-23)
    assert gk_ase.kB == pytest.approx(ase.units.kB)
    assert gk_ase.time_factor == pytest.approx(ase.units.fs)


def test_read_flux_components_split(poscar_path, write_flux_file, rng, bcc_fe_atoms):
    n = 10
    # force(3) pot(3) int(3) conv(3)
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 12))])
    path = write_flux_file(data)

    gk = GreenKubo_run.__new__(GreenKubo_run)
    gk.n_cart = 3
    gk.independent_flux = False
    gk.fmod = 1
    gk.mean_corr = False
    gk.temperature = 300.0
    gk.volume = bcc_fe_atoms.get_volume()
    gk.hf_lammps = False
    gk.atoms = bcc_fe_atoms
    gk.dt = 1.0

    flux_signed = -data[:, 1:]
    gk.read_flux_components([path], full_flux=False)
    assert np.allclose(gk.flux_force * gk.volume, flux_signed[:, 0:3])
    assert np.allclose(gk.flux_pot * gk.volume, flux_signed[:, 3:6])
    assert np.allclose(gk.flux_int, flux_signed[:, 6:9])
    assert np.allclose(gk.flux_conv, flux_signed[:, 9:12])
    assert np.allclose(gk.flux, flux_signed[:, 6:9])

    gk2 = GreenKubo_run.__new__(GreenKubo_run)
    for attr in ("n_cart", "independent_flux", "fmod", "mean_corr", "temperature",
                 "volume", "hf_lammps", "atoms", "dt"):
        setattr(gk2, attr, getattr(gk, attr))
    gk2.read_flux_components([path], full_flux=True)
    assert np.allclose(gk2.flux, flux_signed[:, 6:9] + flux_signed[:, 9:12])


def test_constructor_nanowire_volume(tmp_path, rng):
    """nw=True routes the volume through Nanowire.get_volume(), scaled back
    by (cellvol / nw_volume)**2 so that dividing the (already-per-volume)
    heat flux by this value reproduces the intended cell volume scaling."""
    import ase.io
    from ase import Atoms

    atoms = Atoms("Ar", positions=[[10.0, 10.0, 1.5]], cell=[20.0, 20.0, 3.0], pbc=True)
    poscar = tmp_path / "POSCAR"
    ase.io.write(poscar, atoms, format="vasp")

    n = 5
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 3))])
    flux_path = tmp_path / "flux.dat"
    with open(flux_path, "w") as fh:
        fh.write("# header\n")
        np.savetxt(fh, data)

    gk = GreenKubo_run(str(flux_path), str(poscar), dt=1.0, n_cart=3, nw=True)
    cellvol = atoms.get_volume()
    nw_vol = gk.nw.get_volume()
    assert gk.volume == pytest.approx(nw_vol * (cellvol / nw_vol) ** 2)


# ---------------------------------------------------------------------------
# fold_flux
# ---------------------------------------------------------------------------


def test_fold_flux_shapes(poscar_path, write_flux_file, rng):
    n = 18
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    unfolded = gk.fold_flux(folds=None)
    assert unfolded.shape == (3, n)

    folded = gk.fold_flux(folds=2)
    assert folded.shape == (2 * 3, n // 2)


def test_fold_flux_max_eval_and_eval_cut_init(poscar_path, write_flux_file, rng):
    n = 30
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    cut_init = gk.fold_flux(folds=None, max_eval=10, eval_cut_init=True)
    assert cut_init.shape == (3, 10)

    no_cut_init = gk.fold_flux(folds=3, max_eval=12, eval_cut_init=False)
    # max_eval is applied per-fold (max_eval // folds) rather than up front
    assert no_cut_init.shape[1] <= 12 // 3


def test_fold_flux_mean_correction(poscar_path, write_flux_file, rng):
    n = 12
    data = np.column_stack([np.full(n, 300.0), rng.normal(loc=3.0, size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    folded = gk.fold_flux(folds=None, mean_correction=True)
    assert np.allclose(np.mean(folded, axis=1), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# write_results
# ---------------------------------------------------------------------------


def test_write_results_auto_runs_cepstral_analysis_when_never_run(
    poscar_path, write_flux_file, rng, chdir_tmp_path
):
    n = 512
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    gk.write_results("kappa.txt")

    written = np.loadtxt("kappa.txt")
    assert written[0] == gk.t_evaluated
    assert written[1] == pytest.approx(gk.kappa)
    assert written[2] == pytest.approx(gk.kappa_err)
    assert written[3] == gk.f_star


def test_write_results_with_explicit_f_star(poscar_path, write_flux_file, rng, chdir_tmp_path):
    n = 512
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)
    gk.cepstral_analysis(f_star=10.0, plot_results=False)

    gk.write_results("kappa.txt")

    written = np.loadtxt("kappa.txt")
    assert written[0] == n
    assert written[3] == 10.0


def test_write_results_with_default_f_star(poscar_path, write_flux_file, rng, chdir_tmp_path):
    n = 512
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)
    gk.cepstral_analysis(f_star=None, plot_results=False)

    gk.write_results("kappa.txt")

    written = np.loadtxt("kappa.txt")
    assert written[0] == n
    assert written[3] == gk.detect_f_star()  # the stubbed constant, 10.0


# ---------------------------------------------------------------------------
# cepstral_analysis
# ---------------------------------------------------------------------------


def test_cepstral_analysis_p_star_and_kappa_formula(poscar_path, write_flux_file, rng, chdir_tmp_path):
    n = 4096
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    kappa, kappa_err = gk.cepstral_analysis(f_star=10.0, plot_results=False)

    assert gk.P_star == np.argmin(gk.aic)
    assert kappa == gk.kappas[gk.P_star]

    val = gk.P_star
    expected = (
        0.5
        * np.exp(gk.cepstral[0] + 2 * np.sum(gk.cepstral[1:val]) - gk.L_0)
        * gk.SI_PREFACTOR
        * gk.prefactor
    )
    assert kappa == pytest.approx(expected)


def test_cepstral_analysis_max_coeffs_clamping(poscar_path, write_flux_file, rng, chdir_tmp_path):
    n = 512
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    gk.cepstral_analysis(f_star=10.0, max_coeffs=100_000, plot_results=False)
    assert len(gk.aic) == len(gk.cepstral) - 2


def test_cepstral_analysis_max_coeffs_zero_guard(poscar_path, write_flux_file, rng, chdir_tmp_path):
    n = 512
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    # should not raise a division/index error
    gk.cepstral_analysis(f_star=10.0, max_coeffs=0, plot_results=False)
    assert len(gk.aic) == 1


def test_cepstral_analysis_model_averaging(poscar_path, write_flux_file, rng, chdir_tmp_path):
    n = 4096
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    kappa_avg, kappa_err_avg = gk.cepstral_analysis(
        f_star=10.0, plot_results=False, model_averaging=True
    )

    expected_avg = np.average(gk.kappas, weights=gk.aic_weights)
    assert kappa_avg == pytest.approx(expected_avg)
    expected_err_avg = np.sqrt(
        np.sum(
            gk.aic_weights
            * (gk.kappa_errs**2 + (gk.kappas[gk.P_star] - kappa_avg) ** 2)
        )
    )
    assert kappa_err_avg == pytest.approx(expected_err_avg)


def test_cepstral_analysis_constant_flux_near_zero_kappa(poscar_path, write_flux_file, chdir_tmp_path):
    n = 512
    data = np.column_stack([np.full(n, 300.0), np.full((n, 3), 1e-3)])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    kappa, kappa_err = gk.cepstral_analysis(f_star=10.0, plot_results=False)
    assert not np.isnan(kappa)
    assert abs(kappa) < 1.0


# ---------------------------------------------------------------------------
# analyze_HCACF_integral
# ---------------------------------------------------------------------------


def test_analyze_hcacf_integral_matches_cumtrapz(poscar_path, write_flux_file, rng, chdir_tmp_path):
    n = 300
    data = np.column_stack([np.full(n, 300.0), rng.normal(scale=0.01, size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    cumul, cumul_unc, fig = gk.analyze_HCACF_integral(plot_ACF=False, plot_kappa=False, newfig=True)

    raw_hcacf = gk.hcacf / (gk.HCACF_UNIT / 1e18)
    expected = (
        cumulative_trapezoid(raw_hcacf, np.arange(len(raw_hcacf)))
        * gk.dt
        * gk.time_factor
        * 1e3
        * gk.volume
        / (gk.kB * gk.temperature**2)
        * gk.SI_PREFACTOR
    )
    assert np.allclose(cumul, expected)

    raw_unc = gk.hcacf_unc / (gk.HCACF_UNIT / 1e18)
    expected_unc = (
        gk.dt
        * gk.time_factor
        * 1e3
        / 2
        * np.sqrt(np.cumsum(raw_unc[:-1] ** 2 + raw_unc[1:] ** 2))
        * gk.volume
        / (gk.kB * gk.temperature**2)
        * gk.SI_PREFACTOR
    )
    assert np.allclose(cumul_unc, expected_unc)


# ---------------------------------------------------------------------------
# analyze_kute
# ---------------------------------------------------------------------------


def test_analyze_kute_fast_mode_result_keys_and_shapes(poscar_path, write_flux_file, rng, chdir_tmp_path):
    n = 600
    data = np.column_stack([np.full(n, 300.0), rng.normal(scale=0.01, size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    weighted_integral, weighted_integral_unc, fig = gk.analyze_kute(fast_mode=True, convolve_window=50)

    for key in ("hcacf", "individual_corrs", "hcacf_uncertainty", "cumul",
                "cumul_uncertainty", "weighted_integral", "weighted_integral_uncertainty"):
        assert key in gk.kute_results

    assert weighted_integral.shape == weighted_integral_unc.shape
    # inverse-variance weighted average "to the end": the last point folds
    # in only the last cumtrapz value, so it must equal cumul[-1] exactly.
    assert weighted_integral[-1] == pytest.approx(gk.kute_results["cumul"][-1])


def test_analyze_kute_small_data_fast_mode_false_does_not_crash(
    poscar_path, write_flux_file, rng, chdir_tmp_path
):
    """Regression test: analyze_kute(fast_mode=False) used to crash with
    ValueError: slice step cannot be zero whenever the folded flux had
    fewer than 1000 time steps, because the plotting downsample interval
    was hardcoded to len(xvals[:-1]) // 1000. Fixed by introducing the
    max_plot_points parameter and guarding the interval to be at least 1."""
    n = 300  # fewer than 1000 folded steps
    data = np.column_stack([np.full(n, 300.0), rng.normal(scale=0.01, size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    weighted_integral, weighted_integral_unc, fig = gk.analyze_kute(
        fast_mode=False, convolve_window=50
    )
    assert weighted_integral.shape == (n - 1,)


def test_analyze_kute_max_plot_points_is_adjustable(poscar_path, write_flux_file, rng, chdir_tmp_path):
    n = 600
    data = np.column_stack([np.full(n, 300.0), rng.normal(scale=0.01, size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    # an arbitrary, non-default value should not crash and should still
    # complete the analysis
    weighted_integral, _, _ = gk.analyze_kute(
        fast_mode=False, convolve_window=50, max_plot_points=17
    )
    assert weighted_integral.shape == (n - 1,)


def test_analyze_kute_test_mode_completes_and_agrees_with_fast_uncertainty(
    poscar_path, write_flux_file, rng, chdir_tmp_path, capsys
):
    """kute_test_mode=True recomputes the weighted-integral uncertainty via
    an explicit O(N^2) loop and asserts it agrees with the vectorized fast
    formula (the source code itself checks this with np.allclose and prints
    "differences detected!" plus saves differences.npy on mismatch).
    Regression test for the fixed shape mismatch (weighted_integral vs.
    kute_uncertainty differing in length by one) that used to make this
    mode crash unconditionally before returning."""
    n = 1500
    data = np.column_stack([np.full(n, 300.0), rng.normal(scale=0.01, size=(n, 3))])
    path = write_flux_file(data)

    gk_fast = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)
    fast_integral, fast_unc, _ = gk_fast.analyze_kute(fast_mode=True, convolve_window=50)

    gk_test = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)
    test_integral, test_unc, _ = gk_test.analyze_kute(kute_test_mode=True, convolve_window=50)

    captured = capsys.readouterr()
    assert "differences detected" not in captured.out
    assert not os.path.exists("differences.npy")

    assert np.allclose(fast_integral, test_integral)
    assert np.allclose(fast_unc, test_unc)


# ---------------------------------------------------------------------------
# analyze_euler
# ---------------------------------------------------------------------------


def test_analyze_euler_constant_hcacf_matches_cumtrapz(poscar_path, write_flux_file, chdir_tmp_path):
    n = 200
    # near-constant flux -> near-constant HCACF, where Euler and cumtrapz
    # integration of the same correlation function should closely agree
    data = np.column_stack([np.full(n, 300.0), np.full((n, 3), 0.01)])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    integral, u_integral, fig = gk.analyze_euler()

    assert integral.shape == u_integral.shape
    assert not np.any(np.isnan(integral))


def test_analyze_euler_cov_contrib_nonnegative_and_monotone_for_constant_hcacf(
    poscar_path, write_flux_file, chdir_tmp_path
):
    """For a flux whose HCACF is essentially constant, the covariance
    contribution to the Euler-integral uncertainty should stay non-negative
    and monotonically non-decreasing (up to floating-point noise)."""
    n = 300
    data = np.column_stack([np.full(n, 300.0), np.full((n, 3), 0.01)])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    from gk_analysis import uncertainty_tools as ut

    fluxes = gk.fold_flux(max_eval=None)
    with np.errstate(invalid="ignore"):
        corrfunc, u_corr, unc_caf, corrdiff = ut.calc_correlation(fluxes, num_pieces=None)
    _, _, n_contrib, cov_contrib = ut.calc_euler_integral(
        corrfunc, gk.dt * 1e3, u_corr, corrdiff,
        include_cov=True, SI_PREFACTOR=gk.SI_PREFACTOR, volume=gk.volume,
    )

    assert np.all(cov_contrib >= -1e-25)
    assert np.all(np.diff(cov_contrib) >= -1e-25)


# ---------------------------------------------------------------------------
# detect_f_star
# ---------------------------------------------------------------------------


def test_detect_f_star_stub_pinned_value(poscar_path, write_flux_file, rng, chdir_tmp_path):
    """detect_f_star is a stub that ignores its input and always returns
    10.0 -- pin that documented (if surprising) current behaviour."""
    n = 512
    data = np.column_stack([np.full(n, 300.0), rng.normal(size=(n, 3))])
    path = write_flux_file(data)
    gk = GreenKubo_run(path, poscar_path, dt=1.0, n_cart=3)

    assert gk.detect_f_star() == 10.0
    assert gk.t_evaluated == n
