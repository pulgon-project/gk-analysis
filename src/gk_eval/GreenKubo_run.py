#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

"""
This file contains tools to evaluate Green-Kubo simulations
"""


import numpy as np
import ase.io
import ase.units
import scipy.optimize
import scipy.signal
from scipy.integrate import cumulative_trapezoid
from scipy.special import polygamma, digamma
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import tqdm
import gk_eval.uncertainty_tools as ut


KAPPA_LABEL = "$\kappa$ / $\mathrm{W\,m^{-1}\,K^{-1}}$"
HFACF_LABEL = "HFACF / GW$^2\,$m$^{-4}$"
TIME_LABEL = "time / ns"


def calc_spectrum(
    fluxes: np.ndarray, delta_t: float, n_dim: int = 3, time_factor: float = 1.0
):
    """
    Calculate the spectra of given fluxes.

    Parameters:
    fluxes (numpy array): Array of fluxes.
    delta_t (float): Time interval between flux measurements.
    n_dim (int, optional): Number of dimensions. Defaults to 3.

    Returns:
    freqs (numpy array): Frequencies of the spectrum.
    ffts (numpy array): Fast Fourier Transforms.
    spectra (numpy array): Calculated spectra.
    """
    n_steps = fluxes.shape[1]
    print(np.shape(fluxes))
    n_dim = fluxes.shape[-1]
    ffts = np.fft.fft(fluxes, axis=1)
    freqs = np.fft.fftfreq(n_steps, delta_t)
    spectra = np.einsum("isk,jsk->ijs", ffts.conj(), ffts) / float(n_dim)
    spectra *= delta_t * time_factor / float(n_steps) * 1e3
    return freqs, ffts, spectra


def AIC(P, cepstral, var, AICc=False):
    """
    Calculate the Akaike Information Criterion (AIC) for a given set of cepstral coefficients.

    Parameters:
    P (int): Number of cepstral coefficients.
    cepstral (list): List of cepstral coefficients.
    var (float): Variance
    Returns:
    float: Calculated AIC.
    """
    N = len(cepstral)
    # print("N:",N)
    AIC = N / var * np.sum(cepstral[P : N // 2] ** 2) + 2 * P
    if AICc:
        # print(N,P)
        AIC += 2 * P * (P + 1) / (N - P - 1)
    return AIC


def AIC_weight(AICs):
    """
    Calculate the AIC weights for a given set of AIC values.

    Parameters:
    AICs (list): List of AIC values.

    Returns:
    list: List of AIC weights.
    """
    aic_opt = np.min(AICs)
    return np.exp(-0.5 * (AICs - aic_opt)) / np.sum(np.exp(-0.5 * (AICs - aic_opt)))


def expdecay(t, tau):
    return np.exp(-t / tau)


class GreenKubo_run:
    def __init__(
        self,
        flux_file,
        atoms_file,
        max_rows=None,
        take_every=1,
        dt=1.0,
        n_cart=3,
        col_ind=1,
        units="metal",
        temperature=None,
        nw=False,
        convective=False,
        independent_flux=False,
        fmod=1,
        hf_lammps=False,
    ):
        """
        Initializes a GreenKubo_run object.

        Args:
            flux_file (str): The file paths to read the flux data from.
            atoms_file (str): The file path to read the atoms data from.
            max_rows (int, optional): The maximum number of rows to read from the flux file. Defaults to None.
            take_every (int, optional): The interval to sample the flux data. Defaults to 1.
            dt (float, optional): The time step in femtoseconds. Defaults to 1.0.
            n_cart (int, optional): The number of Cartesian components of the flux. Defaults to 3.
            col_ind (int, optional): The column index of the first flux component data in the files. Defaults to 1.
            units (str, optional): The units of the flux. Can be "metal" or "ase". Defaults to "metal".
            temperature (float, optional): The temperature in Kelvin. If not specified, the temperature will be
                computed from the mean of the first column of the flux file. Defaults to None.
            nw (bool, optional): Flag to indicate if the simulation was performed on a nanowire. Defaults to False.
            convective (bool, list, optional): Flag to specify a file to read the convective component of the flux. Defaults to False.
            independent_flux (bool, optional): Flag to indicate if the flux from each file is to be treated as if originating from independent simulations. Defaults to False.
            fmod (int, optional): The number to modulo the flux length by. This is to cut away minor outliers Defaults to 1.
            hf_lammps (bool, optional): Flag to indicate if the heat flux was calculated as in the lammps tutorial.

        Returns:
            None
        """
        if type(flux_file) == str:
            self.flux_file = [flux_file]
        else:
            self.flux_file = flux_file
        self.atoms_file = atoms_file
        self.dt = dt * 1e-3 * take_every
        self.n_cart = n_cart
        self.convective = convective
        self.independent_flux = independent_flux
        self.fmod = fmod
        self.hf_lammps = hf_lammps
        self.atoms = ase.io.read(atoms_file)
        if nw:
            from gk_eval.struct.nanowire import Nanowire

            self.nw = Nanowire(self.atoms)
            # assuming the heat flux was already divided by this volume
            cellvol = self.atoms.get_volume()
            self.volume = self.nw.get_volume() * (cellvol / self.nw.get_volume()) ** 2
        else:
            self.volume = self.atoms.get_volume()
        self.temperature = temperature
        if convective:
            self.read_flux_components(
                self.convective,
                max_rows=max_rows,
                take_every=take_every,
                full_flux=True,
            )
        else:
            self.read_flux(
                self.flux_file,
                max_rows=max_rows,
                take_every=take_every,
                col_ind=col_ind,
            )

        if units == "metal":
            eV2J = 1.60218e-19
            self.kB = 1.38064852e-23
            A2m = 1.0e-10
            ps2s = 1.0e-12
            fs2s = 1.0e-15
            flux2SI = eV2J / A2m**2 / ps2s
            self.SI_PREFACTOR = flux2SI**2 * A2m**3 * fs2s
            self.HCACF_UNIT = flux2SI**2
            self.time_factor = 1.0
        elif units == "ase":
            self.SI_PREFACTOR = 1 / (ase.units.J / (ase.units.s * ase.units.m))
            self.kB = ase.units.kB
            self.time_factor = ase.units.fs
            self.HCACF_UNIT = ase.units.J**2

    def _read_flux(self, flux_file, max_rows=None, take_every=1):
        """
        Reads the heat flux from several files in an array.
        Parameters:
            flux_file (str): The file paths to read the flux data from.
            max_rows (int, optional): The maximum number of rows to read from the file. Defaults to None (all rows).
            take_every (int, optional): The interval to sample the flux data. Defaults to 1.
        """
        # with the way lammps handles restarts - we remove the first entry from every file that is not the first to prevent duplicates
        if self.independent_flux:
            skip = 0
        else:
            skip = 1

        if max_rows is not None:
            if isinstance(max_rows, int):
                max_rows = [max_rows] * len(flux_file)
            elif len(max_rows) != len(flux_file):
                print(
                    "WARNING: max_rows does not have the same length as flux_file, using first value for all files"
                )
                max_rows = [max_rows[0]] * len(flux_file)
        else:
            max_rows = [None] * len(flux_file)

        for fid, ff in enumerate(flux_file):
            if fid == 0:

                data = np.loadtxt(ff, skiprows=1, max_rows=max_rows[fid])
                cutnum = data.shape[0] % self.fmod
                if cutnum != 0:
                    data = data[:-cutnum]
                self.flens = [len(data)]
            else:
                ndata = np.loadtxt(ff, skiprows=1, max_rows=max_rows[fid])
                cutnum = ndata[skip:].shape[0] % self.fmod
                if cutnum != 0:
                    ndata = ndata[:-cutnum]
                data = np.vstack(
                    (
                        data,
                        ndata[skip:],
                    ),
                )
                self.flens.append(len(ndata[skip:]))

        flux = data[:, 1:]
        self.temp = data[:, 0]
        flux = -flux[::take_every]

        self.temp = self.temp[::take_every]
        if self.temperature is None:
            self.temperature = np.mean(self.temp)
        self.time_index = np.array(range(len(self.temp))) * self.dt

        return flux

    def read_flux(self, flux_file, max_rows=None, take_every=1, col_ind=1):
        """
        Read flux data from a file and store it in the object.

        Parameters:
            flux_file (str): The file path to read the flux data from.
            max_rows (int, optional): The maximum number of rows to read from the file. Defaults to None.
            take_every (int, optional): The interval to sample the data. Defaults to 1.
        """
        self.flux = self._read_flux(flux_file, max_rows=max_rows, take_every=take_every)
        self.flux = self.flux[:, (col_ind - 1) : (col_ind - 1 + self.n_cart)]

        if self.hf_lammps:
            self.flux = self.flux / self.volume

    def read_flux_components(
        self, flux_comp_file, max_rows=None, take_every=1, full_flux=False
    ):
        """
        Read flux components from a file and store it in the object.

        Parameters:
            flux_comp_file (str): The file containing flux components data.
            max_rows (int, optional): The maximum number of rows to read from the file. Defaults to None.
            take_every (int, optional): Take every nth row from the data. Defaults to 1.
            full_flux (bool, optional): Flag to calculate full flux or only internal flux. Defaults to False.
        """

        flux_data = self._read_flux(
            flux_comp_file, max_rows=max_rows, take_every=take_every
        )

        self.time_index = np.array(range(len(self.temp))) * self.dt
        self.flux_force = flux_data[:, 0 : (self.n_cart)] / self.volume
        self.flux_pot = (
            flux_data[:, (self.n_cart + 1) : (self.n_cart + self.n_cart)] / self.volume
        )
        self.flux_int = flux_data[
            :, (self.n_cart * 2) : (self.n_cart * 2 + self.n_cart)
        ]
        self.flux_conv = flux_data[
            :, (self.n_cart * 3) : (self.n_cart * 3 + self.n_cart)
        ]
        if full_flux:
            self.flux = self.flux_conv + self.flux_int
        else:
            self.flux = self.flux_int

        if self.hf_lammps:
            self.flux = self.flux / self.volume

    def create_plot_figures(
        self,
        pdf_name="cepstral_analysis.pdf",
        N_smooth=10,
        clf=True,
        manual_axs=None,
        indiv_figs=False,
    ):
        """
        Create plot figures for the cepstral analysis.

        Parameters:
            pdf_name (str): The name of the PDF file to save the figures in. Default is "cepstral_analysis.pdf".
            N_smooth (int): The number of points to smooth the spectra. Default is 10.
            clf (bool): Whether to clear the figure after saving. Default is True.

        Returns:
            matplotlib.figure.Figure: The figure object containing the plot.
        """
        if manual_axs is not None:
            f = plt.gcf()
            axs = manual_axs
        else:
            if not indiv_figs:
                if self.kappa_averaged is not None:
                    f, axs = plt.subplots(2, 3)
                else:
                    f, axs = plt.subplots(2, 3)
                axs = axs.ravel()
            else:
                axs = []
                f = []
                for i in range(6):
                    fig = plt.figure(i + 2)
                    axs.append(fig.gca())
                    f.append(fig)

        unit_factor = self.SI_PREFACTOR * self.prefactor / 2
        plt.sca(axs[0])
        num_2_plot = self.P_star * 4
        plt.plot(
            self.freqs[self.freqs >= 0],
            self.spectra[self.freqs >= 0] * unit_factor,
            label="raw spectrum",
        )
        interval_freqs = np.linspace(0, max(self.freqs), 50)
        plt.xlim([0, max(self.freqs)])
        plt.xlabel("frequency / THz")
        plt.ylabel("$\\bar{S}(f)$ / $\mathrm{Wm^{-1}K^{-1}}$")
        # plt.tight_layout()

        plt.sca(axs[1])
        plt.plot(range(len(self.cepstral)), self.cepstral)
        plt.xlim([0, self.P_star * 4])
        # plt.semilogy()
        plt.axvline(x=self.P_star, ls="--", c="r")
        plt.xlabel("P")
        plt.ylabel("cepstral coefficient")

        plt.sca(axs[2])
        plt.plot(range(len(self.aic)), self.aic)
        # plt.semilogy()
        plt.xlim([0, self.P_star * 4])
        plt.ylim([np.min(self.aic) * 0.9, np.min(self.aic) * 1.5])
        plt.xlabel("P")
        plt.ylabel("AIC$_c$")
        plt.axvline(self.P_star, color="red", ls="--")
        plt.axhline(np.min(self.aic), color="red", ls="--")

        plt.sca(axs[3])
        val_range = range(len(self.kappas[:num_2_plot]))
        plt.plot(val_range, self.kappas[:num_2_plot])  # , label="$\kappa$")
        plt.fill_between(
            val_range,
            self.kappas[:num_2_plot] - self.kappa_errs[:num_2_plot],
            self.kappas[:num_2_plot] + self.kappa_errs[:num_2_plot],
            alpha=0.3,
        )
        plt.axvline(x=self.P_star, ls="--", c="r", label="optimal P")
        if self.kappa_averaged is not None:
            plt.axhline(y=self.kappa_averaged, ls="--", c="g", label="model averaging")
            plt.axhspan(
                self.kappa_averaged - self.kappa_err_averaged,
                self.kappa_averaged + self.kappa_err_averaged,
                alpha=0.3,
                color="g",
            )
        plt.legend(fontsize=10)
        plt.xlabel("P")
        plt.ylabel(KAPPA_LABEL)

        if self.kappa_averaged is not None:
            plt.sca(axs[4])
            plt.plot(val_range, self.aic_weights[:num_2_plot], color="C2")
            plt.axvline(x=self.P_star, ls="--", c="r", label="optimal P")
            plt.xlabel("P")
            plt.ylabel("AIC$_c$ weight $w_i$")

        plt.sca(axs[5])
        plt.plot(
            self.freqs[self.freqs >= 0],
            self.log_spectra[self.freqs >= 0],
            label="log $S_k$",
        )
        to_test = [
            self.P_star * 4,
            self.P_star * 3,
            self.P_star * 2,
            self.P_star,
        ]  # , len(self.cepstral)]
        for P in to_test:
            if len(self.cepstral[:P]) > 1:
                log_spec_from_cepstral = np.fft.fft(
                    np.concatenate((self.cepstral[:P], self.cepstral[-P:])),
                )
                log_freqs = np.fft.fftfreq(
                    len(self.cepstral[:P]) * 2, self.resampled_delta_t
                )
                plt.plot(
                    log_freqs[log_freqs >= 0],
                    log_spec_from_cepstral[log_freqs >= 0],
                    label=f"P = {P}",
                )
        plt.legend(fontsize="x-small")
        plt.xlim([0, max(self.freqs)])
        plt.xlabel("frequency / THz")
        plt.ylabel("log $S_k$")

        plt.sca(axs[0])
        if len(self.cepstral[: self.P_star]) > 1:
            plt.plot(
                log_freqs[log_freqs >= 0],
                np.exp(log_spec_from_cepstral[log_freqs >= 0] - self.L_0) * unit_factor,
                color="r",
                label="optimal P",
            )
            plt.legend(fontsize=10)

        if pdf_name is not None:
            with matplotlib.backends.backend_pdf.PdfPages(pdf_name) as pdf:
                if not indiv_figs:
                    pdf.savefig(f)
                    if clf:
                        f.clf()
                else:
                    for fig in f:
                        pdf.savefig(fig)
                        if clf:
                            fig.clf()
        return f

    def detect_f_star(self, max_eval=None):
        """
        Detects the frequency at which the flux data should be resampled to compute the thermal conductivity.
        This is done based on the first prominent feature in the cepstra.

        Returns:
            float: The frequency at which the flux data should be resampled.
        """

        if max_eval is not None:
            max_eval = int(max_eval)
            flux = self.flux[:max_eval]
        else:
            flux = self.flux
        self.t_evaluated = len(flux)

        n_fluxes = flux.shape[1]

        fluxes = flux.reshape((flux.shape[0], -1, self.n_cart)).transpose((1, 0, 2))

        freqs, ffts, spectra = calc_spectrum(
            fluxes, self.dt, self.n_cart, time_factor=self.time_factor
        )

        print(np.shape(ffts))

        import pandas as pd

        filtered_spectrum = (
            pd.Series(spectra[0, 0, :])
            .rolling(1000, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )
        plt.plot(freqs, filtered_spectrum)

        return 10.0

    def cepstral_analysis(
        self,
        f_star=None,
        max_coeffs=1000,
        plot_results=False,
        max_eval=None,
        folds=None,
        AICc=True,
        model_averaging=False,
    ):
        """
        Performs a cepstral analysis on the given flux data to compute the thermal conductivity.

        Parameters:
            f_star (float, optional): The frequency at which to compute the cutoff for resampling the flux data. Defaults to 10.0.
            max_coeffs (int, optional): The maximum number of cepstral coefficients to consider. Defaults to 1000.
            plot_results (bool, optional): Whether to create a plot of the results. Defaults to False.
            max_eval (int, optional): The maximum time step to consider. Defaults to None (use all).
            folds (int, optional): The number of folds to use for the heat flux before analysis. Defaults to None.
            AICc (bool, optional): Whether to use the AICc criterion for selecting the number of cepstral coefficients. Defaults to True.
            model_averaging (bool, optional): Whether to use model averaging when selecting the number of cepstral coefficients. Defaults to False.
        Returns:
            tuple: A tuple containing the thermal conductivity and its error.
        """
        self.kappa_averaged = None
        if f_star is None:
            f_star = self.detect_f_star()
        # computing the cutoff to be used for the resample_poly function
        cutoff_steps = int(round(1.0 / (f_star * 2.0 * self.dt)))
        if cutoff_steps == 0:
            cutoff_steps = 1
        cutoff = 1.0 / (cutoff_steps * 2.0 * self.dt)
        print("Cutoff:")
        print("\t- Steps:", cutoff_steps)
        print("\t- Frequency:", cutoff, "THz")
        resampled_delta_t = cutoff_steps * self.dt
        self.resampled_delta_t = resampled_delta_t

        fluxes = self.fold_flux(folds=folds, max_eval=max_eval, mean_correction=False)

        # the initial function was written for the possibility of doing multi-component analysis. Here, we assume there is only 1.
        fluxes = fluxes.T.reshape(1, fluxes.shape[1], fluxes.shape[0])
        print(fluxes.shape)
        n_fluxes = fluxes.shape[-1]

        # exit()
        resampled_fluxes = scipy.signal.resample_poly(fluxes, 1.0, cutoff_steps, axis=1)

        freqs, ffts, spectra = calc_spectrum(
            resampled_fluxes,
            resampled_delta_t,
            self.n_cart,
            time_factor=self.time_factor,
        )
        print(np.shape(spectra))
        spectra = spectra[0, 0, :]
        self.freqs = freqs
        self.spectra = spectra

        # for now use single component
        n_components = 1
        ndf_chi = n_fluxes - n_components + 1
        print("number of fluxes:", n_fluxes)
        print("number of components:", n_components)
        var = polygamma(1, ndf_chi)
        self.L_0 = digamma(ndf_chi) - np.log(ndf_chi)

        # compute cepstrum
        log_spectra = np.log(spectra)
        cepstral = np.fft.ifft(log_spectra).real
        self.quefrency = np.fft.fftfreq(len(cepstral), resampled_delta_t)

        relevant_freqs = (self.freqs < cutoff) & (self.freqs >= 0)
        # try:
        #     self.expfit = curve_fit(
        #         expdecay, self.freqs[relevant_freqs], self.spectra[relevant_freqs], p0=[1.0]
        #     )
        # except:
        self.expfit = None

        # nums_cut = [2,3,4,5,6,7,8,9,10]
        # plt.plot(freqs, log_spectra)
        # for num_cut in nums_cut:
        #     sanity = np.fft.fft(cepstral[:-num_cut])
        #     log_freqs = np.fft.fftfreq(
        #         len(cepstral[:-num_cut]), self.resampled_delta_t
        #     )
        #     plt.plot(log_freqs, sanity)

        # plt.show()

        max_max_coeffs = len(cepstral) - 2

        if max_coeffs > max_max_coeffs:
            max_coeffs = max_max_coeffs
            print(
                "warning: max_coeffs larger than the number of cepstral coefficients, setting to",
                max_max_coeffs,
            )
        if max_coeffs == 0:
            # this is to prevent division by 0 errors
            cepstral = np.concatenate((cepstral, [0]))
            max_coeffs = 1
        aic = np.array([AIC(P, cepstral, var, AICc=AICc) for P in range(max_coeffs)])
        if len(aic) > 0:
            P_star = np.argmin(aic)
        else:
            P_star = 0
            aic = [1]
        if model_averaging:
            self.aic_weights = AIC_weight(aic)

        print("optimal number of cepstral coefficients:", P_star)
        self.cepstral = cepstral
        self.log_spectra = log_spectra
        self.aic = aic
        self.P_star = P_star

        self.prefactor = self.volume / (self.kB * self.temperature**2)
        num_2_eval = P_star * 4
        kappas = []
        kappa_errs = []
        for val in range(max_coeffs):
            kappa = (
                1
                / 2
                * np.exp(
                    cepstral[0] + 2 * np.sum(cepstral[1:(val)]) - self.L_0
                )  # use val here, since it leads to P*-1
                * self.SI_PREFACTOR
                * self.prefactor
            )
            if (4 * val - 2) < 0:
                kappa_err = 0
            else:
                kappa_err = (var * (4 * val - 2) / cepstral.size) ** 0.5 * kappa
            kappas.append(kappa)
            kappa_errs.append(kappa_err)

        self.kappas = np.array(kappas)
        self.kappa_errs = np.array(kappa_errs)
        if model_averaging:
            self.kappa_averaged = np.average(kappas, weights=self.aic_weights)
            self.kappa_err_averaged = (
                np.sum(
                    self.aic_weights
                    * (self.kappa_errs**2 + (kappas[P_star] - self.kappa_averaged) ** 2)
                )
                ** 0.5
            )

        self.kappa = kappas[P_star]
        self.kappa_err = kappa_errs[P_star]

        print("thermal conductivity:", self.kappa)
        print("thermal conductivity error:", self.kappa_err)

        if plot_results:
            self.create_plot_pdf()
        if model_averaging:
            return self.kappa_averaged, self.kappa_err_averaged
        else:
            return self.kappa, self.kappa_err

    def write_results(self, filename="kappa.txt", mode="w"):
        """
        Writes the results obtained from cepstral analysis to a file.

        Parameters:
            filename (str, optional): The name of the file to write the results to. Defaults to "kappa.txt".
            mode (str, optional): The mode to open the file in. Defaults to "w".
        """
        if self.kappas is None:
            self.cepstral_analysis()
        with open(filename, mode) as fp:
            np.savetxt(
                fp, np.c_[self.t_evaluated, self.kappa, self.kappa_err, self.f_star]
            )

    def fold_flux(
        self, folds=None, max_eval=None, mean_correction=False, eval_cut_init=True
    ):
        """
        Method to fold the full flux data in several pieces.
        Parameters:
            folds (int, optional): The number of folds to create. Defaults to None.
            max_eval (int, optional): The maximum number of evaluations to use. Defaults to None.
            mean_correction (bool, optional): Whether to apply a simple mean correction to the flux data (can lead to minor noise improvements). Not recommended. Defaults to False.
            eval_cut_init (bool, optional): Whether to apply the maximum time evaluation at the beginning. The difference is that the ACF might get cut short if this is not done. Defaults to True.
        """
        max_eval_folds = None

        flux = self.flux
        if max_eval is not None:
            if folds is None or folds == 1 or eval_cut_init:
                max_eval = int(max_eval)
                flux = self.flux[:max_eval]
            else:
                max_eval_folds = max_eval // (folds)

        n_steps = flux.shape[0]
        if mean_correction:
            flux = flux - np.mean(flux, axis=0)
            print(
                "Mean correction:",
                np.mean(flux, axis=0),
                np.max(flux, axis=0),
                np.min(flux, axis=0),
            )
        fluxes = flux.transpose((1, 0))
        # make sure that we do not get fluxes across the individual simulation boundaries
        # breakpoints = self.flens
        if folds is not None:
            folds *= self.n_cart

            num_folds = int(n_steps / folds)
            fluxes = fluxes[:, : folds * num_folds]
            self.t_evaluated = fluxes.shape[1]
            fluxes = fluxes.reshape((folds, -1))

            if max_eval_folds is not None:
                fluxes = fluxes[:, :max_eval_folds]
        else:
            folds = 1 * self.n_cart

        return fluxes

    def analyze_HCACF_integral(
        self,
        convolve_window=1000,
        mean_correction=False,
        folds=None,
        max_eval=None,
        newfig=True,
        raw_HCACF=False,
        fast_mode=False,
        plot_ACF=True,
        plot_kappa=True,
    ):
        """
        Analyzes the HCACF integral and visualizes the cumulative thermal conductivity.

        Some aspects of it are identical to the kute variant. TODO: create proper functions for this

        Parameters :
            convolve_window (int, optional): The window size for the convolution. Defaults to 1000.
            euler (bool, optional): Whether to use Euler's method for integration. Defaults to False.
        """
        fluxes = self.fold_flux(
            folds=folds, mean_correction=mean_correction, max_eval=max_eval
        )

        num_fluxes = fluxes.shape[0]
        len_segments = fluxes.shape[1]
        print(fluxes.shape)
        hcacf = np.zeros(fluxes.shape[1])

        for i in range(fluxes.shape[0]):
            corr = scipy.signal.correlate(
                fluxes[i, :], fluxes[i, :], "full", method="fft"
            )
            hcacf += corr[int(len(corr) / 2) :] / (
                np.flip(np.array(range(fluxes.shape[1])) + 1)
            )
        hcacf /= fluxes.shape[0]

        corrfunc_weight_factor = np.flip(np.array(range(len_segments)) + 1)
        hcacf_uncertainty = np.zeros(len_segments)
        for i in range(num_fluxes):
            corr = scipy.signal.correlate(
                fluxes[i, :] ** 2, fluxes[i, :] ** 2, "full", method="fft"
            )
            corr = corr[int(len(corr) / 2) :] / corrfunc_weight_factor
            hcacf_uncertainty += corr

        div_factor = num_fluxes * corrfunc_weight_factor - 1
        stddev_conv_factor = (num_fluxes * corrfunc_weight_factor) ** 0.5
        div_factor[div_factor == 0] = 1
        hcacf_uncertainty = (
            ((hcacf_uncertainty / num_fluxes) - hcacf**2) ** 0.5
            * 1
            / (div_factor) ** 0.5
        )

        cumul = (
            cumulative_trapezoid(hcacf, np.array(range(len(hcacf))))
            * self.dt
            * self.time_factor
            * 1e3
        )

        cumul = (
            cumul * self.volume / (self.kB * self.temperature**2) * self.SI_PREFACTOR
        )

        cumul_uncertainty = (
            self.dt
            * self.time_factor
            * 1e3
            / 2
            * np.cumsum(hcacf_uncertainty[:-1] ** 2 + hcacf_uncertainty[1:] ** 2) ** 0.5
        )

        cumul_uncertainty *= (
            self.volume / (self.kB * self.temperature**2) * self.SI_PREFACTOR
        )

        if newfig:
            fig = plt.figure()
        else:
            fig = plt.gcf()

        plt.xlabel(TIME_LABEL)
        plt.ylabel(KAPPA_LABEL, color="C0")
        # plt.xlim([0, self.dt * 1e6])
        ax1 = plt.gca()
        if plot_ACF:
            if plot_kappa:
                ax2 = plt.twinx(ax1)
            else:
                ax2 = ax1
            ax2.axhline(0, c="k", alpha=0.5)
            plt.ylabel(HFACF_LABEL, color="green")
            if raw_HCACF:
                ax2.plot(
                    np.array(range(len(hcacf))) * self.dt * 1e3 / 1e6,
                    hcacf * self.HCACF_UNIT / 1e18,
                    c="r",
                    alpha=0.5,
                    label="HFACF",
                    zorder=-5,
                )
            if convolve_window is not None:
                N = convolve_window
                xvals = np.array(range(len(hcacf)))[int(N / 2) : int(-N / 2) + 1]
                if len(xvals) > 0:
                    convolution = (
                        np.convolve(hcacf, np.ones(N) / N, mode="valid")
                        * self.HCACF_UNIT
                        / 1e18
                    )

                    ax2.plot(
                        xvals * self.dt * 1e3 / 1e6,
                        convolution,
                        c="green",
                        alpha=0.5,
                        label="HFACF (smooth)",
                    )

                    ax2.set_ylim([min(convolution), max(convolution)])
        if plot_kappa:
            ax1.plot(
                np.array(range(len(cumul))) * self.dt * 1e3 / 1e6,
                cumul,
                label="thermal cond.",
                zorder=6,
                color="C0",
            )
            if plot_ACF:
                ax1.set_zorder(ax2.get_zorder() + 1)
                ax1.set_frame_on(False)

        leghandles1, _ = ax1.get_legend_handles_labels()
        if plot_ACF and plot_kappa:
            leghandles2, _ = ax2.get_legend_handles_labels()
            plt.legend(
                handles=leghandles1 + leghandles2, loc="lower right", fontsize=10
            )
        else:
            plt.legend(handles=leghandles1, loc="lower right", fontsize=10)
        # plt.semilogx()
        plt.savefig("kappa_cumul_HCACF.png")
        return cumul, cumul_uncertainty, fig

    def analyze_kute(
        self,
        convolve_window=1000,
        mean_correction=False,
        folds=None,
        max_eval=None,
        raw_HCACF=False,
        fast_mode=False,
        kute_test_mode=False,
    ):
        """
        following paper: 10.1021/acs.jcim.4c02219
        does not support cross-correlation yet
        """

        if kute_test_mode:
            fast_mode = False

        kute_results = {}

        fluxes = self.fold_flux(
            folds=folds, mean_correction=mean_correction, max_eval=max_eval
        )

        num_fluxes = fluxes.shape[0]
        len_segments = fluxes.shape[1]

        # normal mean
        hcacf = np.zeros(len_segments)
        individual_corrs = []
        individual_cumuls = []

        corrfunc_weight_factor = np.flip(np.array(range(len_segments)) + 1)

        for i in range(num_fluxes):
            corr = scipy.signal.correlate(
                fluxes[i, :], fluxes[i, :], "full", method="fft"
            )
            individual_corr = corr[int(len(corr) / 2) :] / corrfunc_weight_factor
            individual_corrs.append(individual_corr)
            hcacf += individual_corr
            if not fast_mode:
                cumul = (
                    cumulative_trapezoid(
                        individual_corr, np.array(range(len(individual_corr)))
                    )
                    * self.dt
                    * self.time_factor
                    * 1e3
                )
                individual_cumuls.append(cumul)

        hcacf /= num_fluxes
        individual_corrs = np.array(individual_corrs)
        kute_results["hcacf"] = hcacf
        kute_results["individual_corrs"] = individual_corrs
        if not fast_mode:
            individual_cumuls = np.array(individual_cumuls)
            individual_cumuls *= (
                self.volume / (self.kB * self.temperature**2) * self.SI_PREFACTOR
            )
            kute_results["individual_cumuls"] = individual_cumuls
        # uncertainty from the squared fluxes
        hcacf_uncertainty = np.zeros(len_segments)
        for i in range(num_fluxes):
            corr = scipy.signal.correlate(
                fluxes[i, :] ** 2, fluxes[i, :] ** 2, "full", method="fft"
            )
            corr = corr[int(len(corr) / 2) :] / corrfunc_weight_factor
            hcacf_uncertainty += corr

        div_factor = num_fluxes * corrfunc_weight_factor - 1
        stddev_conv_factor = (num_fluxes * corrfunc_weight_factor) ** 0.5
        div_factor[div_factor == 0] = 1
        hcacf_uncertainty = (
            ((hcacf_uncertainty / num_fluxes) - hcacf**2) ** 0.5
            * 1
            / (div_factor) ** 0.5
        )
        kute_results["hcacf_uncertainty"] = hcacf_uncertainty
        # basic cumulative trapz integral
        cumul = (
            cumulative_trapezoid(hcacf, np.array(range(len(hcacf))))
            * self.dt
            * self.time_factor
            * 1e3
        )

        cumul = (
            cumul * self.volume / (self.kB * self.temperature**2) * self.SI_PREFACTOR
        )

        kute_results["cumul"] = cumul

        # uncertainty of the integral
        cumul_uncertainty = (
            self.dt
            * self.time_factor
            * 1e3
            / 2
            * np.cumsum(hcacf_uncertainty[:-1] ** 2 + hcacf_uncertainty[1:] ** 2) ** 0.5
        )

        cumul_uncertainty *= (
            self.volume / (self.kB * self.temperature**2) * self.SI_PREFACTOR
        )

        kute_results["cumul_uncertainty"] = cumul_uncertainty

        # compute the weighted thermal conductivity
        # this is an "average to the end" type of quantity
        flipped_weights = np.flip(cumul_uncertainty**-2)
        weighted_integral = np.flip(
            np.cumsum(np.flip(cumul) * flipped_weights) / np.cumsum(flipped_weights)
        )

        kute_results["weighted_integral"] = weighted_integral

        # uncertainty of the weighted integral
        running_average2 = np.flip(
            np.cumsum(np.flip(cumul**2) * flipped_weights) / np.cumsum(flipped_weights)
        )[:-1]
        contributions = np.flip(np.arange(len(cumul) - 1) + 2)
        kute_uncertainty = np.sqrt(
            (running_average2 - weighted_integral[:-1] ** 2) / contributions
        )
        if not fast_mode:
            # this was just to test the implementation, not recommended as it is very slow
            if kute_test_mode:
                weighted_integral_uncertainty = np.zeros(weighted_integral.shape)
                for i in tqdm.tqdm(range(len(weighted_integral))):
                    weighted_integral_uncertainty[i] = (
                        1
                        / (len(weighted_integral) - i)
                        * np.sum(
                            (weighted_integral[i] - cumul[i:]) ** 2
                            / (cumul_uncertainty[i:] ** 2)
                        )
                        / np.sum(cumul_uncertainty[i:] ** -2)
                    ) ** 0.5

                kute_uncertainty_test = np.append(
                    kute_uncertainty, weighted_integral_uncertainty[-1]
                )

                if not np.allclose(
                    weighted_integral_uncertainty, kute_uncertainty_test, rtol=1e-8
                ):
                    print("differences detected!")
                    plt.figure()
                    plt.scatter(weighted_integral_uncertainty, kute_uncertainty_test)
                    np.save(
                        "differences.npy",
                        np.array(
                            [
                                cumul,
                                cumul_uncertainty,
                                weighted_integral_uncertainty,
                                kute_uncertainty,
                            ]
                        ),
                    )
        weighted_integral_uncertainty = np.append(kute_uncertainty, 0)

        kute_results["weighted_integral_uncertainty"] = weighted_integral_uncertainty
        # plot stuff
        ns_factor = self.dt * 1e3 / 1e6
        xvals = np.array(range(len_segments)) * ns_factor
        hcacf *= self.HCACF_UNIT / 1e18
        hcacf_uncertainty *= self.HCACF_UNIT / 1e18
        individual_corrs *= self.HCACF_UNIT / 1e18
        fig, axs = plt.subplots(2, 2, sharex=True)

        plt.sca(axs[0, 0])
        # plt.title("individual")
        plt.ylabel(HFACF_LABEL)
        # if not fast_mode:
        #     for i in range(num_fluxes):
        #         plt.plot(xvals[:-1], individual_corrs[i])
        plt.plot(xvals, hcacf, color="k", label="HFACF")
        plt.fill_between(
            xvals,
            hcacf + hcacf_uncertainty,
            hcacf - hcacf_uncertainty,
            alpha=0.5,
            color="k",
            zorder=5,
        )
        plt.legend()

        plt.sca(axs[0, 1])
        # plt.title("averaged")
        # for i in range(num_fluxes):
        #     convolve_xvals = np.array(range(len(individual_corrs[i])))[
        #         int(convolve_window / 2) : int(-convolve_window / 2) + 1
        #     ] * ns_factor
        #     convolution = (
        #         np.convolve(individual_corrs[i], np.ones(convolve_window) / convolve_window, mode="valid")
        #     )
        #     plt.plot(convolve_xvals, convolution)

        convolve_xvals = (
            np.array(range(len(hcacf)))[
                int(convolve_window / 2) : int(-convolve_window / 2) + 1
            ]
            * ns_factor
        )
        convolution = np.convolve(
            hcacf, np.ones(convolve_window) / convolve_window, mode="valid"
        )
        if len(convolve_xvals) == len(convolution):
            plt.plot(convolve_xvals, convolution, color="k", label="smoothened HFACF")
        plt.legend()

        plt.sca(axs[1, 0])
        plt.xlabel(TIME_LABEL)
        plt.ylabel(KAPPA_LABEL)
        interval = int(len(xvals[:-1]) / 1000)
        if not fast_mode:
            for i in range(num_fluxes):
                plt.plot(xvals[:-1][::interval], individual_cumuls[i][::interval])
            mean_cumuls = np.mean(individual_cumuls, axis=0)
            ylims = [np.min(mean_cumuls), np.max(mean_cumuls)]
        else:
            ylims = None

        plt.plot(xvals[:-1], weighted_integral, color="k")
        plt.fill_between(
            xvals[:-1],
            weighted_integral + weighted_integral_uncertainty,
            weighted_integral - weighted_integral_uncertainty,
            alpha=0.5,
            color="k",
        )
        plt.ylim(ylims)

        plt.sca(axs[1, 1])
        plt.xlabel(TIME_LABEL)
        plt.plot(xvals[:-1], cumul, label="cumulative", color="b")
        plt.fill_between(
            xvals[:-1],
            cumul + cumul_uncertainty,
            cumul - cumul_uncertainty,
            alpha=0.5,
            color="b",
        )
        plt.plot(xvals[:-1], weighted_integral, label="KUTE", color="k")
        plt.fill_between(
            xvals[:-1],
            weighted_integral + weighted_integral_uncertainty,
            weighted_integral - weighted_integral_uncertainty,
            alpha=0.5,
            color="k",
        )
        if kute_test_mode:
            plt.fill_between(
                xvals[:-1],
                weighted_integral + kute_uncertainty,
                weighted_integral - kute_uncertainty,
                alpha=0.3,
                color="r",
            )
        if not fast_mode:
            plt.plot(xvals[:-1], mean_cumuls, "g--", alpha=0.5, label="mean")
        plt.ylim(ylims)
        plt.legend()
        plt.tight_layout()
        plt.savefig("kappa_cumul_HCACF_KUTE.png")
        if kute_test_mode:
            plt.show()
        self.kute_results = kute_results
        return weighted_integral, weighted_integral_uncertainty, fig

    def analyze_euler(
        self,
        convolve_window=1000,
        mean_correction=False,
        folds=None,
        max_eval=None,
        raw_HCACF=False,
        fast_mode=False,
        kute_test_mode=False,
    ):
        """
        Method to analyze the ACF integral using Euler's method with uncertainty estimation based on the covariance matrix
        """
        dt = self.dt * 1e3
        ns_factor = 1e6
        fluxes = self.fold_flux(folds=folds, max_eval=max_eval, mean_correction=False)

        # calculate correlation
        corrfunc, u_corr, unc_caf, corrdiff = ut.calc_correlation(
            fluxes, num_pieces=None
        )

        integral, u_integral, n_contrib, cov_contrib = ut.calc_euler_integral(
            corrfunc,
            dt,
            u_corr,
            corrdiff,
            include_cov=True,
            SI_PREFACTOR=self.SI_PREFACTOR,
            volume=self.volume,
        )

        # normal variant
        cumtrapz, u_cumtrapz = ut.calc_cumtrapz_integral(
            corrfunc, dt, u_corr, SI_PREFACTOR=self.SI_PREFACTOR, volume=self.volume
        )

        fig = plt.figure()
        plt.plot(np.arange(len(integral)) * dt / ns_factor, integral, label="Euler")
        plt.fill_between(
            np.arange(len(integral)) * dt / ns_factor,
            integral - u_integral,
            integral + u_integral,
            alpha=0.5,
            color="C0",
        )
        plt.fill_between(
            np.arange(len(integral)) * dt / ns_factor,
            integral - n_contrib,
            integral + n_contrib,
            alpha=0.3,
            color="C0",
        )
        plt.plot(np.arange(len(cumtrapz)) * dt / ns_factor, cumtrapz, label="cumtrapz")
        plt.fill_between(
            np.arange(len(cumtrapz)) * dt / ns_factor,
            cumtrapz - u_cumtrapz,
            cumtrapz + u_cumtrapz,
            alpha=0.5,
            color="C1",
        )
        plt.xlabel(TIME_LABEL)
        plt.ylabel(KAPPA_LABEL)
        plt.legend()
        plt.savefig(f"kappa_integral_euler.pdf")

        return integral, u_integral, fig
