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

from gk_eval.GreenKubo_run import GreenKubo_run, KAPPA_LABEL, TIME_LABEL
import matplotlib.pyplot as plt
import matplotlib
import sys
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description="Compute kappa from the thermal flux")
    parser.add_argument(
        "run_file",
        type=str,
        nargs="*",
        help="files containing the thermal flux. If multiples are given, they are treated as restarts and appended after each other.",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=20.0,
        help="cutoff linear frequency in THz",
    )
    parser.add_argument(
        "--manual_cutoff",
        action="store_true",
        default=False,
        help="adds a popup to analyze the cutoff in detail",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=None,
        help="set a fixed temperature",
    )
    parser.add_argument(
        "-m",
        "--max_time",
        type=int,
        default=None,
        nargs="*",
        help="maximum number of time steps to be evaluated",
    )
    parser.add_argument(
        "-p",
        "--poscar",
        type=str,
        default="POSCAR",
        help="poscar file to read the volume",
    )
    parser.add_argument(
        "-nw",
        dest="nw",
        action="store_true",
        default=False,
        help="specify if the volume of a nanowire is to be used. This will assume that the full cell volume was used during the flux calculation and correct it accordingly.",
    )
    parser.add_argument(
        "--convective",
        type=str,
        default=None,
        nargs="*",
        help="Include convective flux in the analysis. Provide the name of the components file.",
    )
    parser.add_argument(
        "--num_dim",
        type=int,
        default=3,
        help="Number of dimensions",
    )
    parser.add_argument(
        "--col_ind",
        type=int,
        default=1,
        help="column index for the first flux to be evaluated in the file",
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=20,
        help="Number of tests to perform",
    )
    parser.add_argument(
        "--num_c_tests",
        type=int,
        default=None,
        help="Number of cutoff tests to perform. If set to None, same as num_tests",
    )
    parser.add_argument(
        "--take_every",
        type=int,
        default=1,
        help="Only take every this many time steps for the analysis.",
    )
    parser.add_argument(
        "--units",
        type=str,
        default="metal",
        help="units of the heat flux. Allowed unit systems: metal, ase.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Fold the trajectory in this many segments.",
    )
    parser.add_argument(
        "--max_coeffs",
        type=int,
        default=1000,
        help="Maximum number of cepstral coefficients.",
    )
    parser.add_argument(
        "--independent",
        dest="independent",
        action="store_true",
        default=False,
        help="if set, the runs given to the script are not treated as restarts. The folding parameter will be set accordingly.",
    )
    parser.add_argument(
        "--noshow",
        dest="noshow",
        action="store_true",
        default=False,
        help="do not show the figures",
    )
    parser.add_argument(
        "--nocepstral",
        dest="nocepstral",
        action="store_true",
        default=False,
        help="do not use cepstral analysis",
    )

    parser.add_argument(
        "--nohcacf",
        dest="nohcacf",
        action="store_true",
        default=False,
        help="do not use the integral of the HCACF",
    )
    parser.add_argument(
        "--kute",
        dest="kute",
        action="store_true",
        default=False,
        help="if set, use KUTE to analyze the direct integral. (see 10.1021/acs.jcim.4c02219)",
    )
    parser.add_argument(
        "--euler",
        dest="euler",
        action="store_true",
        default=False,
        help="if set, use Euler integration method to obtain the cumulative integral",
    )
    parser.add_argument(
        "--fast",
        dest="fast",
        action="store_true",
        default=False,
        help="fast mode, faster evaluations but figures contain fewer details",
    )
    parser.add_argument(
        "--hfacf_extract",
        dest="hfacf_extract",
        type=float,
        nargs="*",
        default=[0.5],
        help="at which fraction of the spectrum is the thermal conductivity obtained",
    )
    parser.add_argument(
        "--fmod",
        dest="fmod",
        type=int,
        default=1,
        help="modulo flux length to use for analysis",
    )
    parser.add_argument(
        "--model_averaging",
        dest="model_averaging",
        action="store_true",
        default=False,
        help="use model averaging for the selection of the thermal conductivity",
    )
    parser.add_argument(
        "--raw_HCACF",
        dest="raw_HCACF",
        action="store_true",
        default=False,
        help="show the raw HCACF",
    )
    parser.add_argument(
        "--indiv_figs",
        dest="indiv_figs",
        action="store_true",
        default=False,
        help="store cepstral plots as individual figures",
    )
    parser.add_argument(
        "--full_fold",
        dest="full_fold",
        action="store_true",
        default=False,
        help="do foldings for equally long correlation lengths. Example: for 10 foldings, perform this many number of tests with 1, 2, 3, 4, 5, 6, 7, 8,9 and 10 foldings while always adding the same amount of data.",
    )
    parser.add_argument(
        "--hf_lammps",
        dest="hf_lammps",
        action="store_true",
        default=False,
        help="use definition of heat flux as in lammps tutorial (volume)",
    )
    parser.add_argument(
        "-d", "--delta_t", type=float, default=1, help="time step in fs"
    )
    args = parser.parse_args()

    afp = open("p_cmnd.log", "a")
    afp.write(" ".join(sys.argv) + "\n")
    afp.close()
    if args.num_c_tests is not None:
        num_c_tests = args.num_c_tests
    else:
        num_c_tests = args.num_tests

    file_suffix = ""
    if args.folds is not None:
        file_suffix += f"_folds"
    if args.full_fold:
        file_suffix = "_full" + file_suffix
    part_fsuff = file_suffix
    if args.folds is not None:
        file_suffix += f"_{args.folds}"

    fname = args.run_file
    struct_file = args.poscar
    gkrun = GreenKubo_run(
        fname,
        struct_file,
        dt=args.delta_t,
        units=args.units,
        max_rows=args.max_time,
        n_cart=args.num_dim,
        col_ind=args.col_ind,
        temperature=args.temperature,
        nw=args.nw,
        take_every=args.take_every,
        convective=args.convective,
        independent_flux=args.independent,
        fmod=args.fmod,
        hf_lammps=args.hf_lammps,
    )
    print("read run")
    n_steps = len(gkrun.temp)
    num_tests = args.num_tests
    time_conv = np.linspace(0, n_steps, num_tests + 1, dtype=int)[1:]

    if args.manual_cutoff:
        kappa, kappa_err = gkrun.cepstral_analysis(f_star=None)

    if args.folds is None:
        folds = [1]
    else:
        folds = args.folds

    if not args.nocepstral:
        plt.figure(1)
        for fold in folds:
            pstars = []
            kappas = []
            kappas_0 = []
            kappa_errs = []
            with matplotlib.backends.backend_pdf.PdfPages(
                f"cepstral_analysis_time_convergence{part_fsuff}_{fold}.pdf"
            ) as pdf:
                if args.full_fold:
                    # always set the num_tests to the number of foldings
                    corresponding_time_conv = np.linspace(
                        0, n_steps, fold + 1, dtype=int
                    )[1:]
                    selected_time_conv = corresponding_time_conv
                else:
                    selected_time_conv = time_conv

                for vid, val in enumerate(selected_time_conv):
                    if args.full_fold:
                        new_fold = int(fold * (vid + 1) / len(corresponding_time_conv))
                    else:
                        new_fold = fold
                    print(val, new_fold)
                    kappa, kappa_err = gkrun.cepstral_analysis(
                        f_star=args.cutoff,
                        max_eval=int(val),
                        folds=new_fold,
                        max_coeffs=args.max_coeffs,
                        AICc=True,
                        model_averaging=args.model_averaging,
                    )
                    aic = gkrun.aic
                    pstar = gkrun.P_star
                    pstars.append(pstar)
                    unit_factor = gkrun.SI_PREFACTOR * gkrun.prefactor / 2
                    kappas_0.append(gkrun.spectra[0] * unit_factor)
                    kappas.append(kappa)
                    kappa_errs.append(kappa_err)

                    fig = gkrun.create_plot_figures(
                        None, clf=True, indiv_figs=args.indiv_figs
                    )
                    if args.indiv_figs:
                        for f in fig:
                            f.suptitle(f"n_step: {val}")
                            f.tight_layout()
                            pdf.savefig(f)
                            plt.close(f)
                    else:
                        fig.suptitle(f"n_step: {val}")
                        fig.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
            if args.full_fold:
                labelstr = f"corr. len: {selected_time_conv[0] * args.delta_t / 1e6} ns"
            else:
                labelstr = f"folding {fold}"
            plt.figure(1)
            plt.plot(
                selected_time_conv * args.delta_t / 1e6,
                kappas,
                label=labelstr,
                marker="o",
            )
            plt.fill_between(
                selected_time_conv * args.delta_t / 1e6,
                np.array(kappas) - np.array(kappa_errs),
                np.array(kappas) + np.array(kappa_errs),
                alpha=0.3,
            )
            plt.figure(3)
            plt.plot(
                selected_time_conv * args.delta_t / 1e6,
                pstars,
                label=labelstr,
                marker="o",
            )

            np.savetxt(
                f"kappa_time_convergence{part_fsuff}_{fold}.txt",
                np.c_[selected_time_conv * args.delta_t, kappas, kappa_errs],
            )

            plt.figure(8)
            plt.plot(
                selected_time_conv * args.delta_t / 1e6,
                kappas_0,
                label=labelstr,
                marker="o",
            )

        plt.figure(1)
        if args.folds is not None:
            plt.legend()
        plt.xlabel(TIME_LABEL)
        plt.ylabel(KAPPA_LABEL)
        plt.savefig(f"kappa_time_convergence{file_suffix}.pdf")
        plt.figure(3)
        if args.folds is not None:
            plt.legend()
        plt.xlabel(TIME_LABEL)
        plt.ylabel("$P^*$")
        plt.savefig(f"opt_cepstral_time_convergence{file_suffix}.pdf")

        # the zero value - no filtering at all
        plt.figure(8)
        if args.folds is not None:
            plt.legend()
        plt.xlabel(TIME_LABEL)
        plt.ylabel(KAPPA_LABEL)
        plt.savefig(f"kappa_0_time_convergence{file_suffix}.pdf")
        

        cut_conv = np.linspace(0, args.cutoff, num_c_tests + 1)[1:]
        print(cut_conv)
        for fold in folds:
            kappas = []
            kappa_errs = []
            pstars = []
            with matplotlib.backends.backend_pdf.PdfPages(
                f"cepstral_analysis_cutoff_convergence_folds{fold}.pdf"
            ) as pdf:
                for val in cut_conv:
                    print(val)
                    kappa, kappa_err = gkrun.cepstral_analysis(
                        f_star=val,
                        folds=fold,
                        max_coeffs=args.max_coeffs,
                        AICc=True,
                        model_averaging=args.model_averaging,
                    )
                    kappas.append(kappa)
                    kappa_errs.append(kappa_err)
                    pstar = gkrun.P_star
                    pstars.append(pstar)
                    fig = gkrun.create_plot_figures(
                        None, clf=True, indiv_figs=args.indiv_figs
                    )
                    if args.indiv_figs:
                        for f in fig:
                            f.suptitle(f"F_star: {val}")
                            f.tight_layout()
                            pdf.savefig(f)
                            plt.close(f)
                    else:
                        fig.suptitle(f"F_star: {val}")
                        fig.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
            plt.figure(2)
            plt.plot(cut_conv, kappas, label=f"folding {fold}")
            plt.fill_between(
                cut_conv,
                np.array(kappas) - np.array(kappa_errs),
                np.array(kappas) + np.array(kappa_errs),
                alpha=0.3,
            )

            plt.figure(4)
            plt.plot(cut_conv, pstars, label=f"folding {fold}")

            np.savetxt(
                f"kappa_cutoff_convergence_{fold}.txt",
                np.c_[cut_conv, kappas, kappa_errs],
            )

        plt.figure(2)
        if args.folds is not None:
            plt.legend()
        plt.xlabel("F* / THz")
        plt.ylabel(KAPPA_LABEL)
        plt.savefig(f"kappa_cutoff_convergence{file_suffix}.pdf")

        plt.figure(4)
        if args.folds is not None:
            plt.legend()
        plt.xlabel("F* / THz")
        plt.ylabel("$P^*$")
        plt.savefig(f"opt_cepstral_cutoff_convergence{file_suffix}.pdf")

    # also perform the conventional integral

    if args.kute:
        gkevalfunc = gkrun.analyze_kute
    elif args.euler:
        gkevalfunc = gkrun.analyze_euler
    else:
        gkevalfunc = gkrun.analyze_HCACF_integral

    kappas = {}
    kappa_errs = {}
    if not args.nohcacf:
        hcacf_extract_values = args.hfacf_extract
        for fold in folds:
            kappas[fold] = {}
            kappa_errs[fold] = {}
            for hcacf_val in hcacf_extract_values:
                kappas[fold][hcacf_val] = []
                kappa_errs[fold][hcacf_val] = []

            if not args.full_fold:
                with matplotlib.backends.backend_pdf.PdfPages(
                    f"HCACF_analysis_time_convergence_folds{fold}.pdf"
                ) as pdf:
                    for val in time_conv:
                        print(val)
                        kappa, kappa_err, fig = gkevalfunc(
                            folds=fold,
                            max_eval=int(val),
                            mean_correction=False,
                            fast_mode=args.fast,
                            raw_HCACF=args.raw_HCACF,
                        )
                        fig.suptitle(f"n_step: {val}")
                        plt.tight_layout()
                        pdf.savefig(fig)
                        # plt.clf()
                        plt.close(fig)
                        for hcacf_val in hcacf_extract_values:
                            kappas[fold][hcacf_val].append(
                                kappa[int(len(kappa) * hcacf_val)]
                            )
                            kappa_errs[fold][hcacf_val].append(
                                kappa_err[int(len(kappa) * hcacf_val)]
                            )

                    for hcacf_val in hcacf_extract_values:
                        np.savetxt(
                            f"kappa_time_convergence_HCACF_ex_{hcacf_val}_{fold}.txt",
                            np.c_[
                                time_conv * args.delta_t,
                                kappas[fold][hcacf_val],
                                kappa_errs[fold][hcacf_val],
                            ],
                        )
            elif args.full_fold:
                # this is the more reasonable approach for the direct evaluation for multiple independent simulations
                with matplotlib.backends.backend_pdf.PdfPages(
                    f"HCACF_analysis_time_convergence_full_folds{fold}.pdf"
                ) as pdf:
                    corresponding_time_conv = np.linspace(
                        0, n_steps, fold + 1, dtype=int
                    )[1:]
                    for vid, val in enumerate(corresponding_time_conv):
                        new_fold = int(fold * (vid + 1) / len(corresponding_time_conv))
                        print(val, new_fold)
                        kappa, kappa_err, fig = gkevalfunc(
                            folds=new_fold,
                            max_eval=int(val),
                            mean_correction=False,
                            fast_mode=args.fast,
                        )
                        fig.suptitle(f"n_step: {val}")
                        plt.tight_layout()
                        pdf.savefig(fig)
                        # plt.clf()
                        plt.close(fig)

                        for hcacf_val in hcacf_extract_values:
                            kappa_errs[fold][hcacf_val].append(
                                kappa_err[int(len(kappa) * hcacf_val)]
                            )
                            kappas[fold][hcacf_val].append(
                                kappa[int(len(kappa) * hcacf_val)]
                            )
                    for hcacf_val in hcacf_extract_values:
                        np.savetxt(
                            f"kappa_time_convergence_HCACF_full_ex_{hcacf_val}_{fold}.txt",
                            np.c_[
                                corresponding_time_conv * args.delta_t,
                                kappas[fold][hcacf_val],
                                kappa_errs[fold][hcacf_val],
                            ],
                        )
        conv_fig = plt.figure()
        for hcacf_val in hcacf_extract_values:
            for fold in folds:

                if args.full_fold:
                    corresponding_time_conv = np.linspace(0, n_steps, fold + 1, dtype=int)[
                        1:
                    ]
                else:
                    corresponding_time_conv = time_conv
                plt.plot(
                    corresponding_time_conv * args.delta_t / 1e6,
                    kappas[fold][hcacf_val],
                    label=f"corr. time: {corresponding_time_conv[0] * args.delta_t / 1e6} ns",
                )
            if args.folds is not None:
                plt.legend()
            plt.xlabel(TIME_LABEL)
            plt.ylabel(KAPPA_LABEL)
            if not args.full_fold:
                plt.savefig(
                    f"kappa_time_convergence_HCACF_ex_{hcacf_val}{file_suffix}.pdf"
                )
            elif args.full_fold:
                plt.savefig(
                    f"kappa_time_convergence_HCACF_fullex_{hcacf_val}{file_suffix}.pdf"
                )
            if hcacf_val != hcacf_extract_values[-1]:
                conv_fig.clf()

    if not args.noshow:
        plt.show()

    plt.close("all")


if __name__ == "__main__":
    main()
