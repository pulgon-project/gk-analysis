# GK_eval_cepstral

A package to evaluate Green-Kubo simulations including cepstral and uncertainty analysis. The units are set up for the thermal conductivity. However, there is no reason to assume that the implemented approaches would not work for other transport quantities.

## Evaluate Green-Kubo integrals using various statistical approaches

Implemented approaches:
 - Direct integration
 - Cepstral analysis (see [Ercole et al. 2017](http://dx.doi.org/10.1038/s41598-017-15843-2) and [Ercole et al. 2023](https://linkinghub.elsevier.com/retrieve/pii/S0010465522001898))
 - KUTE (see [Otero-Lema et al. 2025](https://doi.org/10.1021/acs.jcim.4c02219))
 - Uncertainty analysis using Euler integrals

systematic_GK_analysis provides a general tool to evaluate Green-Kubo integral and creates a large quantity of figures to allow the user to judge convergence of the simulations.

## Installation

```bash
pip install .
```

## Usage as command line interface

The most straightforward way to use the package is by utilizing the "systematic_GK_analysis" script, which requires the input of heat flux files in a N+1-column format, where N is the number of Cartesian directions for which the flux was computed. The first column should be the temperature. The first line of the files will be considered a header and skipped. Internally, the system is considered to be isotropic. If an anisotropic analysis is desired, separate heat flux files can be used and the script should be run for each direction individually. Alternatively, the `--col_ind` option can be used in combination with reducing `--num_dim` to specify the precise column index to be used. The units of the heat flux should correspond to the "metal" units used in LAMMPS. If the actual heat flux $\mathbf{J}$ is provided, `--hf_lammps` should be used and the option is to be ommitted if $V\mathbf{J}$ is provided.

The script will automatically produce time convergence figures while also outputting many further figures detailling the individual steps of the evaluation procedure. The user is strongly encouraged to study these in detail when any unexpected results arise.

A simple usage example would be:

```bash
systematic_GK_analysis heat_flux.dat -d 5 -p POSCAR_SUPERCELL -c 0.5
```

Here, the ASE-readable structure provided with `-p` should be the simulation supercell and is used to evaluate the volume of the structure (for lower-dimensional structures, you can specify `--nw` to compute a volume from Van der Waals spheres to get an estimate without the vacuum) and `-d` is the time step in fs. The script will then compute the thermal conductivity using cepstral analysis and a direct extraction approach. For the latter, a fixed position of the autocorrelation is set as a fraction of the correlation time which can be customized using the option `--hfacf_extract` with a value between 0 and 1. Time convergence figures will be created for both approaches for `--num_tests` different simulation times, defaulting to 20. The same convergence test is performed for the cutoff frequency of the power spectra in the cepstral analysis up to a user-specified cutoff `-c` given in THz.

If multiple independent runs or restarts are to be analyzed, they should be provided as a list of individual files:

```bash
systematic_GK_analysis heat_flux_1.dat heat_flux_2.dat heat_flux_3.dat -d 5 -p POSCAR_SUPERCELL
```

To improve the analysis of the time convergence `--folds` can be used to average over pieces of the autocorrelation function. Several numbers can be given in one call for a comparison figure.

For independent runs it can be useful to use `--independent`. The only difference is that if this setting is not given, the first step of every further run will be considered the restarted value. Additionally, for independent runs, `--folds` should always be used in combination with the setting `--full_fold` (This will force the correlation times in the time convergence plots to be the same. The full folding will only be applied to the full simulation time.) to avoid discontinuities in the heat flux trajectory.

To enhance the direct approach with uncertainty metrics either use the `--kute` or `--euler` approaches (see paper for details). For KUTE, it can be useful to specify the `--fast` option. These evaluation approaches will take significantly longer and can be quite memory heavy.

To only do either the direct of cepstral analysis, the options `--nocepstral` or `--nohcacf` can be useful.

There are a few more implemented niche features that might be useful. Use `--help` for more information or use the API directly. 


## Usage as API

The primary class to analyse Green-Kubo calculations is provided in the class `GreenKubo_run` and can be used to obtain the same functionality as the CLI (and more) to evaluate Green-Kubo simulations. An example to create an object and for some useful methods to call:


```python
from gk_eval.GreenKubo_run import GreenKubo_run
fold = len(heat_flux_path_list)
gkrun = GreenKubo_run(
    heat_flux_path_list,
    struct_file_name,
    dt=5,
    units="metal",
    max_rows=max_time,
    n_cart=1,
    nw=True,
)

# cepstral analysis
kappa, kappa_err = gkrun.cepstral_analysis(
    f_star=0.5,  # frequency cutoff in THz
    folds=fold,
    AICc=True,
    model_averaging=True,
)
gkrun.create_plot_figures()

# the following provide an array of thermal conductivity values. You are meant to choose a value along the HFACF based on uncertainty metrics or multiple runs
# basic direct evaluation
kappa, kappa_err, fig = gkrun.analyze_HCACF_integral(folds=fold)

# analysis using the KUTE one-shot approach
kappa_kute, kappa_unc_kute, fig_kute = gkrun.analyze_kute(folds=fold)
results_dict = gkrun.kute_results

# analysis with the simplified Euler integral and uncertainties including the covariance matrix
kappa_euler, kappa_unc_euler, fig_euler = gkrun.analyze_euler(folds=fold)


```

Please consult the method headers for more information regarding the features.