"""Fit single Sersic 1-D profile"""

import numpy as np
from numpy.random import multivariate_normal

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

from scipy.optimize import curve_fit

import corner
import emcee

from kungpao.model.component import Sersic
from kungpao.model.parameters import ProfileParams

ORG = plt.get_cmap('OrRd')
ORG_2 = plt.get_cmap('YlOrRd')
BLU = plt.get_cmap('PuBu')

plt.rcParams['figure.dpi'] = 100.0
plt.rc('text', usetex=True)
rcParams.update({'axes.linewidth': 1.5})
rcParams.update({'xtick.direction': 'in'})
rcParams.update({'ytick.direction': 'in'})
rcParams.update({'xtick.minor.visible': 'True'})
rcParams.update({'ytick.minor.visible': 'True'})
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '8.0'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '4.0'})
rcParams.update({'xtick.minor.width': '1.5'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '8.0'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '4.0'})
rcParams.update({'ytick.minor.width': '1.5'})
rcParams.update({'axes.titlepad': '10.0'})
rcParams.update({'font.size': 25})


__all__ = ['lnlike_prof', 'norm_prof', 'config_params', 'ln_probability',
           'prof_curvefit', 'update_params', 'reinitialize_ball_covar',
           'emcee_fit_one_sersic', 'organize_results', 'plot_mcmc_corner',
           'plot_mcmc_trace', 'display_model_1d']


def lnlike_prof(theta, rad, rho, err, min_r=6.0, max_r=120.0):
    """LnLikelihood of the model profile.

    Parameters
    ----------
    theta: tuple or list
        A set of model parameters: [n, I0, Re]
    rad: list or 1-D array
        Radius array.
    rho: list or 1-D array
        Surface mass density profile.
    err: list or 1-D array
        Uncertainties of surface mass density profile.
    min_r: float, optional
        Minimal radii for fitting. Default=6.0
    max_r: float, optional
        Maximal radii for fitting. Default=120.0

    Returns
    -------
    The ln(likelihood) of the model profile.

    """
    params = list(theta)

    # Radial mask
    flag = (rad >= min_r) & (rad <= max_r)

    var = err ** 2
    chi2 = (Sersic(rad, params[0], params[1], params[2]) - rho) ** 2 / var
    chi2 = chi2[flag].sum()

    return -0.5 * (chi2 + np.log(2 * np.pi * var[flag].sum()))


def norm_prof(rad, rho, min_r=6.0, max_r=120):
    """Generate the normalized profile and get the fitting range for I0.

    Parameters
    ----------
    rad: list or 1-D array
        Radius array.
    rho: list or 1-D array
        Surface mass density profile.
    min_r: float, optional
        Minimal radii for fitting. Default=6.0
    max_r: float, optional
        Maximal radii for fitting. Default=120.0

    Returns
    -------
    norm: float
        Nomalization factor.
    i0_min: float
        Minimum fitting range for i0
    i0_max: float
        Maximum fitting range for i0

    """
    # Radial mask
    flag = (rad >= min_r) & (rad <= max_r)

    # Normalization factor
    norm = np.nanmedian(rho[flag])

    # Get the min and max range for the I0
    rho_min = np.percentile(rho[flag], 84)
    rho_max = np.percentile(rho, 99.5)

    return norm, rho_min / norm, rho_max / norm


def config_params(rad, rho, err, min_r=6.0, max_r=120.0,
                  param_config=None, err_inflation=1.0):
    """Config the model parameters.

    Parameters
    ----------
    rad: list or 1-D array
        Radius array.
    rho: list or 1-D array
        Surface mass density profile.
    err: list or 1-D array
        Uncertainties of surface mass density profile.
    min_r: float, optional
        Minimal radii for fitting. Default: 6.0
    max_r: float, optional
        Maximal radii for fitting. Default: 120.0
    param_config: dict, optional
        Dictionary for parameters. Default: None
    err_inflation: float, optional
        Factor to inflate the uncertainties. Default: 1.0

    Returns
    -------
    rho_norm: 1-D array
        Normalized surface density profile.
    err_norm: 1-D array
        Normalized uncertainty profile after error inflation.
    params: Parameters object
        Objects to deal with parameters and priors.

    """
    # Normalization factor, (min, max) of I0
    norm, i0_min, i0_max = norm_prof(rad, rho, min_r=min_r, max_r=max_r)

    # Get the normalized profiles
    rho_norm = np.asarray(rho).flatten() / norm
    err_norm = np.asarray(err).flatten() / norm * err_inflation

    if param_config is None:
        # Radial mask
        flag = (rad >= min_r) & (rad <= max_r)

        i0_ini = np.percentile(rho_norm, 90)
        i0_std = np.std(rho_norm[flag])

        # Define a dict for model parameters
        # Right now we just assume very simple priors:
        #    n: Sersic index, flat prior between 1.0 and 8.0; initial guess is 3.0
        #   I0: Central intensity, flat prior between the min and max range decided on normalized profile.
        #   Re: Effective radius, flat prior between the min and max fitting range; initial guess is 10.0
        # TODO: This should be replaced with more appropriate priors
        param_config = {
            'n': {
                'name': 'n', 'label':r'$n_{\rm Ser}$', 'ini': 3.0,
                'min': 1.0, 'max': 8.0, 'type': 'flat', 'sig': 1.0
            },
            'I0': {
                'name': 'I0', 'label':r'$I_{0}$', 'ini': i0_ini,
                'min': i0_min, 'max': i0_max, 'type': 'flat', 'sig': i0_std
            },
            'Re': {
                'name': 'Re', 'label':r'$R_{\rm e}$', 'ini': 10.,
                'min': min_r, 'max': max_r, 'type': 'flat', 'sig': 20.
            }
        }

    params = ProfileParams(param_config)

    return rho_norm, err_norm, params


def ln_probability(theta, params, rad, rho, err, nested=False):
    """Probability function to sample in an MCMC.

    Parameters
    ----------
    theta: tuple or list
        One set of model parameters.
    params: asap.Parameters object
        Object for model parameters.
    rad: list or 1-D array
        Radius array.
    rho: list or 1-D array
        Surface mass density profile.
    err: list or 1-D array
        Uncertainties of surface mass density profile.
    nested: bool, optional
        Using dynamical nested sampling or not. Default:False.

    Returns
    -------
        The ln(likelihood) of the model given the input parameters.

    """
    ln_prior = params.lnprior(theta, nested=nested)

    if not np.isfinite(ln_prior):
        return -np.inf

    return ln_prior + lnlike_prof(theta, rad, rho, err)


def prof_curvefit(func, rad, rho, err, params, min_r=6.0, max_r=120.0):
    """Get the best fit result using scipy.curvefit.

    Parameters
    ----------
    func: function
        Functional form of the profile to fit. e.g. Sersic.
    params: asap.Parameters object
        Object for model parameters.
    rad: list or 1-D array
        Radius array.
    rho: list or 1-D array
        Surface mass density profile.
    err: list or 1-D array
        Uncertainties of surface mass density profile.
    min_r: float, optional
        Minimal radii for fitting. Default=6.0
    max_r: float, optional
        Maximal radii for fitting. Default=120.0

    Returns
    -------
    best_curvefit: array
        Best-fit parameters from curvefit.
    cov_curvefit: array
        Covariance matrix of the parameters.

    """
    flag = (rad >= min_r) & (rad <= max_r)

    best_curvefit, cov_curvefit = curve_fit(
        func, rad[flag], rho[flag], p0=params.get_ini(),
        bounds=(params.get_low(), params.get_upp()),
        sigma=err[flag], absolute_sigma=False)

    return best_curvefit, cov_curvefit


def update_params(pbest, pcov, nsig=5.0):
    """Update the parameter constraint based on the best-fit result from curvefit.

    Parameters
    ----------
    pbest: 1-D array
        Best-fit parameters from curvefit.
    pcov: 2-D array
        Covariance matrix of the best-fit parameters from curvefit.
    nsig: float, optional
        N-sigma value to define the fitting range of parameters.
        Default: 5.

    Returns
    -------
    params_update: Parameters object
        Updated parameter constraints and priors.

    """
    # Convert the covariance matrix into error of parameters using just the
    # diagonal terms
    perr = np.sqrt(np.diag(pcov))

    param_config = {
        'n': {
            'name': 'n', 'label':r'$n_{\rm Ser}$', 'ini': pbest[0],
            'min': pbest[0] - nsig * perr[0], 'max': pbest[0] + nsig * perr[0],
            'type': 'flat', 'sig': perr[0]
        },
        'I0': {
            'name': 'I0', 'label':r'$I_{0}$', 'ini': pbest[1],
            'min': pbest[1] - nsig * perr[1], 'max': pbest[1] + nsig * perr[1],
            'type': 'flat', 'sig': perr[1]
        },
        'Re': {
            'name': 'Re', 'label':r'$R_{\rm e}$', 'ini': pbest[2],
            'min': pbest[2] - nsig * perr[2], 'max': pbest[2] + nsig * perr[2],
            'type': 'flat', 'sig': perr[2]
        }
    }

    return ProfileParams(param_config)


def reinitialize_ball_covar(pos, prob, threshold=50.0, center=None,
                            disp_floor=0.0, **extras):
    """Estimate the parameter covariance matrix from the positions of a
    fraction of the current ensemble and sample positions from the multivariate
    gaussian corresponding to that covariance matrix.  If ``center`` is not
    given the center will be the mean of the (fraction of) the ensemble.
    :param pos:
        The current positions of the ensemble, ndarray of shape (nwalkers, ndim)
    :param prob:
        The current probabilities of the ensemble, used to reject some fraction
        of walkers with lower probability (presumably stuck walkers).  ndarray
        of shape (nwalkers,)
    :param threshold: default 50.0
        Float in the range [0,100] giving the fraction of walkers to throw away
        based on their ``prob`` before estimating the covariance matrix.
    :param center: optional
        The center of the multivariate gaussian. If not given or ``None``, then
        the center will be estimated from the mean of the postions of the
        acceptable walkers.  ndarray of shape (ndim,)
    :param limits: optional
        An ndarray of shape (2, ndim) giving lower and upper limits for each
        parameter.  The newly generated values will be clipped to these limits.
        If the result consists only of the limit then a vector of small random
        numbers will be added to the result.
    :returns pnew:
        New positions for the sampler, ndarray of shape (nwalker, ndim)

    Notes
    -----
    This is from `prospect.fitting.ensemble` by Ben Johnson:
        https://github.com/bd-j/prospector/blob/master/prospect/fitting/ensemble.py
    """
    pos = np.atleast_2d(pos)
    nwalkers = prob.shape[0]
    good = prob > np.percentile(prob, threshold)

    if center is None:
        center = pos[good, :].mean(axis=0)

    Sigma = np.cov(pos[good, :].T)
    Sigma[np.diag_indices_from(Sigma)] += disp_floor**2
    pnew = resample_until_valid(multivariate_normal, center, Sigma,
                                nwalkers, **extras)

    return pnew


def clip_ball(pos, limits, disp):
    """Clip to limits.  If all samples below (above) limit, add (subtract) a
    uniform random number (scaled by ``disp``) to the limit.
    """
    npos = pos.shape[0]
    pos = np.clip(pos, limits[0], limits[1])

    for i, p in enumerate(pos.T):
        u = np.unique(p)
        if len(u) == 1:
            tiny = disp[i] * np.random.uniform(0, disp[i], npos)
            if u == limits[0, i]:
                pos[:, i] += tiny
            if u == limits[1, i]:
                pos[:, i] -= tiny

    return pos


def resample_until_valid(sampling_function, center, sigma, nwalkers,
                         limits=None, maxiter=1e3, prior_check=None):
    """Sample from the sampling function, with optional clipping to prior
    bounds and resampling in the case of parameter positions that are outside
    complicated custom priors.
    :param sampling_function:
        The sampling function to use, it must have the calling sequence
        ``sampling_function(center, sigma, size=size)``
    :param center:
        The center of the distribution
    :param sigma:
        Array describing the scatter of the distribution in each dimension.
        Can be two-dimensional, e.g. to describe a covariant multivariate
        normal (if the sampling function takes such a thing).
    :param nwalkers:
        The number of valid samples to produce.
    :param limits: (optional)
        Simple limits on the parameters, passed to ``clip_ball``.
    :param prior_check: (optional)
        An object that has a ``prior_product()`` method which returns the prior
        ln(probability) for a given parameter position.
    :param maxiter:
        Maximum number of iterations to try resampling before giving up and
        returning a set of parameter positions at least one of which is not
        within the prior.
    :returns pnew:
        New parameter positions, ndarray of shape (nwalkers, ndim)

    Notes
    -----
    This is from `prospect.fitting.ensemble` by Ben Johnson:
        https://github.com/bd-j/prospector/blob/master/prospect/fitting/ensemble.py
    """
    invalid = np.ones(nwalkers, dtype=bool)
    pnew = np.zeros([nwalkers, len(center)])

    for i in range(int(maxiter)):
        # replace invalid elements with new samples
        tmp = sampling_function(center, sigma, size=invalid.sum())
        pnew[invalid, :] = tmp
        if limits is not None:
            # clip to simple limits
            if sigma.ndim > 1:
                diag = np.diag(sigma)
            else:
                diag = sigma
            pnew = clip_ball(pnew, limits, diag)

        if prior_check is not None:
            # check the prior
            lnp = np.array([prior_check.lnprior(pos, nested=False) for pos in pnew])
            invalid = ~np.isfinite(lnp)
            if invalid.sum() == 0:
                # everything is valid, return
                return pnew
        else:
            # No prior check, return on first iteration
            return pnew
    # reached maxiter, return whatever exists so far
    print("initial position resampler hit ``maxiter``")

    return pnew


def emcee_fit_one_sersic(rad, rho, err, min_r=6.0, max_r=120.0, pool=None,
                         n_walkers=128, n_burnin=100, n_samples=100, output=None,
                         moves_burnin=None, moves_final=None, verbose=True):
    """Fit a single Sersic model to a 1-D profile.

    Parameters
    ----------

    Returns
    -------

    """
    # 3 parameters for a single Sersic model
    n_dim = 3

    # Decide the behaviour of the sampler
    if moves_burnin is None:
        moves_burnin = emcee.moves.DESnookerMove()
    if moves_final is None:
        moves_final = emcee.moves.StretchMove(a=4)

    # Normalize the input profile and uncertainty, decide the fitting range, and
    # setup the initial parameter ranges for fitting.
    rho_norm, err_norm, params = config_params(
        rad, rho, err, min_r=min_r, max_r=max_r)

    # Fit the Sersic profile using scipy.curvefit() to get the simple
    # best-fit parameters (pbest) and the associated covariance matrix (pcov)
    # The later can be used to estimate parameter errors.
    pbest, pcov = prof_curvefit(Sersic, rad, rho_norm, err_norm, params)
    if verbose:
        print("Best-fit Sersic parameters from curvefit:", pbest)
        print("Error of Sersic parameters from curvefit:", np.sqrt(np.diag(pcov)))

    # Update the parameter ranges based on the best-fit result
    params_update = update_params(pbest, pcov, nsig=5.0)

    # Initial postioins of each walker
    params_ini = params_update.sample(nsamples=n_walkers)

    # Parameter limits
    params_limits = np.array([params_update.low, params_update.upp])

    # Config the ensemble sampler
    args = [params_update, rad, rho_norm, err_norm]
    sampler_burnin = emcee.EnsembleSampler(
        n_walkers, n_dim, ln_probability, moves=moves_burnin, args=args, pool=pool)

    # Run burn-in step
    if verbose:
        print("# Running burn-in step...")
    burnin_results = sampler_burnin.run_mcmc(
        params_ini, n_burnin, store=True, progress=True)

    burnin = organize_results(
        burnin_results, sampler_burnin, n_dim, output=None,
        verbose=verbose, frac=0.1)

    # Find best walker position
    burnin_pos, burnin_prob, _ = burnin_results
    burnin_best = sampler_burnin.flatlnprobability.argmax()

    # Get the new initial positions for walkers
    initial_center = sampler_burnin.flatchain[burnin_best, :]

    new_ini = reinitialize_ball_covar(
        burnin_pos, burnin_prob, center=initial_center,
        limits=params_limits, disp_floor=0.1,
        prior_check=params_update, threshold=30)

    # Config the ensemble sampler
    if verbose:
        print("# Running final sampling step...")
    sampler_final = emcee.EnsembleSampler(
        n_walkers, n_dim, ln_probability, moves=moves_final, args=args, pool=pool)

    # Run the final sampling step
    sample_results = sampler_final.run_mcmc(
        new_ini, n_samples, store=True, progress=True)

    # Organize results
    results = organize_results(
        sample_results, sampler_final, n_dim, output=output,
        verbose=verbose, frac=0.1)

    # Add in the curve-fit results
    results['best_curvefit'] = pbest
    results['cov_curvefit'] = pcov
    results['err_curvefit'] = np.sqrt(np.diag(pcov))

    return results, burnin


def samples_stats(samples):
    """1D marginalized parameter constraints."""
    return map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
               zip(*np.percentile(samples, [16, 50, 84], axis=0)))


def organize_results(results, sampler, ndims, output=None, verbose=True, frac=0.1):
    """Organize the MCMC run results.

    Parameters
    ----------
    """
    position, lnprob, _ = results

    samples = sampler.chain[:, :, :].reshape((-1, ndims))
    chains = sampler.chain
    lnprob = sampler.lnprobability

    params_stats = samples_stats(samples)

    # Best parameter using the best log(prob)
    ind_1, ind_2 = np.unravel_index(np.argmax(lnprob, axis=None), lnprob.shape)
    best = chains[ind_2, ind_1, :]

    # Best parameters using the mean of the last few samples
    _, n_step, n_dim = chains.shape
    mean = np.nanmean(
        chains[:, -int(n_step * frac):, :].reshape([-1, n_dim]), axis=0)

    if output:
        np.savez(output,
                 samples=samples, lnprob=np.array(lnprob),
                 best=np.array(best), mean=np.asarray(mean),
                 chains=chains,
                 position=np.asarray(position),
                 acceptance=np.array(sampler.acceptance_fraction))

    if verbose:
        print("#------------------------------------------------------")
        print("#  Mean acceptance fraction",
              np.mean(sampler.acceptance_fraction))
        print("#------------------------------------------------------")
        print("#  Best ln(Probability): %11.5f" % np.max(lnprob))
        print(best)
        print("#------------------------------------------------------")
        print("#  Best parameters (mean):")
        print(mean)
        print("#------------------------------------------------------")
        for param_stats in params_stats:
            print(param_stats)
        print("#------------------------------------------------------")

    return {'samples': samples, 'lnprob': np.array(lnprob),
            'best': np.array(best), 'mean': np.asarray(mean),
            'chains': chains, 'position': np.asarray(position),
            'acceptance': np.array(sampler.acceptance_fraction)
           }


def plot_mcmc_corner(mcmc_samples, mcmc_labels, fontsize=26, labelsize=20, **corner_kwargs):
    """Corner plots for MCMC samples."""
    fig = corner.corner(
        mcmc_samples,
        bins=40, color=ORG(0.7),
        smooth=2, labels=mcmc_labels,
        label_kwargs={'fontsize': fontsize},
        quantiles=[0.16, 0.5, 0.84],
        levels=[0.16, 0.50, 0.84],
        plot_contours=True,
        fill_contours=True,
        show_titles=True,
        title_kwargs={"fontsize": labelsize},
        hist_kwargs={"histtype": 'stepfilled', "alpha": 0.5,
                     "edgecolor": "none"},
        use_math_text=True,
        **corner_kwargs
        )

    return fig


def plot_mcmc_trace(mcmc_chains, mcmc_labels, mcmc_best=None, figsize=None,
                    mcmc_burnin=None, burnin_alpha=0.2, trace_alpha=0.2):
    """Traceplot for MCMC results."""
    if figsize is None:
        if mcmc_burnin is not None:
            fig = plt.figure(figsize=(12, 15))
        else:
            fig = plt.figure(figsize=(10, 15))
    else:
        fig = plt.figure(figsize=figsize)

    fig.subplots_adjust(hspace=0.0, wspace=0.0, bottom=0.027, top=0.97,
                        left=0.06, right=0.94)

    # I want the plot of individual walkers to span 2 columns
    nparam = len(mcmc_labels)
    if mcmc_burnin is not None:
        gs = GridSpec(nparam, 5)
    else:
        gs = GridSpec(nparam, 3)

    if mcmc_best is not None:
        assert len(mcmc_best) == len(mcmc_labels)

    for ii, param in enumerate(mcmc_labels):
        # Getthe chains from burn-in process and the final sampling process
        param_chain = mcmc_chains[:, :, ii]
        if mcmc_burnin is not None:
            param_burnin = mcmc_burnin[:, :, ii]
            # Get the range of Y-axis
            y_min = np.min([np.min(param_chain), np.min(param_burnin)])
            y_max = np.max([np.max(param_chain), np.max(param_burnin)])
        else:
            y_min = np.min(param_chain)
            y_max = np.max(param_chain)

        # Maximum variance of the walkers
        max_var = max(np.var(param_chain[:, :], axis=1))

        # Trace plot
        if mcmc_burnin is None:
            ax1 = plt.subplot(gs[ii, :2])
        else:
            ax1 = plt.subplot(gs[ii, 2:4])
        ax1.yaxis.grid(linewidth=1.5, linestyle='--', alpha=0.5)

        for walker in param_chain:
            ax1.plot(np.arange(len(walker)), walker, alpha=trace_alpha,
                     drawstyle="steps", color=ORG_2(1.0 - np.var(walker) / max_var))

            if mcmc_burnin is None:
                ax1.set_ylabel(param, fontsize=28, labelpad=18, color='k')

            # Don't show ticks on the y-axis
            ax1.tick_params(labelleft=False)

        # For the plot on the bottom, add an x-axis label. Hide all others
        if ii != (nparam - 1):
            ax1.tick_params(labelbottom=False)
        else:
            for tick in ax1.xaxis.get_major_ticks():
                tick.label.set_fontsize(20)

        # Posterior histograms
        ax2 = plt.subplot(gs[ii, -1])
        ax2.grid(linewidth=1.5, linestyle='--', alpha=0.5)

        ax2.hist(np.ravel(param_chain[:, :]),
                 bins=np.linspace(ax1.get_ylim()[0], ax1.get_ylim()[1], 100),
                 orientation='horizontal', alpha=0.7, facecolor=ORG_2(0.9),
                 edgecolor="none")

        ax1.set_xlim(1, len(walker))
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(ax1.get_ylim())

        ax1.get_xaxis().tick_bottom()
        ax2.xaxis.set_visible(False)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        for tick in ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)

        if mcmc_best is not None:
            ax1.axhline(mcmc_best[ii], linestyle='--', linewidth=2,
                        c=BLU(1.0), alpha=0.8)
            ax2.axhline(mcmc_best[ii], linestyle='--', linewidth=2,
                        c=BLU(1.0), alpha=0.8)

        # Trace plot for burnin
        if mcmc_burnin is not None:
            param_burnin = mcmc_burnin[:, :, ii]
            ax3 = plt.subplot(gs[ii, :2])
            ax3.yaxis.grid(linewidth=1.5, linestyle='--', alpha=0.5)

            for walker in param_burnin:
                ax3.plot(np.arange(len(walker)), walker, drawstyle="steps",
                         color=BLU(np.var(walker) / max_var), alpha=burnin_alpha)

                ax3.set_ylabel(param, fontsize=25, labelpad=18, color='k')

                # Don't show ticks on the y-axis
                ax3.tick_params(labelleft=False)
                ax3.set_xlim(1, len(walker))
                ax3.set_ylim(y_min, y_max)
                ax3.get_xaxis().tick_bottom()

        # For the plot on the bottom, add an x-axis label. Hide all others
        if ii != (nparam - 1):
            ax1.xaxis.set_visible(False)
            if mcmc_burnin is not None:
                ax3.xaxis.set_visible(False)
        else:
            if mcmc_burnin is not None:
                for tick in ax3.xaxis.get_major_ticks():
                    tick.label.set_fontsize(20)

        if ii == 0:
            t = ax1.set_title(r"$\mathrm{Sampling}$", fontsize=28, color='k')
            t.set_y(1.01)
            t = ax2.set_title(r"$\mathrm{Posterior}$", fontsize=28, color='k')
            t.set_y(1.01)
            if mcmc_burnin is not None:
                t = ax3.set_title(r"$\mathrm{Burnin}$", fontsize=28, color='k')
                t.set_y(1.01)

    return fig


def visual_emcee(results, burnin=None, fontsize=20, alpha=0.3):
    """Visualize the emcee result."""
    params_label = [r'$n_{\rm Ser}$', r'$I_{0}$', r'$R_{\rm e}$']

    from matplotlib import rcParams
    rcParams.update({'font.size': 20})

    mod_corner = plot_mcmc_corner(
        results['chains'].reshape([-1, 3]), params_label,
        truths=results['best_curvefit'], truth_color='skyblue',
        fontsize=26, labelsize=22,
        **{'title_fmt': '.2f', 'ranges': None, 'plot_datapoints': False})

    mod_trace = plot_mcmc_trace(
        results['chains'], params_label,
        mcmc_best=results['best_curvefit'], mcmc_burnin=burnin['chains'],
        burnin_alpha=alpha, trace_alpha=alpha, figsize=(8, 6))

    return mod_corner, mod_trace


def display_model_1d(
    rad, rho, err, min_r=6.0, max_r=100.0, log_r=True,
    x_lim=None, y_lim=None, res_lim=None, info=None, info_pos=None,
    model=None, models=None, samples=None, model_label=r'$\rm Model$',
    models_label=None, normed=False):
    """Display 1-D profile """
    fig = plt.figure(constrained_layout=False, figsize=(7, 6))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0,
                        wspace=0.00, hspace=0.00)
    gs = GridSpec(3, 3, figure=fig)

    # Compare the profile
    ax1 = fig.add_subplot(gs[0:2, :])

    ax1.grid(linestyle='--', alpha=0.4, linewidth=2)
    ax1.set_xscale("log", nonposx='clip')
    ax1.set_yscale("log", nonposy='clip')
    ax1.axvline(min_r, linewidth=4.0, linestyle='--', c='k', alpha=0.5)
    ax1.axvline(max_r, linewidth=4.0, linestyle='--', c='k', alpha=0.5)

    # Data
    ax1.fill_between(rad, rho-err, rho+err, alpha=0.4, label='__no_label__')
    ax1.plot(rad, rho, linewidth=4.0, alpha=1.0, label=r'$\rm Data$')

    # Sample profiles
    if samples is not None:
        prof_3sig_low, prof_3sig_upp = np.percentile(samples, [0.3, 99.7], axis=0)
        ax1.fill_between(
            rad, prof_3sig_low, prof_3sig_upp, facecolor='grey', edgecolor='k',
            alpha=0.7, label=r'$3\-\sigma$')

        prof_low, prof_upp = np.percentile(samples, [16, 84], axis=0)
        ax1.fill_between(
            rad, prof_low, prof_upp, facecolor='orangered', edgecolor='orangered',
            alpha=0.7, label=r'$1\-\sigma$')

    # Model
    if model is not None:
        ax1.plot(rad, model, linestyle='--', linewidth=5.0, alpha=0.8,
                 label=model_label)

    # Plot multiple models
    if models is not None:
        if models_label is None:
            models_label = ['__no_label__'] * len(models)
        else:
            assert len(models) == len(models_label), "Wrong size of labels!"
        for mod, lab in zip(models, models_label):
            ax1.plot(rad, mod, linestyle='--', linewidth=4.0, alpha=0.8,
                     label=lab)

    if x_lim is not None:
        _ = ax1.set_xlim(x_lim)
    if y_lim is not None:
        _ = ax1.set_ylim(y_lim)

    if normed:
        _ = ax1.set_ylabel(r'$\rm Normalized\ Surface\ Intensity$', fontsize=26)
    else:
        _ = ax1.set_ylabel(r'$\rm Surface\ Intensity$', fontsize=26)

    ax1.legend(loc='best', fontsize=22)

    if info is not None:
        if info_pos is None:
            info_pos = [0.30, 0.22]
        else:
            info_pos = list(info_pos)
        ax1.text(info_pos[0], info_pos[1], info, transform=ax1.transAxes, fontsize=25)

    # Residual
    ax2 = fig.add_subplot(gs[2, :], sharex=ax1)

    ax2.grid(linestyle='--', alpha=0.4, linewidth=2)
    ax2.set_xscale("log", nonposx='clip')
    ax2.axvline(min_r, linewidth=4.0, linestyle='--', c='k', alpha=0.5)
    ax2.axvline(max_r, linewidth=4.0, linestyle='--', c='k', alpha=0.5)
    ax2.axhline(0.0, linewidth=4.0, linestyle='-', c='k', alpha=0.3)

    # Sample profiles
    if samples is not None:
        prof_3sig_low, prof_3sig_upp = np.percentile(samples, [0.3, 99.7], axis=0)
        ax2.fill_between(
            rad, (prof_3sig_low - rho) / rho, (prof_3sig_upp - rho) / rho,
            facecolor='grey', edgecolor='k',alpha=0.7, label=r'$3\-\sigma$')

        prof_low, prof_upp = np.percentile(samples, [16, 84], axis=0)
        ax2.fill_between(
            rad, (prof_low - rho) / rho, (prof_upp - rho) / rho,
            facecolor='orangered', edgecolor='orangered',
            alpha=0.7, label=r'$1\-\sigma$')

    # Model
    if model is not None:
        ax2.plot(rad, (model - rho) / rho, linewidth=5.0, alpha=0.7)

    # Multiple models
    if models is not None:
        for mod in models:
            ax2.plot(rad, (mod - rho) / rho, linestyle='--', linewidth=4.0, alpha=0.8)

    if res_lim is not None:
        _ = ax2.set_ylim(res_lim)

    _ = ax2.set_xlabel(r'$\rm Radius$', fontsize=30)
    _ = ax2.set_ylabel(r'$\rm Residual$', fontsize=26)