#!/usr/bin/env python3

"""Maximum likelihood estimation on a simulated Duffing oscillator."""


import argparse
import datetime
import functools
import importlib
import itertools
import json
import os
import pickle
import time

import hedeut
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from scipy import interpolate, optimize, signal, stats

import gvispe.stats
from gvispe import common, estimators, modeling, sde


class DuffingSDE:
    nx = 2
    """Total number of states."""

    nw = 1
    """Dimension of the driving Wiener process."""

    nq = 6
    """Number of classical (deterministic) parameters."""

    nu = 1
    """Number of exogenous inputs."""
    
    @hedeut.jax_vectorize_method(signature='(x),(u),(q)->(x)')
    def f(self, x, u, q):
        """Drift vector field."""
        x1, x2 = x
        a, b, d, g = q[:4]
        x1d = x2
        x2d = g*u - a*x1**3 - b*x1 - d*x2
        return jnp.r_[x1d, x2d]
    
    def G(self, q):
        """Diffusion matrix."""
        return jnp.array([[0], [jnp.exp(q[4])]])


class DiscretizedDuffing(sde.SO15ITScheme):
    ny = 1
    """Number of outputs."""

    def __init__(self, dt=None, df=np.inf):
        super().__init__(DuffingSDE(), dt)

        self.df = df
        """Degrees of freedom of Student's t distribution."""

    def h(self, x, u, q):
        return x[..., :1]

    @hedeut.jax_vectorize_method(signature='(y),(x),(u),(q)->()')
    def meas_logpdf(self, y, x, u, q):
        mean = self.h(x, u, q)[0]
        scale = jnp.exp(q[5])
        if np.isinf(self.df):
            return jsp.stats.norm.logpdf(y[0], mean, scale)
        else:
            return jsp.stats.t.logpdf(y[0], self.df, mean, scale)
    
    @hedeut.jax_vectorize_method(signature='(x),(q)->()')
    def prior_logpdf(self, x, q):
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, fromfile_prefix_chars='@'
    )
    parser.add_argument(
        '--jax-x64', dest='jax_x64', action=argparse.BooleanOptionalAction,
        help='Use double precision (64bits) in JAX.',
    )
    parser.add_argument(
        '--jax-platform', dest='jax_platform', choices=['cpu', 'gpu'],
        help='JAX platform (processing unit) to use',
    )
    parser.add_argument(
        '--reload', default=[], nargs='*',
        help='Modules to reload'
    )
    parser.add_argument(
        '--df', default=np.inf, type=float,
        help='Degrees of freedom of Student\'s t distribution.',
    )
    parser.add_argument(
        '--seedsim', default=0, type=int,
        help='Simulation random number generator seed.',
    )
    parser.add_argument(
        '--tf', default=50, type=float,
        help='Simulation final time.',
    )
    parser.add_argument(
        '--poutlier', default=0, type=float,
        help='Outlier probability.',
    )
    parser.add_argument(
        '--niter', default=10_000, type=int,
        help='Maximum number of optimization iterations.',
    )
    parser.add_argument(
        '--lrate0', default=1e-2, type=float,
        help='Stochastic optimization initial learning rate.',
    )
    parser.add_argument(
        '--transition_steps', default=1000, type=float,
        help='Learning rate "transition_steps" parameter.',
    )
    parser.add_argument(
        '--decay_rate', default=0.8, type=float,
        help='Learning rate "decay_rate" parameter.',
    )
    parser.add_argument(
        '--optimizer', choices=['stochastic', 'deterministic'],
        default='stochastic', help='What type of optimization to perform.'
    )
    parser.add_argument(
        '--problem', choices=['steady-state', 'transient', 'smoother-kernel'],
        default='steady-state', help='Problem parameterization.',
    )
    parser.add_argument(
        '--nwin', type=int, default=21, help='Convolution window size.',
    )
    parser.add_argument(
        '--pickleout', type=argparse.FileType('wb'), help='Pickle output file.',
    )
    parser.add_argument(
        '--txtout', type=argparse.FileType('w'), help='Text output file.',
    )
    args = parser.parse_args()

    # Apply JAX config options
    if args.jax_x64:
        jax.config.update("jax_enable_x64", True)
    if args.jax_platform:
        jax.config.update('jax_platform_name', args.jax_platform)

    # Reload modules
    for modname in args.reload:
        importlib.reload(globals()[modname])

    # Problem parameters
    ystd = 0.05
    outstd = 0.2
    poutlier = args.poutlier
    seedsim = args.seedsim
    x0 = np.r_[1, -1]
    q = np.r_[1, -1, 0.2, 0.3, np.log(0.1), np.log(ystd)]
    dtsim = 0.025
    dtest = 0.1
    Ts = 0.1
    tf = args.tf
    seedest = 10
    Nsamp = 2**9
    start = datetime.datetime.today()
    config = dict(
        seedsim=seedsim, x0=x0, q=q,
        dtsim=dtsim, dtest=dtest, Ts=Ts, tf=tf, 
        start=str(start),
    )
    
    # Create variables for simulation
    Nsim = round(tf / dtsim) + 1
    tsim = jnp.linspace(0, tf, Nsim)
    x0sim = jnp.array(x0, float)
    usim = jnp.cos(tsim)[:, None]
    qsim = jnp.array(q, float)
    key = jax.random.PRNGKey(seedsim)
    np.random.seed(seedsim)
    
    # Simulate the SDE
    model = DiscretizedDuffing(dtest, args.df)
    nx = model.nx
    key, subkey = jax.random.split(key)
    xsim = model.sim(x0sim, usim, qsim, tsim, subkey)
    
    # Simulate the SDE measurements
    y_sim_skip = round(Ts / dtsim)
    tmeas = tsim[::y_sim_skip]
    umeas = usim[::y_sim_skip]
    xmeas = xsim[::y_sim_skip]
    h = model.h(xmeas, umeas, qsim)
    key, subkey = jax.random.split(key)
    v = jax.random.normal(subkey, h.shape)
    key, subkey = jax.random.split(key)
    outlier = jax.random.uniform(subkey, h.shape) < poutlier
    y = h + v * np.where(outlier, outstd, ystd)
    
    # Generate initial guess
    lpf = signal.butter(2, 0.2, output='sos')
    yf = signal.sosfiltfilt(lpf, y, axis=0)
    dyf = np.diff(yf, axis=0, prepend=yf[:1]) / Ts
    d2yf = np.diff(dyf, axis=0, prepend=dyf[:1]) / Ts
    dyf[0] = dyf[1]
    d2yf[:2] = d2yf[2]
    xguess = interpolate.interp1d(tmeas, np.c_[yf, dyf], axis=0)
    qguess = jnp.array(q)

    # Prepare for estimation
    kmeas = jnp.s_[::round(Ts / dtest)]
    uest = usim[::round(dtest / dtsim)]
    Nest = round(tf / dtest) + 1
    test = np.linspace(0, tf, Nest)

    # Create data object
    data = estimators.Data(y, uest)

    # Create problem object and initial guess
    if args.problem == 'steady-state':
        p = estimators.SteadyState(model, -1)
        dec0 = p.Decision(
            q=jnp.zeros(model.nq),
            xbar=jnp.zeros((Nest, nx)),
            vech_log_S_cond=jnp.zeros(p.ntrilx),
            S_cross=jnp.zeros((model.nx, model.nx))
        )
    elif args.problem == 'transient':
        p = estimators.Transient(model, -1)
        dec0 = p.Decision(
            q=jnp.zeros(model.nq),
            xbar=jnp.zeros((Nest, nx)),
            vech_log_S_cond=jnp.zeros((Nest, p.ntrilx)),
            S_cross=jnp.zeros((Nest-1, model.nx, model.nx))
        )
    elif args.problem == 'smoother-kernel':
        p = estimators.SmootherKernel(model, args.nwin, -1)
        dec0 = p.Decision(
            q=jnp.zeros(model.nq),
            K=jnp.zeros((2, 2, args.nwin)),
            vech_log_S_cond=jnp.zeros(p.ntrilx),
            S_cross=jnp.zeros((model.nx, model.nx))
        )
    
    # Obtain integration coefficients
    pair_us_dev, xpair_w = gvispe.stats.ghcub(2, 2*nx)
    coeff = estimators.ExpectationCoeff(
        estimators.XCoeff(*gvispe.stats.ghcub(2, nx)),
        estimators.XPairCoeff(pair_us_dev[:, :nx], pair_us_dev[:, nx:], xpair_w)
    )

    # Run deterministic optimization
    if args.optimizer == 'deterministic':
        # Wrap optimization functions
        packer = p.packer(Nest)
        pack = lambda dec: packer.pack(*dec)
        unpack = lambda xvec: p.Decision(**packer.unpack(xvec))
        fix = dict(data=data, coeff=coeff, packer=packer)
        cost = p.fix_and_jit('elbo_packed', **fix)
        grad = p.fix_and_jit('elbo_grad_packed', **fix)
        hvp = p.fix_and_jit('elbo_hvp_packed', **fix)

        # Run jax JIT on optimization functions
        decvec0 = pack(dec0)
        cost(decvec0)
        grad(decvec0)
        hvp(decvec0, decvec0)

        # Save start of optimization
        opt_start = datetime.datetime.today()

        # Run optimization
        sol = optimize.minimize(
            cost, decvec0, jac=grad, hessp=hvp, method='trust-constr',
            options={'verbose': 2, 'maxiter': args.niter, 'disp': 1},
        )
        decopt = unpack(sol.x)
    if args.optimizer == 'stochastic':
        # Initialize stochastic optimization
        dec = dec0
        mincost = np.inf
        sched = optax.exponential_decay(
            init_value=args.lrate0, 
            transition_steps=args.transition_steps,
            decay_rate=args.decay_rate,
        )
        optimizer = optax.adam(sched)
        opt_state = optimizer.init(dec)
        np.random.seed(seedest)
    
        # JIT functions
        cost = p.fix_and_jit('elbo')
        grad = p.fix_and_jit('elbo_grad')
        hvp = p.fix_and_jit('elbo_hvp_packed')

        # Save start of optimization
        opt_start = datetime.datetime.today()

        # Perform optimization
        for i in range(args.niter):
            # Sample deviation
            rqmc_pair = stats.qmc.MultivariateNormalQMC(
                np.zeros(2 * model.nx), seed=seedest
            )
            rqmc_x = stats.qmc.MultivariateNormalQMC(
                np.zeros(model.nx), seed=seedest+1,
            )
            e_pair = rqmc_pair.random(Nsamp)
            e_x = rqmc_x.random(Nsamp)
            coeff = estimators.ExpectationCoeff(
                estimators.XCoeff(e_x, 1 / Nsamp),
                estimators.XPairCoeff(e_pair[:, :nx], e_pair[:, nx:], 1/Nsamp)
            )

            # Calculate cost and gradient
            cost_i = cost(dec, data, coeff)
            grad_i = grad(dec, data, coeff)
            mincost = min(mincost, cost_i)

            fooc = p.Decision(*[jnp.sum(v**2) ** 0.5 for v in grad_i])
            print(
                f'{i}', f'c={cost_i:1.5e}', f'm={mincost:1.5e}',
                f'{fooc.xbar=:1.2e}', f'{fooc.q=:1.2e}', f'{sched(i)=:1.1e}',
                *[f'q[{i}]={v:1.3e}' for i,v in enumerate(dec.q)], 
                sep='\t'
            )

            if any(jnp.any(~jnp.isfinite(v)) for v in grad_i):
                break

            updates, opt_state = optimizer.update(grad_i, opt_state)
            dec = optax.apply_updates(dec, updates)

        # Final decision variables of stochastic optimization
        decopt = dec

    # Compute time taken
    endtime = datetime.datetime.today()
    opt_time = endtime - opt_start
    time_elapsed = endtime - start

    # Calculate optimal problem variables
    vopt = p.problem_variables(decopt, data)

    # Save results
    if args.txtout is not None:
        qerr = np.abs(decopt.q - qsim)
        skip = args.nwin//2
        xwin = xmeas[skip:-skip] if args.problem == "smoother-kernel" else xmeas
        xerr = np.abs(xwin - vopt.xbar).mean(0)
        print(
            args.problem, args.nwin, args.tf, Nest, opt_time.total_seconds(),
            *qerr, *xerr,
            file=args.txtout
        )
    if args.pickleout is not None:
        pickleout = args.pickleout

        # Delete arguments that cannot be pickled
        del args.pickleout
        del args.txtout
        
        outdata = dict(
            sim=dict(t=tsim, x=xsim, y=y, u=usim, q=qsim),
            args=vars(args),
            config=config,
            decopt=decopt._asdict(),
            vopt=vopt._asdict(),
            twin=test[args.nwin//2:-args.nwin//2],
            time_elapsed=time_elapsed.total_seconds(),
            opt_time=opt_time.total_seconds(),
        )
        pickle.dump(outdata, pickleout, protocol=-1)
        pickleout.close()
