#!/usr/bin/env python3

"""Maximum likelihood estimation of ATTAS short period."""


import argparse
import datetime
import functools
import importlib
import itertools
import json
import os
import pathlib
import time

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import hedeut as utils
import numpy as np
import optax
import scipy.linalg
import scipy.io
from scipy import interpolate
from scipy import optimize
from scipy import signal
from scipy import stats

from gvispe import sde
import gvispe.stats


class LinearShortPeriodSDE:
    """Discretized linear short-period model."""
    nx = 2
    """Total number of states."""

    nw = 2
    """Dimension of the driving Wiener process."""

    np = 0
    """Number of Bayesian (probabilistically uncertain) parameters."""

    nq = 9
    """Number of classical (deterministic) parameters."""
    
    @utils.jax_vectorize_method(signature='(x),(u),(p),(q)->(x)')
    def f(self, x, u, p, q):
        """Drift vector field."""
        alpha, pitchrate = x
        de, = u
        Za, Zq, Zde, Ma, Mq, Mde, deeq = q[:7]
        adot = Za * alpha + (1 + Zq) * pitchrate + Zde * (de - deeq)
        qdot = Ma * alpha + Mq * pitchrate + Mde * (de - deeq)
        return jnp.r_[adot, qdot]
    
    @utils.jax_vectorize_method(signature='(p),(q)->(x)')
    def G_diag(self, p, q):
        """Diffusion matrix."""
        G_logdiag = q[7:9]
        return jnp.array(jnp.exp(G_logdiag))


class DiscretizedLinearShortPeriod(sde.EulerScheme):    
    nq = 12
    """Number of classical (deterministic) parameters."""

    def __init__(self, dt=None):
        super().__init__(LinearShortPeriodSDE(), dt)

    @utils.jax_vectorize_method(signature='(x),(u),(p),(q)->(y)')
    def h(self, x, u, p, q):
        qbias = q[11]
        return x + jnp.r_[0, qbias]

    def Sv(self, p, q):
        logsigmay = q[9:11]
        return jnp.diag(jnp.exp(logsigmay))

    @utils.jax_vectorize_method(signature='(x),(p),(q)->()')
    def prior_logpdf(self, x, p, q):
        return 0
    
    @utils.jax_vectorize_method(signature='(y),(x),(u),(p),(q)->()')
    def meas_logpdf(self, y, x, u, p, q):
        mu = self.h(x, u, p, q)
        logsigmay = q[9:11]
        return jsp.stats.norm.logpdf(y, mu, jnp.exp(logsigmay)).sum()


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
        '--maxiter', default=1_000, type=int,
        help='Maximum mumber of iterations of the optimization.',
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

    # Load data
    currdir = pathlib.Path(__file__).parent
    data = scipy.io.loadmat(currdir / 'data' / 'fAttasElv1.mat')['fAttasElv1']
    t = data[:, 0]
    y = data[:, [12, 7]] * np.pi/180
    u = data[:, [21]] * np.pi / 180
    dt, = np.diff(t[:2])
    
    # Create model and problem
    model = DiscretizedLinearShortPeriod(dt)
    p = gvispe.SteadyStateProblem(model, u, y)
    
    dec0 = gvispe.SteadyStateProblem.Decision(
        xbar=y, 
        p=jnp.zeros(model.np),
        q=jnp.zeros(model.nq),
        S_xnext_ltlogd_elem=jnp.zeros(p.ntrilx),
        S_xnext_x=jnp.zeros((model.nx, model.nx))
    )

    # Prepare for optimization
    estacked, w = gvispe.stats.ghcub(2, 2*model.nx)
    e = estacked[:, :model.nx], estacked[:, model.nx:]
    unpack = p.packer.unpack
    pack = p.packer.pack
    cost = jax.jit(lambda x: -p.elbo(unpack(x), e, w))
    grad = jax.jit(lambda x: -pack(**p.elbo_grad(unpack(x), e, w)))
    hvp = jax.jit(lambda x, v: -pack(**p.elbo_hvp(unpack(x), unpack(v), e, w)))
    sol = optimize.minimize(
        cost, pack(**dec0), jac=grad, hessp=hvp, method='trust-constr',
        options={'verbose': 2, 'maxiter': args.maxiter, 'xtol': 1e-10},
    )
    decopt = unpack(sol.x)
    pvarsopt = p.problem_variables(decopt)
    xopt = decopt['xbar']
    popt = decopt['p']
    qopt = decopt['q']
    sPcorropt = pvarsopt['S_x']

    # Get Riccatti solution
    A = jax.jacobian(model._trans_mean)(xopt[0], u[0], popt, qopt, dt)
    C = jax.jacobian(model.h)(xopt[0], u[0], popt, qopt)
    Q = np.diag(model.sde.G_diag(popt, qopt)**2)
    Sv = model.Sv(popt, qopt)
    R = Sv @ Sv.T
    Ppred = scipy.linalg.solve_discrete_are(A.T, C.T, Q, R)
    K = Ppred @ C.T @ np.linalg.inv(R + C @ Ppred @ C.T)
    Pcorr = (np.identity(model.nx) - K @ C) @ Ppred