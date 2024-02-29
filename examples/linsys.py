#!/usr/bin/env python3

"""Maximum likelihood estimation of a simulated linear--Gaussian system."""


import argparse
import datetime
import functools
import importlib
import itertools
import json
import os
import pathlib
import time

import hedeut
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy.io
import scipy.linalg
from scipy import interpolate, optimize, signal, sparse, stats

import gvispe.stats
from gvispe import common, estimators, modeling, sde


class Model(modeling.LinearGaussianModel):
    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny, False)
        self.q_packer = hedeut.Packer(
            A=(self.nx, self.nx),
            B=(self.nx, self.nu),
            D=(self.ny, self.nu),
            vech_log_sQ=(self.ntrilx,),
            vech_log_sR=(self.ntrily,),
        )

        self.nq = self.q_packer.size
        """Number of classical (deterministic) parameters."""

    def C(self, q):
        return jnp.eye(self.ny, self.nx)


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
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Random number generator seed.',
    )
    parser.add_argument(
        '--N', default=1000, type=int,
        help='Number of time samples.',
    )
    parser.add_argument(
        '--nx', default=2, type=int,
        help='Number of model states.',
    )
    parser.add_argument(
        '--ny', default=2, type=int,
        help='Number of model outputs.',
    )
    parser.add_argument(
        '--nu', default=1, type=int,
        help='Number of model exogenous inputs.',
    )
    parser.add_argument(
        '--problem', choices=['steady-state', 'transient', 'smoother-kernel'],
        default='steady-state', help='Problem parameterization.',
    )
    parser.add_argument(
        '--nwin', type=int, default=21, help='Convolution window size.',
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

    # Seed the RNG
    np.random.seed(args.seed)

    # Define model dimensions
    nx = args.nx
    nu = args.nu
    ny = args.ny
    N = args.N

    # Generate the model
    A = np.random.rand(nx, nx)
    L, V = np.linalg.eig(A)
    complex_L = np.iscomplex(L).nonzero()[0]
    stab_L = np.random.rand(nx) * np.exp(1j * np.angle(L))
    stab_L[complex_L[1::2]] = stab_L[complex_L[::2]].conj() 
    L = np.where(np.abs(L) < 1, L, stab_L)
    A = np.real(V @ np.diag(L) @ np.linalg.inv(V))
    B = np.random.randn(nx, nu)
    C = np.eye(ny, nx)
    D = np.zeros((ny, nu))
    sQ = np.diag(np.repeat(0.1, nx))
    sR = np.diag(np.repeat(0.15, ny))
    Q = sQ @ sQ.T
    R = sR @ sR.T

    # Simulate
    t = np.arange(N)
    x = np.zeros((N, nx))
    y = np.zeros((N, ny))
    u = np.where(np.random.rand(N, nu) > 0.5, -1.0, 1.0)
    w = np.random.randn(N, nx)
    v = np.random.randn(N, ny)
    for k in range(N-1):
        x[k+1] = A @ x[k] + B @ u[k] + sQ @ w[k]
    y = x @ C.T + v @ sR.T + u @ D.T

    # Create model and data objects
    model = Model(nx, nu, ny)
    data = estimators.Data(y, u)

    # Create problem object and initial guess
    if args.problem == 'steady-state':
        p = estimators.SteadyState(model, -1)
        dec0 = p.Decision(
            q=jnp.zeros(model.nq),
            xbar=(y @ np.random.randn(ny, nx) + u @ np.random.randn(nu, nx)),
            vech_log_S_cond=jnp.zeros(p.ntrilx),
            S_cross=jnp.zeros((model.nx, model.nx))
        )
    elif args.problem == 'transient':
        p = estimators.Transient(model, -1)
        dec0 = p.Decision(
            q=jnp.zeros(model.nq),
            xbar=(y @ np.random.randn(ny, nx) + u @ np.random.randn(nu, nx)),
            vech_log_S_cond=jnp.zeros((N, p.ntrilx)),
            S_cross=jnp.zeros((N-1, model.nx, model.nx))
        )
    elif args.problem == 'smoother-kernel':
        p = estimators.SmootherKernel(model, args.nwin, -1)
        K0 = np.zeros((nx, ny+nu, args.nwin))
        K0[..., args.nwin//2] = np.random.randn(nx, ny+nu)
        dec0 = p.Decision(
            q=jnp.zeros(model.nq),
            K=jnp.array(K0),
            vech_log_S_cond=jnp.zeros(p.ntrilx),
            S_cross=jnp.zeros((model.nx, model.nx))
        )

    # Obtain integration coefficients
    pair_us_dev, xpair_w = gvispe.stats.sigmapts(2*nx)
    coeff = estimators.ExpectationCoeff(
        estimators.XCoeff(*gvispe.stats.sigmapts(nx)),
        estimators.XPairCoeff(pair_us_dev[:, :nx], pair_us_dev[:, nx:], xpair_w)
    )

    # Wrap optimization functions
    packer = p.packer(N)
    pack = lambda dec: packer.pack(*dec)
    unpack = lambda xvec: p.Decision(**packer.unpack(xvec))
    fix = dict(data=data, coeff=coeff, packer=packer)
    cost = p.fix_and_jit('elbo_packed', **fix)
    grad = p.fix_and_jit('elbo_grad_packed', **fix)
    hvp = p.fix_and_jit('elbo_hvp_packed', **fix)

    # Run optimization
    sol = optimize.minimize(
        cost, pack(dec0), jac=grad, hessp=hvp, method='trust-constr',
        options={'verbose': 2, 'maxiter': args.maxiter, 'disp': 1},
    )

    # Retrieve solution
    decopt = unpack(sol.x)
    pvarsopt = p.problem_variables(decopt, data)
    qopt = decopt.q
    Aopt = model.A(qopt)
    Bopt = model.B(qopt)
    Copt = model.C(qopt)
    Dopt = model.D(qopt)
    Ropt = model.R(qopt)
    Qopt = model.Q(qopt)
    xopt = pvarsopt.xbar
    S_cond = pvarsopt.S_cond
    S_cross = pvarsopt.S_cross
    S = pvarsopt.S

    # Get the Riccatti solution
    Ppred = scipy.linalg.solve_discrete_are(Aopt.T, Copt.T, Qopt, Ropt)
    K = Ppred @ C.T @ np.linalg.inv(Ropt + Copt @ Ppred @ Copt.T)
    Pcorr = (np.identity(model.nx) - K @ Copt) @ Ppred
