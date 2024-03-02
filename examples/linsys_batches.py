#!/usr/bin/env python3

"""Identification of a simulated linear--Gaussian system using mini-batches."""


import argparse
import datetime
import functools
import importlib
import itertools
import json
import os
import pathlib
import pickle
import time

import hedeut
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
import scipy.io
import scipy.linalg
from scipy import interpolate, optimize, signal, sparse, stats

import gvispe.stats
from gvispe import common, estimators, modeling, sde


class Model(modeling.LinearGaussianModel):
    def __init__(self, nx, nu, ny, c_identity):
        super().__init__(nx, nu, ny, not c_identity)
        
        self.c_identity = c_identity
        """Whether C is the identity matrix."""

        self.q_packer = hedeut.Packer(
            A=(self.nx, self.nx),
            B=(self.nx, self.nu),
            C=(0,) if c_identity else (self.ny, self.nx),
            D=(ny, nu),
            vech_log_sQ=(self.ntrilx,),
            vech_log_sR=(self.ntrily,),
        )

        self.nq = self.q_packer.size
        """Number of classical (deterministic) parameters."""

    def C(self, q):
        if self.c_identity:
            return jnp.identity(self.nx)[:self.ny]
        else:
            return self.q_packer.unpack(q)['C']


def impulse_err(imp_true, model, dec):
    """Calculate the impulse response error."""
    n = imp_true.shape[1]
    qdict = model.q_packer.unpack(dec.q)
    sys = signal.StateSpace(qdict['A'], qdict['B'], qdict['C'], qdict['D'], dt=1)
    y = np.array(signal.dimpulse(sys, n=n)[1])
    err = y - imp_true
    return np.sum(err**2, axis=1).mean()


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
        '--c_identity', action=argparse.BooleanOptionalAction,
        help='Whether C is the identity matrix.',
    )
    parser.add_argument(
        '--nx', default=10, type=int,
        help='Number of model states.',
    )
    parser.add_argument(
        '--ny', default=5, type=int,
        help='Number of model outputs.',
    )
    parser.add_argument(
        '--nu', default=5, type=int,
        help='Number of model exogenous inputs.',
    )
    parser.add_argument(
        '--N', default=1000, type=int,
        help='Number samples per batch.',
    )
    parser.add_argument(
        '--Nbatch', default=1000, type=int,
        help='Number of batches.',
    )
    parser.add_argument(
        '--lrate0', default=5e-2, type=float,
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
        '--epochs', default=100, type=int,
        help='Optimization epochs.',
    )
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Random number generator seed.',
    )
    parser.add_argument(
        '--nwin', type=int, default=101, help='Convolution window size.',
    )
    parser.add_argument(
        '--savemat', type=str, 
        help='File name to save data in MATLAB format.',
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

    # Seed the RNG
    np.random.seed(args.seed)

    # Define model dimensions
    nx = args.nx
    nu = args.nu
    ny = args.ny
    N = args.N
    Nbatch = args.Nbatch

    # Generate the model
    A = np.random.rand(nx, nx)
    L, V = np.linalg.eig(A)
    complex_L = np.iscomplex(L).nonzero()[0]
    stab_L = np.random.rand(nx) * np.exp(1j * np.angle(L))
    stab_L[complex_L[1::2]] = stab_L[complex_L[::2]].conj() 
    L = np.where(np.abs(L) < 1, L, stab_L)
    A = np.real(V @ np.diag(L) @ np.linalg.inv(V))
    B = np.random.randn(nx, nu)
    C = np.identity(nx)[:ny] if args.c_identity else np.random.randn(ny, nx)
    D = np.random.randn(ny, nu) * (np.random.rand(ny, nu) > 0.8)
    sQ = np.diag(np.repeat(0.1, nx))
    sR = np.diag(np.repeat(0.15, ny))
    Q = sQ @ sQ.T
    R = sR @ sR.T

    # Simulate
    x = np.zeros((Nbatch*N, nx))
    y = np.zeros((Nbatch*N, ny))
    u = np.where(np.random.rand(Nbatch*N, nu) > 0.5, -1.0, 1.0)
    w = np.random.randn(Nbatch*N, nx)
    v = np.random.randn(Nbatch*N, ny)
    for k in range(N*Nbatch-1):
        x[k+1] = x[k] @ A.T + u[k] @ B.T + w[k] @ sQ.T
    y = x @ C.T + v @ sR.T + u @ D.T

    # Save data to MATLAB file
    if args.savemat is not None:
        matdata = {'A': A, 'B': B, 'C': C, 'D': D, 'y': y, 'u': u}
        scipy.io.savemat(args.savemat, matdata)

    # Divide into batches
    yb = y.reshape(Nbatch, N, ny)
    ub = u.reshape(Nbatch, N, nu)

    # Create model and data objects
    model = Model(nx, nu, ny, args.c_identity)
    data = [estimators.Data(yb[i], ub[i]) for i in range(Nbatch)]
    q_true = model.q_packer.pack(
        A=A, 
        B=B,
        C=np.zeros(0) if args.c_identity else C,
        D=D,
        vech_log_sQ=common.vech(scipy.linalg.logm(sQ)), 
        vech_log_sR=common.vech(scipy.linalg.logm(sR))
    )
    sys_true = signal.StateSpace(A, B, C, D, dt=1)
    imp_true = np.array(signal.dimpulse(sys_true, n=100)[1])

    p = estimators.SmootherKernel(model, args.nwin, elbo_multiplier=-1)
    K0 = np.zeros((nx, ny+nu, args.nwin))
    K0[:, :, args.nwin//2] = np.random.randn(nx, ny+nu) * 1e-3
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

    # JIT the cost and gradient functions
    value_and_grad = jax.jit(jax.value_and_grad(p.elbo))
    value_and_grad(dec0, data[0], coeff)

    # Initialize solver
    dec = dec0
    mincost = np.inf
    sched = optax.exponential_decay(
        init_value=args.lrate0, 
        transition_steps=args.transition_steps,
        decay_rate=args.decay_rate,
    )
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(dec)

    # Run optimization
    start = datetime.datetime.today()
    steps = 0
    for epoch in range(args.epochs):
        for i in np.random.permutation(Nbatch):
            # Calculate cost and gradient
            cost, grad = value_and_grad(dec, data[i], coeff)
            mincost = min(mincost, cost)

            if steps % 100 == 0:
                fooc = p.Decision(*[jnp.sum(v**2) ** 0.5 for v in grad])
                ierr = impulse_err(imp_true, model, dec)
                print(
                    f'{epoch}', f'sched={sched(steps):1.1e}', 
                    f'c={cost:1.2e}', f'm={mincost:1.2e}',
                    f'{fooc.K=:1.2e}', f'{fooc.q=:1.2e}', 
                    f'{fooc.vech_log_S_cond=:1.2e}',
                    f'{fooc.S_cross=:1.2e}',
                    f'{ierr=:1.2e}',
                    sep='\t'
                )
                if args.txtout is not None:
                    secs = (datetime.datetime.today() - start).total_seconds()
                    args.txtout.truncate(0)
                    print(nx, nu, ny, N, Nbatch, ierr, secs, args.seed, 
                          file=args.txtout)

            if any(jnp.any(~jnp.isfinite(v)) for v in grad):
                break

            updates, opt_state = optimizer.update(grad, opt_state)
            dec = optax.apply_updates(dec, updates)
            steps += 1
    secs = (datetime.datetime.today() - start).total_seconds()

    # Save results
    if args.txtout is not None:
        with args.txtout as f:
            ierr = impulse_err(imp_true, model, dec)
            f.truncate(0)
            print(nx, nu, ny, N, Nbatch, ierr, secs, args.seed, file=f)
