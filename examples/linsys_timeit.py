#!/usr/bin/env python3

"""Timining of linear--Gaussian variational system identification functions."""

import argparse
import timeit

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

import gvispe.stats
from gvispe import common, estimators, modeling

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
        '--nwin', type=int, default=101, help='Convolution window size.',
    )
    parser.add_argument(
        '--number', default=10, type=int, 
        help='How many times the function is executed by timeit.',
    )
    parser.add_argument(
        '--repeat', default=10, type=int, 
        help='How many times the timing is repeated by timeit.',
    )
    parser.add_argument(
        '--txtout', type=argparse.FileType('w'), help='Text output file.',
    )
    parser.add_argument(
        '--estimator', choices=['gvi', 'pem'],
        default='gvi', help='Parameter estimation method.',
    )
    args = parser.parse_args()

    # Apply JAX config options
    if args.jax_x64:
        jax.config.update("jax_enable_x64", True)
    if args.jax_platform:
        jax.config.update('jax_platform_name', args.jax_platform)

    # Define model dimensions
    nx = args.nx
    nu = args.nu
    ny = args.ny
    N = args.N
    nwin = args.nwin

    # Create model and data objects
    model = modeling.LinearGaussianModel(nx, nu, ny)

    if args.estimator == 'gvi':
        p = estimators.SmootherKernel(model, nwin)
        dec = p.Decision(
            q=jnp.zeros(model.nq),
            K=jnp.zeros((nx, ny+nu, nwin)),
            vech_log_S_cond=jnp.zeros(p.ntrilx),
            S_cross=jnp.zeros((nx, nx))
        )
        data = estimators.Data(
            y=jnp.zeros((N + nwin -1, ny)), 
            u=jnp.zeros((N + nwin -1, nu))
        )
        # Obtain integration coefficients
        pair_us_dev, xpair_w = gvispe.stats.sigmapts(2*nx)
        coeff = estimators.ExpectationCoeff(
            estimators.XCoeff(*gvispe.stats.sigmapts(nx)),
            estimators.XPairCoeff(pair_us_dev[:, :nx], pair_us_dev[:, nx:], xpair_w)
        )
        value_and_grad = jax.jit(
            jax.value_and_grad(lambda dec, data: p.elbo(dec, data, coeff))
        )
    else:
        p = estimators.PEM(model)
        dec = p.Decision(
            q=jnp.zeros(model.nq),
            K=jnp.zeros((nx, ny)),
            vech_log_sR=jnp.zeros(p.ntrily),
        )
        data = estimators.Data(
            y=jnp.zeros((N, ny)), 
            u=jnp.zeros((N, nu))
        )
        value_and_grad = jax.jit(jax.value_and_grad(p.cost))

    # JIT the cost and gradient functions
    value_and_grad(dec, data)

    # Time the function
    fun = lambda: value_and_grad(dec, data)[1].q.block_until_ready()
    timing_list = timeit.repeat(fun, number=args.number, repeat=args.repeat)
    min_time = min(timing_list) / args.number

    # Print results to stdout
    print(nx, nu, ny, N, nwin, min_time)

    # Save results
    if args.txtout is not None:
        with args.txtout as f:
            print(nx, nu, ny, N, nwin, min_time, file=f)
