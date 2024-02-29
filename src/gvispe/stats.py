"""Statistical helper functions."""


from inspect import signature
import jax.numpy as jnp
import jax.scipy as jsp
import hedeut as utils
import numpy as np
import math
from scipy import special


@utils.jax_vectorize(signature='(x),(x),(x,x)->()')
def mvn_logpdf(x, mu, isCov):
    """Multivariate normal log-density. 
    
    `isCov` is the inverse of the cholesky factor of the covariance matrix.
    """
    cte = -0.5 * len(x) * jnp.log(2 * jnp.pi)
    dev = x - mu
    normdev = isCov @ dev
    return -0.5 * jnp.sum(normdev ** 2) + jnp.log(isCov.diagonal()).sum() + cte


@utils.jax_vectorize(signature='(x),(x),(x,x)->()')
def mvn_logpdf_logchol(x, mu, log_chol_cov):
    """Multivariate normal log-density with log-chol covariance matrix.
    
    `log_chol_cov` is the matrix logarithm of the Cholesky factor of the 
    covariance matrix.
    """
    inv_chol_cov = jsp.linalg.expm(-log_chol_cov)
    cte = -0.5 * len(x) * jnp.log(2 * jnp.pi)
    dev = x - mu
    normdev = inv_chol_cov @ dev
    return -0.5 * jnp.sum(normdev ** 2) - jnp.trace(log_chol_cov) + cte


def ghcub(order, dim):
    """Gauss-Hermite nodes and weights for Gaussian cubature."""
    x, w_unnorm = special.roots_hermitenorm(order)
    w = w_unnorm / w_unnorm.sum()
    xrep = [x] * dim
    wrep = [w] * dim
    xmesh = np.meshgrid(*xrep)
    wmesh = np.meshgrid(*wrep)
    X = np.hstack(tuple(xi.reshape(-1,1) for xi in xmesh))
    W = math.prod(wmesh).flatten()
    return X, W


def sigmapts(dim: int):
    """Sigma points and weights for unscented transform without center point."""
    X = np.r_[np.eye(dim), -np.eye(dim)] * np.sqrt(dim)
    W = 0.5 / dim
    return X, W
