"""Common functions and utilities."""


import hedeut
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt


@hedeut.jax_vectorize(signature='(n,m)->(p)')
def vech(M: npt.ArrayLike) -> npt.NDArray:
    """Pack the lower triangle of a matrix into a vector, columnwise.
    
    Follows the definition of Magnus and Neudecker (2019), Sec. 3.8, 
    DOI: 10.1002/9781119541219
    """
    return M[jnp.triu_indices_from(M.T)[::-1]]


@hedeut.jax_vectorize(signature='(m)->(n,n)')
def matl(v: npt.ArrayLike) -> npt.NDArray:
    """Unpack a vector into a square lower triangular matrix."""
    n = int(np.sqrt(2 * v.size + 0.25) - 0.5)
    assert v.ndim == 1
    assert n * (n + 1) / 2 == v.size
    M = jnp.zeros((n, n))
    return M.at[jnp.triu_indices_from(M)[::-1]].set(v)


def tria_qr(*args) -> npt.NDArray:
    """Array triangularization routine using QR decomposition."""
    M = jnp.concatenate(args, axis=-1)
    Q, R = jnp.linalg.qr(M.T)
    sig = jnp.sign(jnp.diag(R))
    return R.T * sig


def tria_chol(*args) -> npt.NDArray:
    """Array triangularization routine using Cholesky decomposition."""
    M = jnp.concatenate(args, axis=-1)
    MMT = M @ M.T
    return jnp.linalg.cholesky(MMT)


@hedeut.jax_vectorize(signature='(k,m),(k,n)->(k,k)')
def tria2_qr(m1, m2):
    """Triangularization of two matrices using QR decomposition."""
    return tria_qr(m1, m2)


@hedeut.jax_vectorize(signature='(k,m),(k,n)->(k,k)')
def tria2_chol(m1, m2):
    """Triangularization of two matrices using Cholesky decomposition."""
    return tria_chol(m1, m2)


@hedeut.jax_vectorize(signature='(n),(m)->(p)')
def conv(a, v):
    """Vectorized convolution of a vector with a kernel."""
    return jnp.convolve(a, v, mode='valid')

