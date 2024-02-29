"""Common SDE modelling/estimation models."""


import functools

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import hedeut as utils


class BaseCondNormalScheme:
    """Base SDE discretization scheme for conditionally normal transitions."""

    sigma_diagonal = False
    """Whether transition covariance matrix is diagonal."""

    def __init__(self, sde, dt=None):
        self.sde = sde
        self.dt = dt
    
    @functools.cached_property
    def nx(self):
        """Number of states."""
        return self.sde.nx

    @functools.cached_property
    def nu(self):
        """Number of exogenous inputs."""
        return self.sde.nu

    @functools.cached_property
    def nq(self):
        """Number of classical (deterministic) parameters."""
        return self.sde.nq

    @utils.jax_vectorize_method(signature='(x),(x),(u),(q),()->()')
    def _trans_logpdf(self, xnext, x, u, q, dt):
        mu = self._trans_mean(x, u, q, dt)
        if self.sigma_diagonal:
            sigma_diag = self._trans_sigma_diag(x, u, q, dt)
            return jsp.stats.norm.logpdf(xnext, mu, sigma_diag).sum(-1)
        else:
            sigma = self._trans_sigma(x, u, q, dt)
            cov = sigma @ sigma.T
            return jsp.stats.multivariate_normal.logpdf(xnext, mu, cov)
    
    def trans_logpdf(self, xnext, x, u, q, dt=None):
        dt = dt or self.dt
        assert dt is not None
        return self._trans_logpdf(xnext, x, u, q, dt)
    
    @utils.jax_jit_method
    def step(self, x, u, q, dt, key):
        mu = self._trans_mean(x, u, q, dt)
        eps = jax.random.normal(key, (self.neps,))
        if self.sigma_diagonal:
            sigma_diag = self._trans_sigma_diag(x, u, q, dt)
            return mu + sigma_diag * eps
        else:
            sigma = self._trans_sigma(x, u, q, dt)
            return mu + sigma @ eps
    
    def sim(self, x0, u, q, t, key):
        dt = jnp.diff(t)
        xk = x0
        x = [x0]
        for k, dtk in enumerate(dt):
            uk = u[..., k,:]
            key, subkey = jax.random.split(key)
            xk = self.step(xk, uk, q, dtk, subkey)
            x.append(xk)
        return jnp.array(x)


class EulerScheme(BaseCondNormalScheme):
    """Euler--Maruyama SDE discretization scheme."""

    def __init__(self, sde, dt=None):
        super().__init__(sde, dt)

        self.neps = sde.nw
        """Dimension of the simulation step noise vector."""

        self.sigma_diagonal = hasattr(sde, 'G_diag')
        """Whether transition covariance matrix is diagonal."""

    def _trans_mean(self, x, u, q, dt):
        """Mean of SDE transition, given the previous state."""
        f = self.sde.f(x, u, q)
        return x + f*dt

    def _trans_sigma(self, x, u, q, dt):
        """Mean of 'square-root' factor of transition covariance."""
        G = self.sde.G(q)
        return G * jnp.sqrt(dt)

    def _trans_sigma_diag(self, x, u, q, dt):
        """Mean of 'square-root' factor of transition covariance."""
        return self.sde.G_diag(q) * jnp.sqrt(dt)


class SO15ITScheme(BaseCondNormalScheme):
    """Strong order 1.5 Ito--Taylor rule method for additive noise."""

    def __init__(self, sde, dt=None):
        super().__init__(sde, dt)

        self.neps = 2 * sde.nw
        """Dimension of the simulation step noise vector."""
    
    @functools.cached_property
    def nx(self):
        """Number of states."""
        return self.sde.nx
    
    @functools.cached_property
    def _df_dx(self):
        """Drift jacobian."""
        s = "(x),(u),(q)->(x,x)"
        return jax.jit(jnp.vectorize(jax.jacobian(self.sde.f), signature=s))
    
    @functools.cached_property
    def _d2f_dx2(self):
        """Second derivatives of drift function."""
        nx = self.sde.nx
        df_dx_flat = lambda x,u,q: self._df_dx(x, u, q).flatten()
        d2f_dx2_flat_jac = jax.jacobian(df_dx_flat)
        d2f_dx2 = lambda x,u,q: d2f_dx2_flat_jac(x,u,q).reshape(nx, nx, nx)
        s = '(x),(u),(q)->(x,x,x)'
        return jax.jit(jnp.vectorize(d2f_dx2, signature=s))
    
    def _trans_mean(self, x, u, q, dt):
        """Mean of SDE transition, given the previous state."""
        f = self.sde.f(x, u, q)
        df_dx = self._df_dx(x, u, q)
        d2f_dx2 = self._d2f_dx2(x, u, q)
        G = self.sde.G(q)
        
        Q = G @ G.T
        L0f = (df_dx @ f[..., None]) [..., 0]
        L0f = L0f + 0.5 * jnp.sum(Q[None] * d2f_dx2, (-1,-2))
        return x + f*dt + 0.5 * L0f * dt**2
    
    def _trans_sigma(self, x, u, q, dt):
        """Mean of 'square-root' factor of transition covariance."""
        df_dx = self._df_dx(x, u, q)
        G = self.sde.G(q)
        Lf = df_dx @ G
        
        M1 = jnp.sqrt(dt) * G + 0.5 * dt ** 1.5 * Lf
        M2 = 0.5 * dt ** 1.5 / jnp.sqrt(3) * Lf
        return jnp.c_[M1, M2]
