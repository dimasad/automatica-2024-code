"""Estimation problems and algorithms."""


import functools
import typing

import hedeut
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpy.typing as npt
from scipy import sparse

from . import common, stats


class Data(typing.NamedTuple):
    y: npt.NDArray
    """Measurements."""

    u: npt.NDArray
    """Exogenous inputs."""

    def zeros_like(self):
        u = jnp.zeros_like(self.u)
        y = jnp.zeros_like(self.y)
        return self.__class__(u=u, y=y)


class XCoeff(typing.NamedTuple):
    """Tuple of coeffs for expectation wrt the state at each time sample."""

    us_dev: npt.NDArray
    """Unscaled deviations of the state."""

    w: npt.NDArray | float
    """Expectation weights."""

    def zeros_like(self):
        us_dev = jnp.zeros_like(self.us_dev)
        w = jnp.zeros_like(self.w)
        return self.__class__(us_dev, w)


class XPairCoeff(typing.NamedTuple):
    """Tuple of coeffs for expectation wrt pairs of consecutive states."""

    curr_us_dev: npt.NDArray
    """Unscaled deviations of the current state."""

    next_us_dev: npt.NDArray
    """Unscaled deviations of the next state."""

    w: npt.NDArray
    """Expectation weights."""

    def zeros_like(self):
        curr_us_dev = jnp.zeros_like(self.curr_us_dev)
        next_us_dev = jnp.zeros_like(self.next_us_dev)
        w = jnp.zeros_like(self.w)
        return self.__class__(curr_us_dev, next_us_dev, w)


class ExpectationCoeff(typing.NamedTuple):
    """Tuple of coefficients for expectation wrt posterior distributions."""
    x: XCoeff
    xpair: XPairCoeff

    def zeros_like(self):
        x = self.x.zeros_like()
        xpair = self.xpair.zeros_like()
        return self.__class__(x, xpair)


class GVIProblemVariables(typing.NamedTuple):
    """Represents decision variables and/or derived quantities."""
    xbar: npt.NDArray
    log_S_cond: npt.NDArray
    S: npt.NDArray
    S_cond: npt.NDArray
    S_cross: npt.NDArray


class GVI:
    """Base for Gaussian Variational Inference estimators."""

    tria2 = staticmethod(common.tria2_qr)
    """Matrix triangularization routine."""

    def __init__(self, model, elbo_multiplier=1):
        self.model = model
        """The underlying dynamical system model."""

        nx = self.model.nx
        self.ntrilx = nx * (nx + 1) // 2
        """Number of elements in lower triangle of nx by nx matrix."""

        self.elbo_multiplier = elbo_multiplier
        """Multiplier for the ELBO (to use with minimizers)."""

    def sample_x(self, v: GVIProblemVariables, coeff: XCoeff):
        """Sample the state from the assumed density."""
        # Obtain scaled deviation from mean
        x_dev = jnp.inner(coeff.us_dev, v.S)

        if x_dev.ndim == 2:
            x_dev = x_dev[:, None]

        # Add mean to deviation and return
        return v.xbar + x_dev

    def sample_xpair(self, v: GVIProblemVariables, coeff: XPairCoeff):
        """Sample the a pair of consecutive states from the assumed density."""
        # Get the square-root correlations with correct shape
        steady_state = v.S_cond.ndim == 2
        S = v.S[None] if steady_state else v.S[:-1]
        S_cond = v.S_cond[None] if steady_state else v.S_cond[1:]
        S_cross = v.S_cross[None] if steady_state else v.S_cross

        # Obtain scaled deviations from mean
        xcurr_dev = jnp.inner(coeff.curr_us_dev, S)
        xnext_dev = (jnp.inner(coeff.curr_us_dev, S_cross)
                     + jnp.inner(coeff.next_us_dev, S_cond))

        # Add mean to deviations and return
        xcurr = v.xbar[:-1] + xcurr_dev
        xnext = v.xbar[1:] + xnext_dev
        return xnext, xcurr

    def elbo(self, dec, data: Data, coeff: ExpectationCoeff):
        # Compute problem variables
        v: GVIProblemVariables = self.problem_variables(dec, data)

        # Sample from the assumed density
        x = self.sample_x(v, coeff.x)
        xnext, xcurr = self.sample_xpair(v, coeff.xpair)

        # Get the data variables
        u = getattr(v, 'u', data.u)
        y = getattr(v, 'y', data.y)

        # Compute elements of the ELBO
        model = self.model
        entropy = self.entropy(v)
        prior_logpdf = model.prior_logpdf(x[:, 0], dec.q)
        meas_logpdf = model.meas_logpdf(y, x, u, dec.q)
        trans_logpdf = model.trans_logpdf(xnext, xcurr, u[:-1], dec.q)

        # Get the average log densities
        avg_prior_logpdf = (prior_logpdf * coeff.x.w).sum(0)
        avg_meas_logpdf = (meas_logpdf.sum(-1) * coeff.x.w).sum(0)
        avg_trans_logpdf = (trans_logpdf.sum(-1) * coeff.xpair.w).sum(0)

        elbo = entropy + avg_prior_logpdf + avg_meas_logpdf + avg_trans_logpdf
        return self.elbo_multiplier * elbo

    @functools.cached_property
    def elbo_grad(self):
        return jax.grad(self.elbo)
    
    def elbo_hvp(self, dec, dec_d, data: Data, coeff: ExpectationCoeff):
        primals = dec, data, coeff
        duals = dec_d, data.zeros_like(), coeff.zeros_like()
        return jax.jvp(self.elbo_grad, primals, duals)[1]
    
    def elbo_packed(self, dvec, data: Data, coeff: ExpectationCoeff, packer):
        dec = self.Decision(**packer.unpack(dvec))
        return self.elbo(dec, data, coeff)

    def elbo_grad_packed(self, dvec, data: Data, coeff: ExpectationCoeff,
                         packer):
        dec = self.Decision(**packer.unpack(dvec))
        grad = self.elbo_grad(dec, data, coeff)
        return packer.pack(*grad)

    def elbo_hvp_packed(self, dvec, dvec_d, data:Data, coeff:ExpectationCoeff,
                        packer):
        dec = self.Decision(**packer.unpack(dvec))
        dec_d = self.Decision(**packer.unpack(dvec_d))
        hvp = self.elbo_hvp(dec, dec_d, data, coeff)
        return packer.pack(*hvp)

    def fix_and_jit(self, fname: str, /, **kwargs):
        f = functools.partial(getattr(self, fname), **kwargs)
        return jax.jit(f)


class SteadyState(GVI):
    class Decision(typing.NamedTuple):
        """Problem decision variables."""
        q: npt.NDArray
        xbar: npt.NDArray
        vech_log_S_cond: npt.NDArray
        S_cross: npt.NDArray

    def packer(self, N):
        return hedeut.Packer(
            q=(self.model.nq,),
            xbar=(N, self.model.nx),
            vech_log_S_cond=(self.ntrilx,),
            S_cross=(self.model.nx, self.model.nx),
        )

    def problem_variables(self, dec: Decision, data: Data): 
        log_S_cond = common.matl(dec.vech_log_S_cond)
        S_cond = jsp.linalg.expm(log_S_cond)
        S = self.tria2(S_cond, dec.S_cross)
        return GVIProblemVariables(
            xbar=dec.xbar,
            log_S_cond=log_S_cond,
            S=S,
            S_cond=S_cond,
            S_cross=dec.S_cross,
        )
    
    def entropy(self, v: GVIProblemVariables):
        N = len(v.xbar)

        # Compute initial state entropy
        entro_x0 = jnp.sum(jnp.log(jnp.abs(jnp.diag(v.S))))

        # Compute entropy of the remaining states
        entro_xrem = (N - 1) * jnp.trace(v.log_S_cond)

        # Return joint entropy
        return entro_x0 + entro_xrem


class Transient(GVI):
    class Decision(typing.NamedTuple):
        """Problem decision variables."""
        q: npt.NDArray
        xbar: npt.NDArray
        vech_log_S_cond: npt.NDArray
        S_cross: npt.NDArray

    def packer(self, N: int):
        return hedeut.Packer(
            q=(self.model.nq,),
            xbar=(N, self.model.nx),
            vech_log_S_cond=(N, self.ntrilx),
            S_cross=(N-1, self.model.nx, self.model.nx),
        )

    def problem_variables(self, dec: Decision, data: Data): 
        nx = self.model.nx

        log_S_cond = common.matl(dec.vech_log_S_cond)
        S_cond = jsp.linalg.expm(log_S_cond)
        S_cross_aug = jnp.concatenate((jnp.zeros((1, nx, nx)), dec.S_cross))
        S = self.tria2(S_cond, S_cross_aug)
        return GVIProblemVariables(
            xbar=dec.xbar,
            log_S_cond=log_S_cond,
            S=S,
            S_cond=S_cond,
            S_cross=dec.S_cross,
        )
    
    def entropy(self, v: GVIProblemVariables):
        return jnp.trace(v.log_S_cond, axis1=1, axis2=2).sum()


class SmootherKernel(SteadyState):
    def __init__(self, model, nwin: int, elbo_multiplier=1):
        super().__init__(model, elbo_multiplier)

        assert nwin % 2 == 1, "Window length must be odd"
        self.nwin = nwin
        """Length of the convolution window."""

    class Decision(typing.NamedTuple):
        """Problem decision variables."""
        q: npt.NDArray
        K: npt.NDArray
        vech_log_S_cond: npt.NDArray
        S_cross: npt.NDArray

    class ProblemVariables(typing.NamedTuple):
        """Problem variables of BatchedSteadyState problem formulation."""
        xbar: npt.NDArray
        log_S_cond: npt.NDArray
        S: npt.NDArray
        S_cond: npt.NDArray
        S_cross: npt.NDArray
        u: npt.NDArray
        y: npt.NDArray

    def packer(self, N=None):
        return hedeut.Packer(
            q=(self.model.nq,),
            K=(self.model.nx, self.model.ny + self.model.nu, self.nwin),
            vech_log_S_cond=(self.ntrilx,),
            S_cross=(self.model.nx, self.model.nx),
        )

    def smooth(self, K, data: Data):
        sig = jnp.c_[data.y, data.u].T
        return common.conv(sig, K).sum(1).T

    def problem_variables(self, dec: Decision, data: Data): 
        xbar = self.smooth(dec.K, data)
        log_S_cond = common.matl(dec.vech_log_S_cond)
        S_cond = jsp.linalg.expm(log_S_cond)
        S = self.tria2(S_cond, dec.S_cross)
        skip = self.nwin // 2
        return self.ProblemVariables(
            xbar=xbar,
            log_S_cond=log_S_cond,
            S=S,
            S_cond=S_cond,
            S_cross=dec.S_cross,
            u=data.u[skip:-skip],
            y=data.y[skip:-skip],
        )


class PEM:
    """Prediction Error Method estimator."""

    class Decision(typing.NamedTuple):
        """Problem decision variables."""
        q: npt.NDArray
        K: npt.NDArray
        vech_log_sR: npt.NDArray
        x0: npt.NDArray

    def __init__(self, model):
        self.model = model
        """The underlying dynamical system model."""

        self.ntrily = model.ny * (model.ny + 1) // 2
        """Number of elements in lower triangle of ny by ny matrix."""

    def predfun(self, x, y, u, dec: Decision):
        ypred = self.model.h(x, u, dec.q)
        e = y - ypred
        xnext = self.model.f(x, u, dec.q) + dec.K @ e
        return xnext, ypred

    def cost(self, dec: Decision, data: Data):
        scanfun = lambda x, datum: self.predfun(x, *datum, dec)
        x0 = dec.x0 if len(dec.x0) > 0 else jnp.zeros(self.model.nx)
        xnext, ypred = jax.lax.scan(scanfun, x0, data)

        log_sR = common.matl(dec.vech_log_sR)
        return -stats.mvn_logpdf_logchol(data.y, ypred, log_sR).sum(0)
    
    def cost_grad(self, dec, data: Data):
        return jax.grad(self.cost)(dec, data)

    def cost_hvp(self, dec, dec_d, data: Data):
        primals = dec, data
        duals = dec_d, data.zeros_like()
        duals = jax.tree_util.tree_map(lambda a:jnp.asarray(a, float), duals)
        return jax.jvp(self.cost_grad, primals, duals)[1]
