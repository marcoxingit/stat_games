"""
Module for probability distributions and their conversion to Bayesian Lenses.
"""

from typing import List, Tuple, Optional, Union, Literal
import numpy as np
from discopy.markov import Ty, Box, Copy, Diagram, Id, Discard
from .bayesian_lens import BayesianLens, R, N

LossType = Literal["likelihood", "l2", "absolute"]


class Distribution:
    """Base class for probability distributions."""

    def __init__(self, name: str):
        self.name = name


class MultivariateNormal(Distribution):
    """Multivariate Normal distribution (R^d × R^(d×d)) -> R^d."""

    def __init__(self, dim: int = 1):
        super().__init__(f"MultivariateNormal(dim={dim})")
        self.dim = dim


class MultivariateNormalChannel(Distribution):
    """Channel that produces MultivariateNormal parameters 1 -> (R^d × R^(d×d))."""

    def __init__(self, dim: int = 1, number_of_channels: int = 1):
        super().__init__(f"MultivariateNormalChannel(dim={dim})")
        self.dim = dim
        self.number_of_channels = number_of_channels


class Categorical(Distribution):
    """Categorical distribution over n outcomes."""

    def __init__(self, n_categories: int):
        super().__init__(f"Categorical({n_categories})")
        self.n_categories = n_categories


class Multinomial(Distribution):
    """Multinomial distribution (n, p1,...,pk) -> (x1,...,xk) where sum(xi)=n."""

    def __init__(self, k: int):
        super().__init__(f"Multinomial(k={k})")
        self.k = k  # number of categories


class Dirichlet(Distribution):
    """Dirichlet distribution R^k -> (0,1)^k where sum = 1."""

    def __init__(self, k: int):
        super().__init__(f"Dirichlet(k={k})")
        self.k = k


class NormalWishart(Distribution):
    """Normal-Wishart distribution R^(d + d×d + 1 + 1) -> (R^d × R^(d×d))."""

    def __init__(self, dim: int = 1):
        super().__init__(f"NormalWishart(dim={dim})")
        self.dim = dim


class MultinomialLogisticRegression(Distribution):
    def __init__(self, k: int, dom_dim: int = 1):
        super().__init__(f"MultinomialLogisticRegression(k={k})")
        self.k = k
        self.dom_dim = dom_dim


class MNLRLogisticCoeffPrior(Distribution):
    def __init__(self, k: int, dom_dim: int = 1):
        super().__init__(f"MNLRLogisticCoeffPrior(k={k})")
        self.k = k
        self.dom_dim = dom_dim


class MatrixNormalWishart(Distribution):
    def __init__(self, dx: int = 1, dy: int = 1):
        super().__init__(f"MatrixNormalWishart(dim={dx,dy})")
        self.dx = dx
        self.dy = dy


def get_loss_channel(
    loss_type: LossType, dom_type: Ty, cod_type: Ty
) -> BayesianLens:
    """Get the appropriate loss channel based on the loss type."""
    if loss_type == "likelihood":
        return BayesianLens(
            dom=dom_type @ cod_type,
            cod=R,
            channel_name="NegativeLogLikelihood",
        )
    elif loss_type == "square":
        return BayesianLens(
            dom=dom_type @ cod_type, cod=R, channel_name="SquaredError"
        )
    elif loss_type == "absolute":
        return BayesianLens(
            dom=dom_type @ cod_type, cod=R, channel_name="AbsoluteError"
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def distribution_to_lens(
    dist: Distribution, loss: LossType = "likelihood"
) -> BayesianLens:
    """Convert a probability distribution to a Bayesian Lens with specified loss name."""

    if isinstance(dist, MultivariateNormal):
        # MultivariateNormal as (R^d × R^(d×d)) -> R^d
        dom_type = R ** (dist.dim + dist.dim**2)
        cod_type = R**dist.dim
        return BayesianLens(
            dom=dom_type,  # mean vector and covariance matrix
            cod=cod_type,  # sample vector
            channel_name=dist.name,
            loss_name=loss + dist.name,
        )

    elif isinstance(dist, MultivariateNormalChannel):
        # Channel producing MultivariateNormal parameters 1 -> (R^d × R^(d×d))
        dom_type = Ty(f"{dist.number_of_channels}")
        cod_type = R ** (dist.dim + dist.dim**2)
        return BayesianLens(
            dom=dom_type,
            cod=cod_type,
            channel_name=dist.name,
            loss_name=loss,
        )

    elif isinstance(dist, NormalWishart):
        # Normal-Wishart as (μ₀, Λ, κ, ν) -> (μ, Λ)
        # dom: mean vector μ₀ (d), scale matrix Λ (d×d), scale κ (1), degrees of freedom ν (1)
        # cod: mean vector μ (d) and precision matrix Λ (d×d)
        dom_type = R ** (dist.dim + dist.dim**2 + 1 + 1)
        cod_type = R ** (dist.dim + dist.dim**2)
        return BayesianLens(
            dom=dom_type,
            cod=cod_type,
            channel_name=dist.name,
            loss_name=loss + dist.name,
        )

    elif isinstance(dist, Categorical):
        # Categorical as discrete distribution
        dom_type = Ty()
        cod_type = Ty(f"{dist.n_categories}")
        return BayesianLens(
            dom=dom_type,
            cod=cod_type,
            channel_name=dist.name,
            loss_name=loss + dist.name,
        )

    elif isinstance(dist, Multinomial):
        # Multinomial as (R × R^k) -> N^k
        dom_type = R ** (dist.k + 1)
        cod_type = N**dist.k
        return BayesianLens(
            dom=dom_type,  # n trials and k probabilities
            cod=cod_type,  # k counts
            channel_name=dist.name,
            loss_name=loss + dist.name,
        )

    elif isinstance(dist, Dirichlet):
        # Dirichlet as R^k -> (0,1)^k
        dom_type = R**dist.k
        cod_type = R**dist.k
        return BayesianLens(
            dom=dom_type,  # k concentration parameters
            cod=cod_type,  # k probabilities summing to 1
            channel_name=dist.name,
            loss_name=loss + dist.name,
        )
    elif isinstance(dist, MultinomialLogisticRegression):
        # MultinomialLogisticRegression as (R^k × R^d) -> (R^k)
        dom_type = R ** (dist.k * (dist.dom_dim + 1) + dist.dom_dim)
        cod_type = Ty(f"{dist.k}")

        return BayesianLens(
            dom=dom_type,
            cod=cod_type,
            channel_name=dist.name,
            loss_name=loss + dist.name,
        )

    elif isinstance(dist, MNLRLogisticCoeffPrior):
        # MNLRLogisticCoeffPrior as (R^k × R^d) -> (R^k)
        dom_type = Ty()
        cod_type = R ** (dist.k * (dist.dom_dim + 1))
        return BayesianLens(
            dom=dom_type,
            cod=cod_type,
            channel_name=dist.name,
            loss_name=loss + dist.name,
        )

    elif isinstance(dist, MatrixNormalWishart):
        dom_type = Ty()
        cod_type = R ** (2 * (dist.dx + dist.dy))
        return BayesianLens(
            dom=dom_type, cod=cod_type, channel_name=dist.name
        )

    else:
        raise ValueError(f"Unsupported distribution type: {type(dist)}")


def normal_mixture_model(
    n_components: int, dim: int = 1, loss: LossType = "likelihood"
) -> BayesianLens:
    """Create a Multivariate Normal Mixture Model as composition of lenses."""
    # Create categorical distribution over components
    categorical = distribution_to_lens(Categorical(n_components), loss=loss)

    # Create MultivariateNormal channel that produces parameters
    normal_channel = distribution_to_lens(
        MultivariateNormalChannel(dim, number_of_channels=n_components), loss=loss
    )

    # Compose them
    return categorical >> normal_channel
