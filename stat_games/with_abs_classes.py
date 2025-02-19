from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from discopy.markov import Ty, Box
import numpyro.distributions as dist

R = Ty("R")
N = Ty("N")

@dataclass
class OpenModel(ABC):
    dom: Ty
    cod: Ty
    latent: Optional[Ty] = field(default_factory=lambda: Ty())
    channel: Optional[Box] = None
    channel_name: str = "channel"
    prob_density: Optional[Box] = None

    def __post_init__(self):
        if self.channel is None:
            self.channel = Box(self.channel_name, self.dom, self.latent @ self.cod)
        if self.prob_density is None:
            # Default probability density is set to a Normal distribution.
            self.prob_density = Box(
                self.channel_name + "_density",
                self.dom,
                self.cod,
                distribution=dist.Normal(0., 1.)
            )
    
    @abstractmethod
    def build_model(self):
        """Subclasses should implement this to construct their specific model."""
        pass

@dataclass
class BayesianLens(OpenModel):
    inversion: Optional[Box] = None

    def __post_init__(self):
        super().__post_init__()
        if self.inversion is None:
            self.inversion = lambda prior: OpenModel(self.cod, self.dom, self.latent, self.channel)
    
    @abstractmethod
    def update(self):
        """Subclasses should implement this for their specific update mechanism."""
        pass

@dataclass
class StatGame(BayesianLens):
    loss: Optional[Box] = None
    loss_name: str = "loss"

    def build_model(self):
        # Implementation for building a statistical game model.
        pass

    def update(self):
        # Implementation for updating the model in StatGame.
        pass

@dataclass
class ExponentialFamily(OpenModel, ABC):
    natural_param: Optional[Ty] = field(default_factory=lambda: Ty())
    expectation_params: Optional[Box] = None

    @abstractmethod
    def compute_natural_parameters(self):
        """Compute natural parameters from standard parameters."""
        pass

@dataclass
class GaussianChannel(ExponentialFamily):
    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        # Initialize a NumPyro Normal distribution instance.
        self.prob_density = Box(
            self.channel_name + "_density",
            self.dom,
            self.cod,
            distribution=dist.Normal(self.mu, self.sigma)
        )

    def compute_natural_parameters(self):
        # For a Normal distribution in exponential-family form:
        # Natural parameters are often defined as (mu/sigma^2, -1/(2sigma^2))
        return (self.mu / (self.sigma**2), -1 / (2 * self.sigma**2))
    
    def build_model(self):
        # Here, you could build the specific Gaussian model.
        pass

    @property
    def density(self):
        # Return the log probability method.
        return self.prob_density.distribution.log_prob

# Usage example:
gc = GaussianChannel(dom=R, cod=N, mu=0.0, sigma=1.0)
print("Log-density at 1.0:", gc.density(1.0))
print("Natural parameters:", gc.compute_natural_parameters())
