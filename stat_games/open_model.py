from dataclasses import dataclass, field
from typing import Optional
from discopy.cat import Category
from discopy.markov import Ty, Box
import numpyro.distributions as dist  # NumPyro's distributions

# Define some basic types.
R = Ty("R")
N = Ty("N")

# Extend Box to include a probabilistic distribution.
@dataclass
class ProbBox(Box):
    distribution: Optional[dist.Distribution] = None

# Now define our core classes.
@dataclass
class OpenModel:
    dom: Ty
    cod: Ty
    latent: Optional[Ty] = field(default_factory=lambda: Ty())
    channel: Optional[Box] = None
    channel_name: str = "channel"
    # Here we change the type hint to Optional[ProbBox] to allow a NumPyro distribution.
    prob_density: Optional[ProbBox] = None

    def __post_init__(self):
        # Initialize channel if not provided.
        if self.channel is None:
            self.channel = Box(self.channel_name, self.dom, self.latent @ self.cod)
        # If no probability density is given, set a default using a NumPyro distribution.
        if self.prob_density is None:
            # Here we use a default Normal distribution as an example.
            self.prob_density = ProbBox(
                self.channel_name + "_density",
                self.dom,
                self.cod,
                distribution=dist.Normal(0., 1.)
            )
            
@dataclass
class BayesianLens(OpenModel):
    inversion: Optional[Box] = None

    def __post_init__(self):
        # Call parent post-init.
        super().__post_init__()
        if self.inversion is None:
            # Here we define a trivial inversion as a lambda returning an OpenModel with swapped domain/cod.
            self.inversion = lambda prior: OpenModel(self.cod, self.dom, self.latent, self.channel)

@dataclass
class StatGame(BayesianLens):
    loss: Optional[Box] = None
    loss_name: str = "loss"

@dataclass
class ExponentialFamily(OpenModel):
    natural_param: Optional[Ty] = field(default_factory=lambda: Ty())
    expectation_params: Optional[Box] = None
    # Optionally, you could also store a NumPyro distribution in prob_density here.
