# from discopy.cat import Category
# from discopy.markov import Ty, Box, Copy, Diagram, Id, Swap
# from typing import Optional
# from functools import reduce

from dataclasses import dataclass, field
from typing import Optional
from discopy.cat import Category
from discopy.markov import Ty, Box




R = Ty("R")
N = Ty("N")




@dataclass
class OpenModel:
    dom: Ty
    cod: Ty
    latent: Optional[Ty] = Ty()
    channel: Optional[Box] = None
    channel_name: str = "channel"

    def __post_init__(self):
        # if self.latent is None:
        #     self.latent = Ty()
        if self.channel is None:
            self.channel = Box(self.channel_name, self.dom, self.latent @ self.cod)
            
            
    # define sequential and parallel composition
    

@dataclass
class BayesianLens(OpenModel):
    inversion: Optional[Box] = None
    def __post_init__(self):
        # if self.latent is None:
        #     self.latent = Ty()
        if self.inversion is None:
            self.inversion = lambda prior: OpenModel(self.cod, self.dom, self.latent, self.channel)

@dataclass
class StatGame(BayesianLens):
    loss: Optional[Box] = None
    loss_name: str = "loss"
