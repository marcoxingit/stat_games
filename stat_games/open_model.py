from dataclasses import dataclass, field
from typing import Optional
from discopy.cat import Category
from discopy.markov import Ty, Box, Copy, Diagram, Id, Swap
import numpyro.distributions as dist  # NumPyro's distributions
from functools import partial
def NACopy(x: Ty, n: int = 2) -> Box:
    if x.is_atomic:
        return Copy(x, n)
    elif len(x) == 2 and n == 2:
        return Copy(x[0], n) @ Copy(x[1], n) >> Id(x[0]) @ NASwap(x[0], x[1]) @ Id(x[1])
    return Box("Copy", x, x**n)

def pack_args(x):
    return x if isinstance(x, tuple) else (x,)


# Define some basic types.
R = Ty("R")
N = Ty("N")

def channelize_log_prob(dist):
    return lambda *x: dist(*x).log_prob # We can use partial instead

# Extend Box to include a probabilistic distribution.
@dataclass
class ProbBox(Box):
    log_prob: Optional[dist.Distribution] = None

    def __init__(self, name: str, dom: Ty, cod: Ty, log_prob_channel = None):
        super().__init__(name, dom, cod)  # Call Box's constructor
        self.log_prob_channel = log_prob_channel  # Set the distribution attribute
        
    # def __rshift__(self, other: ProbBox) -> Box:
    #     box = self.compose(other)
    # def compose(self, other: ProbBox) -> Box:
    #     return lambda x: self.log_prob(x, other.log_prob(x))

# Now define our core classes.


@dataclass
class OpenModel:
    dom: Ty
    cod: Ty
    log_prob_channel: Optional[Ty]
    latent: Optional[Ty] = field(default_factory=lambda: Ty())
    symbolic_channel: Optional[ProbBox] = field(default_factory=lambda: None)
    depth: int = 0
    

    def __post_init__(self):
        # Initialize channel if not provided.
        if self.symbolic_channel is None:
            self.symbolic_channel = ProbBox(name="channel", dom=self.dom, cod=self.latent @ self.cod, log_prob_channel=self.log_prob_channel)
            print(self.symbolic_channel.dom)
            
        self.len_target = len(self.latent) + len(self.cod)
        
    def __rshift__(self, other) :
        return self.compose(other)
    # WE need to use some functional stuff, like currying
    def compose(self, other):
        dom = self.dom
        cod = other.cod
        dept = self.depth + 1
        latent_space = self.latent @ other.latent @ other.cod
        print(self.len_target, self.dom, self.cod)
        
        log_prob_channel = lambda x: (lambda *full_target_space: self.log_prob_channel(x)(*full_target_space[:self.len_target]) + other.log_prob_channel(*full_target_space[len(self.latent):len(self.latent)+len(other.dom)])(*full_target_space[len(self.latent)+len(other.dom):]))
        # else:
        #     log_prob_channel = lambda x: (lambda full_target_space: self.log_prob_channel(x)(full_target_space[:self.len_target]))
        # log_prob_channel = lambda x: (lambda full_target_space: self.log_prob_channel(x)(*pack_args(full_target_space)[:self.len_target]))

        symbolic_channel = self.symbolic_channel >> (Id(self.latent) @ NACopy(self.cod) ) >> (Id(self.latent @ self.cod) @ other.symbolic_channel)
        # This is problematic because while in ML we do not need to remember what generated what here we do.
        return OpenModel(dom=dom, cod=cod, latent=latent_space, log_prob_channel=log_prob_channel, symbolic_channel=symbolic_channel, depth=dept)
            
# @dataclass
# class BayesianLens(OpenModel):
#     inversion: Optional[Box] = None

#     def __post_init__(self):
#         # Call parent post-init.
#         super().__post_init__()
#         if self.inversion is None:
#             # Here we define a trivial inversion as a lambda returning an OpenModel with swapped domain/cod.
#             self.inversion = lambda prior: OpenModel(self.cod, self.dom, self.latent, self.channel)

# @dataclass
# class StatGame(BayesianLens):
#     loss: Optional[Box] = None
#     loss_name: str = "loss"

# @dataclass
# class ExponentialFamily(OpenModel):
#     natural_param: Optional[Ty] = field(default_factory=lambda: Ty())
#     expectation_params: Optional[Box] = None
#     # Optionally, you could also store a NumPyro distribution in prob_density here.


# # Here we define a GaussianChannel which is an exponential-family channel.
# @dataclass
# class GaussianChannel(ExponentialFamily):
#     # mu: float = 0.0
#     # sigma: float = 1.0
#     channel =lambda mu, sigma:  dist.Normal(mu,sigma)
    
    
#     def __post_init__(self):
#         # Call parent's post_init to initialize channel and others.
#         super().__post_init__()
#         # Create a NumPyro Normal distribution instance with the given parameters.
#         self.prob_density = distribution=dist.Normal(self.mu, self.sigma)
        

#     @property
#     def density(self):
#         # Return the log_prob function of the distribution.
#         return self.prob_density.distribution.log_prob

# # Usage example:
# gc = GaussianChannel(dom=R, cod=N, mu=0.0, sigma=1.0)
# # Now, calling gc.density on a value computes the log density:
# print(gc.density(1.0))  # log_prob(1.0) for Normal(0,1)
if __name__ == "__main__":
    def sum(x,y):
        print(y)
        return x+y
    channelized_sum = lambda x: partial(sum,x)
    
    sum_model = OpenModel(dom=R, cod=R, log_prob_channel=channelized_sum)
    chaining_sums = sum_model >> sum_model
    chaining_sums.log_prob_channel(1)(1,2)
    chaining_sums = chaining_sums >> sum_model
    # chaining_sums.symbolic_channel.draw()
    print(chaining_sums.log_prob_channel(1)(1,2,3))
    