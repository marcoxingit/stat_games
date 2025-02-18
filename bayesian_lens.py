from discopy.cat import Category
from discopy.markov import Ty, Box, Copy, Diagram, Id, Swap
from typing import Optional
from functools import reduce




R = Ty("R")
N = Ty("N")


def NACopy(x: Ty, n: int = 2) -> Box:
    if x.is_atomic:
        return Copy(x, n)
    elif len(x) == 2 and n == 2:
        return Copy(x[0], n) @ Copy(x[1], n) >> Id(x[0]) @ NASwap(x[0], x[1]) @ Id(x[1])
    return Box("Copy", x, x**n)


def Sum(n=2) -> Box:
    x = R**n
    return Box("Sum", x, R)


def NASwap(x: Ty, y: Ty) -> Box:
    if x.is_atomic and y.is_atomic:
        return Swap(x, y)
    return Box("Swap", x @ y, y @ x)


def Expectation(x: Ty) -> Box:
    return Box("Expectation", x, R)


class BayesianLens:

    def __init__(
        self,
        dom: Ty,
        cod: Ty,
        apex: Optional[Ty] = None,
        left_projection: Optional[Box] = None,
        right_projection: Optional[Box] = None,
        channel: Optional[Box] = None,
        smc_rep: Optional[Box] = None,
        inversion: Optional[Box] = None,
        loss: Optional[Box] = None,
        entropy: Optional[Box] = None,
        channel_name: str = "channel",
        inversion_name: str = "inversion",
        loss_name: str = "loss",
    ):
        self.dom = dom
        self.cod = cod
        self.apex = apex if apex is not None else dom @ cod
        self.channel_name = channel_name 
        self.loss_name = loss_name   
        self.channel = (
            Box(channel_name, self.dom, self.apex) if channel is None else channel
        )
        self.left_projection = (
            Box(f"left_proj_{channel_name}", self.apex, dom)
            if left_projection is None
            else left_projection
        )

        self.right_projection = (
            Box(f"right_proj_{channel_name}", self.apex, cod)
            if right_projection is None
            else right_projection
        )

        self.inversion = (
            Box(inversion_name, self.dom @ self.cod, self.apex)
            if inversion is None
            else inversion
        )
        loss_name = loss_name if loss_name is not None else "loss " + channel_name
        self.loss = Box(loss_name, self.apex, R) if loss is None else loss
        self.entropy = (
            Box("entropy " + channel_name, self.dom @ self.cod, R)
            if entropy is None
            else entropy
        )
        self.smc_rep = (
            Box(channel_name, self.dom, self.cod) if smc_rep is None else smc_rep
        )

        assert (
            self.channel.dom == self.dom
        ), f"Channel domain {self.channel.dom} does not match domain {self.dom}"
        assert (
            self.channel.cod == self.apex
        ), f"Channel codomain {self.channel.cod} does not match apex {self.apex}"
        assert (
            self.inversion.dom == self.dom @ self.cod
        ), f"Inversion domain {self.inversion.dom} does not match {self.dom @ self.cod}"
        assert (
            self.inversion.cod == self.apex
        ), f"Inversion codomain {self.inversion.cod} does not match apex {self.apex}"
        assert (
            self.loss.dom == self.apex
        ), f"Loss domain {self.loss.dom} does not match apex {self.apex}"
        assert self.loss.cod == Ty(
            "R"
        ), f"Loss codomain {self.loss.cod} does not match R"
        assert (
            self.entropy.dom == self.dom @ self.cod
        ), f"Entropy domain {self.entropy.dom} does not match {self.dom @ self.cod}"
        assert self.entropy.cod == Ty(
            "R"
        ), f"Entropy codomain {self.entropy.cod} does not match R"

    def __matmul__(self, other):

        dom = self.dom @ other.dom
        cod = self.cod @ other.cod
        apex = (
            self.dom
            @ other.dom
            @ self.apex[len(self.dom) : -len(self.cod)]
            @ other.apex[len(other.dom) : -len(other.cod)]
            @ self.cod
            @ other.cod
        )

        # Create a permutation diagram that swaps elements according to the list
        # Calculate lengths
        i1 = len(self.dom)
        i2 = len(other.dom)
        o1 = len(self.cod)
        o2 = len(other.cod)
        m1 = len(self.apex[i1:-o1])
        m2 = len(other.apex[i2:-o2])

        # Calculate starting positions in original tensor
        s2 = len(self.apex)  # start of other.apex

        # Create the permutation indices
        perm_indices = (
            # self.dom
            list(range(i1))
            +
            # other.dom
            list(range(s2, s2 + i2))
            +
            # self.apex[middle]
            list(range(i1, i1 + m1))
            +
            # other.apex[middle]
            list(range(s2 + i2, s2 + i2 + m2))
            +
            # self.cod
            list(range(i1 + m1, s2))
            +
            # other.cod
            list(range(s2 + i2 + m2, len(self.apex) + len(other.apex)))
        )

        big_swap = Diagram.permutation(perm_indices, self.apex @ other.apex)

        # Compute inverse permutation
        inverse_perm = [0] * len(perm_indices)
        for i, p in enumerate(perm_indices):
            inverse_perm[p] = i

        big_swap_inverse = Diagram.permutation(inverse_perm, apex)

        channel = (self.channel @ other.channel) >> big_swap
        inversion = (
            (Id(self.dom) @ NASwap(other.dom, self.cod) @ Id(other.cod))
            >> (self.inversion @ other.inversion)
            >> big_swap
        )
        entropy = (
            (Id(self.dom) @ NASwap(other.dom, self.cod) @ Id(other.cod))
            >> (self.entropy @ other.entropy)
            >> Sum()
        )

        left_projection = (
            big_swap_inverse >> self.left_projection @ other.left_projection
        )
        right_projection = (
            big_swap_inverse >> self.right_projection @ other.right_projection
        )

        loss = big_swap_inverse >> self.loss @ other.loss >> Sum()

        return BayesianLens(
            dom=dom,
            cod=cod,
            apex=apex,
            left_projection=left_projection,
            right_projection=right_projection,
            channel=channel,
            smc_rep=self.smc_rep @ other.smc_rep,
            inversion=inversion,
            loss=loss,
            entropy=entropy,
        )

    def __pow__(self, n):
        if n < 1:
            raise ValueError("Exponent must be a positive integer.")
        return reduce(lambda x, y: x @ y, [self] * n)

    def __rshift__(self, other):
        assert (
            self.cod == other.dom
        ), f"Codomain of {self} and domain of {other} do not match{self.cod} != {other.dom}"
        dom = self.dom
        cod = other.cod

        apex = self.apex @ other.apex[len(other.dom) :]

        left_projection = (
            Box(
                f"PB_l({self.right_projection} | {other.left_projection})",
                apex,
                self.apex,
            )
            >> self.left_projection
        )
        right_projection = (
            Box(
                f"PB_r({self.right_projection} | {other.left_projection})",
                apex,
                other.apex,
            )
            >> other.right_projection
        )
        channel = self.channel >> (Id(self.apex[: -(len(other.dom))]) @ other.channel)

        inversion = (
            (NACopy(self.dom, n=2) @ Id(other.cod))
            >> (Id(self.dom) @ self.channel @ Id(other.cod))
            >> (Id(self.dom) @ self.right_projection @ Id(other.cod))
            >> (Id(self.dom) @ other.inversion)
            >> (self.inversion @ Id(other.apex[len(other.dom) :]))
        )

        assert inversion.cod == apex, f"{inversion.cod} != {apex}"
        assert inversion.dom == dom @ cod, f"{inversion.dom} != {dom @ cod}"

        loss = (
            Id(self.apex[: len(self.apex) - len(other.dom)])
            @ NACopy(self.cod, n=2)
            @ Id(other.apex[len(other.dom) :])
            >> (self.loss @ other.loss)
            >> Sum()
        )

        assert loss.dom == apex

        entropy = (
            (NACopy(self.dom, n=2) @ Id(other.cod))
            >> (Id(self.dom) @ self.channel @ Id(other.cod))
            >> (Id(self.dom) @ self.right_projection @ Id(other.cod))
            >> (Id(self.dom) @ NACopy(self.cod @ other.cod))
            >> (Id(self.dom) @ other.inversion @ other.entropy)
            >> (Id(self.dom) @ other.left_projection @ Id(R))
            >> (self.entropy @ Id(R))
            >> (Expectation(R) @ Id(R))
            >> Sum()
        )
        assert entropy.dom == self.dom @ other.cod

        return BayesianLens(
            dom=dom,
            cod=cod,
            apex=apex,
            left_projection=left_projection,
            right_projection=right_projection,
            channel=channel,
            smc_rep=self.smc_rep >> other.smc_rep,
            inversion=inversion,
            loss=loss,
            entropy=entropy,
        )

    def to_diagram(self) -> Diagram:
        """Convert the lens to a string diagram representation."""
        # Create boxes for each component
       
        
        # Create the diagram by composing the boxes
        diagram = self.smc_rep
        
        
        return diagram
    
    def draw(self, **kwargs):
        """Draw the lens as a string diagram."""
        return self.to_diagram().draw(**kwargs)
    
    @classmethod
    def Swap(cls, x:Ty, y:Ty) -> Box:
        return BayesianLens(
            dom=x @ y,
            cod=y @ x,
            channel=NACopy(x@y)>> (Id(x@y) @ NASwap(x,y)),
        )
        
    @classmethod
    def Copy(cls, x: Ty, n: int = 2) -> Box:
        return BayesianLens(
            dom=x,
            cod=x**n,
            channel=NACopy(x, n + 1) >> Id(x ** (n + 1)),
        )
        
        
        
    @classmethod
    def Id(cls, x: Ty):
        return BayesianLens(dom=x, cod=x, channel=NACopy(x))