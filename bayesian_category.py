from discopy.cat import Category
from discopy.markov import Ty, Box,  Diagram, Id, Swap, Category as MarkovCategory
from .bayesian_lens import BayesianLens, R, N, NACopy, NASwap

class BayesianLensCategory(MarkovCategory):
    """Category where:
    - Objects are types (Ty)
    - Morphisms are BayesianLens instances
    """

    def __init__(self):
        super().__init__("BayesianLens")
        self.ob, self.ar = Ty, BayesianLens

    @staticmethod
    def id(x: Ty) -> BayesianLens:
        """Identity morphism for a type"""
        return BayesianLens.Id(x)

    @staticmethod
    def compose(f: BayesianLens, g: BayesianLens) -> BayesianLens:
        """Sequential composition of Bayesian lenses"""
        if f.cod != g.dom:
            raise ValueError(
                f"Cannot compose lenses with mismatched types: {f.cod} ≠ {g.dom}"
            )
        return f >> g

    @staticmethod
    def tensor(f: BayesianLens, g: BayesianLens) -> BayesianLens:
        """Parallel composition (tensor product) of Bayesian lenses"""
        return f @ g

    @staticmethod
    def copy(x: Ty) -> BayesianLens:
        """Copy morphism for a type"""
        return NACopy(x)

    @staticmethod
    def swap(x: Ty, y: Ty) -> BayesianLens:
        """Swap morphism for two types"""
        return BayesianLens.Swap(x,y)

# # Create the category instance
# bayesian_category = BayesianLensCategory()
