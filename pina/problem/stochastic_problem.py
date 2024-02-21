"""Module for the StochasticProblem class"""

from abc import abstractmethod

from .spatial_problem import SpatialProblem 
from .timedep_problem import TimeDependentProblem


class StochasticProblem(SpatialProblem, TimeDependentProblem):

    @abstractmethod
    def stochastic_domain(self):
        """
        The stochastic domain of the problem.
        """
        pass

    @property
    def stochastic_variables(self):
        """
        The stochastic input variables of the problem.
        """
        return self.stochastic_domain.variables
    
