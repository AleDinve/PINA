''' Stochastic solver module. '''


import pytorch_lightning
import torch
from abc import ABCMeta, abstractmethod
from .solver import SolverInterface
from ..utils import check_consistency
from ..problem import StochasticProblem


class StochasticSolverInterface(SolverInterface, ABCMeta):
    def __init__(
        self,
        models,
        problem,
        optimizers,
        optimizers_kwargs,
        extra_features=None
    ):
        """
    :param models: A torch neural network model instance.
    :type models: torch.nn.Module
    :param problem: A problem definition instance.
    :type problem: AbstractProblem
    :param list(torch.optim.Optimizer) optimizer: A list of neural network optimizers to
        use.
    :param list(dict) optimizer_kwargs: A list of optimizer constructor keyword args.
    :param list(torch.nn.Module) extra_features: The additional input
        features to use as augmented input. If ``None`` no extra features
        are passed. If it is a list of :class:`torch.nn.Module`, the extra feature
        list is passed to all models. If it is a list of extra features' lists,
        each single list of extra feature is passed to a model.
    """
        check_consistency(problem,StochasticProblem)
        super(SolverInterface).__init__(
        models=models,
        problem=problem,
        optimizers=optimizers,
        optimizers_kwargs=optimizers_kwargs,
        extra_features=extra_features
        )

        @abstractmethod
        def mean(self):
            pass
        
        @abstractmethod
        def variance(self):
            pass

