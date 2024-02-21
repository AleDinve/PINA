import torch

try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import (
        _LRScheduler as LRScheduler,
    )  # torch < 2.0

import sys
from torch.optim.lr_scheduler import ConstantLR

from .solver import SolverInterface
from ..label_tensor import LabelTensor
from ..utils import check_consistency
from ..loss import LossInterface
from ..problem import InverseProblem
from torch.nn.modules.loss import _Loss

from .stochastic_solver import StochasticSolverInterface
from .pinn import PINN



class PCPINN(StochasticSolverInterface,PINN):
    def __init__(
        self,
        problem,
        model,
        polynomials,
        extra_features=None,
        loss=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.001},
        scheduler=ConstantLR,
        scheduler_kwargs={"factor": 1, "total_iters": 0},
    ):
        super(StochasticSolverInterface).__init__(problem=problem)
        super(PINN).__init__(
            models=[model],
            optimizers=[optimizer],
            optimizers_kwargs=[optimizer_kwargs],
            extra_features=extra_features,
            loss = loss,
            scheduler=scheduler,
            scheduler_kwargs= scheduler_kwargs
        )