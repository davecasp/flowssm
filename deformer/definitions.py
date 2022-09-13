import torch
import torch.optim as optim
import torch.nn as nn

LOSSES = {
    "l1": torch.nn.L1Loss(),
    "l2": torch.nn.MSELoss(),
    "huber": torch.nn.SmoothL1Loss(),
}

OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "rmsprop": optim.RMSprop,
}

SOLVERS = [
    "dopri5",
    "adams",
    "euler",
    "midpoint",
    "rk4",
    "explicit_adams",
    "fixed_adams",
    "bosh3",
    "adaptive_heun",
    "tsit5",
]

LOSSES = {
    "l1": torch.nn.L1Loss(),
    "l2": torch.nn.MSELoss(),
    "huber": torch.nn.SmoothL1Loss(),
}

REDUCTIONS = {
    "mean": lambda x: torch.mean(x, axis=-1),
    "max": lambda x: torch.max(x, axis=-1)[0],
    "min": lambda x: torch.min(x, axis=-1)[0],
    "sum": lambda x: torch.sum(x, axis=-1),
}

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "leakyrelu": nn.LeakyReLU(),
}
