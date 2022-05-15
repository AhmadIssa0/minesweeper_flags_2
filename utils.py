import torch
from typing import Callable


def create_2d_tensor_from_func(rows, cols, fn, **kwargs):
    data = [[fn(i, j) for j in range(cols)] for i in range(rows)]
    return torch.tensor(data=data, **kwargs)
