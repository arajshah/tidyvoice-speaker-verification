import torch
from torch import nn
from torch.autograd import Function


class _GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg().mul(ctx.lambd), None


class GRL(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.register_buffer("_lambda", torch.tensor(float(lambd), dtype=torch.float32))

    def get_lambda(self) -> float:
        return float(self._lambda.item())

    def set_lambda(self, lambd: float) -> "GRL":
        self._lambda.fill_(float(lambd))
        return self

    def forward(self, x: torch.Tensor, lambd: float | None = None) -> torch.Tensor:
        l = self.get_lambda() if lambd is None else float(lambd)
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return x
        return _GradientReversalFn.apply(x, l)
