import torch

torch.set_default_dtype(torch.float64)


class Potential:
    """
    Base class for Potentials
    """

    def __init__(self, *args, **kwargs):
        pass

    def value(self, x: torch.Tensor):
        raise NotImplementedError

    def gradient(self, x: torch.Tensor):
        raise NotImplementedError


class DirichletPotential:
    """
    Dirichlet Potential
    """

    def __init__(self, alpha: torch.Tensor):
        self.alpha = alpha
        self.dimension = alpha.shape[0] - 1  # the alpha is of length d + 1

    def _safe_interior(self, x: torch.Tensor, squeeze_last_dim: bool):
        return torch.clamp_min(
            1 - torch.sum(x, dim=-1, keepdim=not squeeze_last_dim),
            min=1e-08,
        )

    def value(self, x: torch.Tensor):
        return (
            -torch.sum(self.alpha[:-1] * torch.log(x), dim=-1)
            - torch.log(self._safe_interior(x=x, squeeze_last_dim=True))
            * self.alpha[-1]
        )

    def gradient(self, x: torch.Tensor):
        return -self.alpha[:-1] / x + self.alpha[-1] / self._safe_interior(
            x=x, squeeze_last_dim=False
        )


class BayesianLogisticRegressionPotential:
    """
    Bayesian Logistic Regression Potential
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X  # shape N x d
        self.y = y  # shape N
        self.dimension = X.shape[1]

    def _inner_prod(self, thetas: torch.Tensor):
        return torch.einsum("bi,ni->bn", thetas, self.X)

    def value(self, thetas: torch.Tensor):
        inner_prod = self._inner_prod(thetas)
        return -torch.sum(
            inner_prod * self.y - torch.log(1 + torch.exp(inner_prod)), dim=-1
        )

    def gradient(self, thetas: torch.Tensor):
        inner_prod = self._inner_prod(thetas)
        y_minus_sigmoid = self.y - inner_prod.sigmoid_()  # b x N
        return -torch.sum(y_minus_sigmoid.unsqueeze(dim=-1) * self.X, dim=-2)
