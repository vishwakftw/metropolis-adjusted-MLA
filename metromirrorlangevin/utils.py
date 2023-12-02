import numpy as np
import ot
import torch

torch.set_default_dtype(torch.float64)

from matplotlib import animation as anim
from matplotlib import pyplot as plt
from scipy.stats import ortho_group
from typing import Callable, Union

from .barriers import (
    Barrier,
    BoxBarrier,
    EllipsoidBarrier,
    PolytopeBarrier,
    SimplexBarrier,
)
from .potentials import Potential

__all__ = [
    "get_chol",
    "ot_distance",
    "define_ellipsoid",
    "define_box",
    "draw_boundary",
    "AnimationWrapper",
]


def get_chol(is_diagonal: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    if is_diagonal:

        def CHOL(batch_of_matrices: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(batch_of_matrices)  # return diagonal

    else:

        def CHOL(batch_of_matrices: torch.Tensor) -> torch.Tensor:
            L, info = torch.linalg.cholesky_ex(batch_of_matrices, upper=False)
            if info.sum(0) == 0:
                return L

            max_iter = 3  # 3 maximum perturbations
            curr_pert = 1e-06  # perturbation values, increased multiplicatively
            mask = info.to(dtype=torch.bool)
            # we only need to deal with the failed ones below
            copy_failed = batch_of_matrices[mask].clone()
            while info.sum() != 0 and max_iter > 0:
                # do inplace to save memory
                copy_failed.diagonal(dim1=-2, dim2=-1).add_(curr_pert)
                # perform cholesky on the perturbed matrices
                Lfail, info = torch.linalg.cholesky_ex(
                    copy_failed, check_errors=max_iter == 1, upper=False
                )
                curr_pert *= 10  # increase curr_pert by 10 factor
                max_iter -= 1  # reduce max_iter
            # at the end, if there's no erroring out, replace the failed ones
            L[mask] = Lfail
            return L

    return CHOL


def ot_distance(points_a: torch.Tensor, points_b: torch.Tensor, bias: float):
    Na = points_a.shape[0]
    Nb = points_b.shape[0]
    points_a = points_a.numpy()
    points_b = points_b.numpy()
    cost_matrix = ot.utils.dist(points_a, points_b)
    a = np.ones(Na) / Na
    b = np.ones(Nb) / Nb
    loss = ot.sinkhorn2(a=a, b=b, M=cost_matrix, reg=0.001, stopThr=1e-06)
    return loss - bias


def define_ellipsoid(dimension: int, random_seed: int, condition_number: int = 1):
    if dimension < 1:
        raise ValueError("Invalid dimension")
    if condition_number < 1:
        raise ValueError("Invalid condition number")
    np.random.seed(random_seed)
    U = torch.from_numpy(ortho_group.rvs(dim=dimension)).to(
        dtype=torch.get_default_dtype()
    )
    L = torch.linspace(1, condition_number, dimension)
    return {"rot": U, "eigvals": L}


def define_box(dimension: int, condition_number: int = 1):
    if dimension < 1:
        raise ValueError("Invalid dimension")
    if condition_number < 1:
        raise ValueError("Invalid condition number")
    bounds = torch.ones(dimension)
    bounds[0] = 1 / condition_number
    return bounds


def define_polytope(dimension: int, num_constraints: int):
    if dimension < 1:
        raise ValueError("Invalid dimension")
    raise NotImplementedError


def draw_ellipsoid_boundary(ellipsoid, ax=None):
    # only for 2-D
    theta = np.arange(0, 360)
    x = np.cos(np.deg2rad(theta))
    y = np.sin(np.deg2rad(theta))
    x = x / np.sqrt(ellipsoid["eigvals"][0])
    y = y / np.sqrt(ellipsoid["eigvals"][1])
    p = np.stack([x, y], axis=1)
    p = ellipsoid["rot"] @ np.expand_dims(p, -1)
    p = p.squeeze()
    if ax is None:
        plt.xlim(-1 - 0.1, 1 + 0.1)
        plt.ylim(-1 - 0.1, 1 + 0.1)
        plt.plot(p[:, 0], p[:, 1], "k--", lw=3.0)
    else:
        ax.set_xlim(-1 - 0.1, 1 + 0.1)
        ax.set_ylim(-1 - 0.1, 1 + 0.1)
        ax.plot(p[:, 0], p[:, 1], "k--", lw=3.0)


def draw_box_boundary(bounds, ax=None):
    # only for 2-D
    bounds = bounds.numpy()
    if ax is None:
        plt.axhline(
            y=-bounds[1], xmin=-bounds[0], xmax=bounds[0], c="k", ls="--", lw=3.0
        )
        plt.axhline(
            y=bounds[1], xmin=-bounds[0], xmax=bounds[0], c="k", ls="--", lw=3.0
        )
        plt.axvline(
            x=-bounds[0], ymin=-bounds[1], ymax=bounds[1], c="k", ls="--", lw=3.0
        )
        plt.axvline(
            x=bounds[0], ymin=-bounds[1], ymax=bounds[1], c="k", ls="--", lw=3.0
        )
    else:
        ax.axhline(
            y=-bounds[1], xmin=-bounds[0], xmax=bounds[0], c="k", ls="--", lw=3.0
        )
        ax.axhline(y=bounds[1], xmin=-bounds[0], xmax=bounds[0], c="k", ls="--", lw=3.0)
        ax.axvline(
            x=-bounds[0], ymin=-bounds[1], ymax=bounds[1], c="k", ls="--", lw=3.0
        )
        ax.axvline(x=bounds[0], ymin=-bounds[1], ymax=bounds[1], c="k", ls="--", lw=3.0)


def draw_polytope_boundary(polytope_inst, ax=None):
    import polytope

    p = polytope.Polytope(A=polytope_inst["A"].numpy(), b=polytope_inst["b"].numpy())
    bb = p.bounding_box
    if ax is None:
        plt.xlim(bb[0][0] - 0.1 * abs(bb[0][0]), bb[1][0] + 0.1 * abs(bb[1][0]))
        plt.ylim(bb[0][1] - 0.1 * abs(bb[0][1]), bb[1][1] + 0.1 * abs(bb[1][0]))
    else:
        ax.set_xlim(bb[0][0] - 0.1 * abs(bb[0][0]), bb[1][0] + 0.1 * abs(bb[1][0]))
        ax.set_ylim(bb[0][1] - 0.1 * abs(bb[0][1]), bb[1][1] + 0.1 * abs(bb[1][0]))

    p.plot(color="none", ax=ax)


class AnimationWrapper:
    def __init__(
        self, points_to_plot: torch.Tensor, potential: Union[Potential, Barrier]
    ):
        self.points_to_plot = points_to_plot
        self.potential = potential

        fig, ax = plt.subplots(figsize=(6, 6))  # 1 row, 1 columns
        self.fig = fig
        self.ax = ax
        if isinstance(potential, EllipsoidBarrier):
            draw_ellipsoid_boundary(ellipsoid=potential.ellipsoid, ax=ax)
        elif isinstance(potential, BoxBarrier):
            self.ax.set_xlim(-potential.bounds[0] * 1.01, potential.bounds[0] * 1.01)
            self.ax.set_ylim(-potential.bounds[1] * 1.01, potential.bounds[1] * 1.01)
            draw_box_boundary(bounds=potential.bounds, ax=ax)
        elif isinstance(potential, PolytopeBarrier):
            draw_polytope_boundary(polytope_inst=potential.polytope, ax=ax)
        elif isinstance(potential, SimplexBarrier):
            self.ax.set_xlim(0.0 - 0.01, 1.0 + 0.01)
            self.ax.set_ylim(0.0 - 0.01, 1.0 + 0.01)
        canvas = ax.scatter([], [])
        self.canvas = canvas

    def animate(self, itr):
        curr_points = self.points_to_plot[itr]
        if itr % 100 == 0 or itr == len(self.points_to_plot):
            self.ax.set_title(f"Iteration {itr}")
        self.canvas.set_offsets(curr_points.numpy())

        try:
            colours = np.array(["black", "green"])
            in_good_region = self.potential.boundary_to_interior_half(
                curr_points
            ).numpy()
            self.canvas.set_color(colours[in_good_region.astype(int)])
        except NotImplementedError:
            pass

        return (self.canvas,)

    def generate(self, filename: str, interval: int):
        animation = anim.FuncAnimation(
            self.fig,
            self.animate,
            frames=len(self.points_to_plot),
            blit=True,
            interval=interval,
        )
        animation.save(filename=filename)
