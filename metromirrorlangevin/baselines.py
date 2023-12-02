import math
import torch

torch.set_default_dtype(torch.float64)

from tqdm.auto import tqdm

from .barriers import Barrier
from .potentials import Potential
from .utils import get_chol


class MirrorLangevinSampler:
    def __init__(
        self,
        barrier: Barrier,
        potential: Potential,
        num_samples: int,
    ):
        self.barrier = barrier
        self.potential = potential
        self.num_samples = num_samples
        self.CHOL = get_chol(is_diagonal=barrier.diag_hess)

    def set_initial_particles(self, particles: torch.Tensor):
        # NOTE: these are particles in the primal space, not the dual space.
        if particles.shape[0] != self.num_samples:
            raise ValueError(
                "Initialisation doesn't contain the same number "
                "of particles as expected."
            )
        self.initial_particles = particles.clone()

    def mix(
        self,
        num_iters: int,
        stepsize: float,
        return_particles: bool = False,
        no_progress: bool = True,
    ):
        particles = self.initial_particles.clone()
        particles_dual = self.barrier.gradient(particles)
        particles_tracker = [particles]

        full_covariance = self.barrier.hessian(particles)
        # if diag_hess, n x d, else n x d x d
        chol_covariance = self.CHOL(full_covariance)
        # if diag_hess, n x d, else n x d x d
        grad_potential = self.potential.gradient(particles)
        for _ in tqdm(range(num_iters), disable=no_progress):
            noise = torch.randn_like(particles_dual)
            if self.barrier.diag_hess:
                # scaled_noise = Lx @ xi = Lx * xi
                scaled_noise = chol_covariance * noise
            else:
                # scaled_noise = Lx @ xi
                scaled_noise = torch.einsum("bij,bj->bi", chol_covariance, noise)

            particles_dual = (
                particles_dual
                - stepsize * grad_potential
                + math.sqrt(2 * stepsize) * scaled_noise
            )
            particles = self.barrier.inverse_gradient(particles_dual)
            full_covariance = self.barrier.hessian(particles)
            chol_covariance = self.CHOL(full_covariance)
            particles_tracker.append(particles)
        if return_particles:
            return torch.stack(particles_tracker)
        else:
            return particles
