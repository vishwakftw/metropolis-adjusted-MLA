import numpy as np
import pandas as pd
import torch

torch.set_default_dtype(torch.float64)

from argparse import ArgumentParser
from time import time
from tqdm.auto import tqdm

from metromirrorlangevin.algorithms.mirror_algorithms import GeneralMAMLASampler
from metromirrorlangevin.barriers import SimplexBarrier
from metromirrorlangevin.potentials import DirichletPotential
from metromirrorlangevin.utils import ot_distance


def generate_compute_ranges(max_iters: int):
    section_one = np.arange(0, int(max_iters * 0.05) + 1)
    section_two = np.arange(
        int(max_iters * 0.05) + 1,
        int(max_iters * 0.30) + 1,
        min(int(max_iters * 0.01), 5),
    )
    section_three = np.arange(
        int(max_iters * 0.30) + 1, max_iters, min(int(max_iters * 0.05), 20)
    )
    all_sections = np.concatenate([section_one, section_two, section_three])
    if not np.isin(max_iters, all_sections):
        all_sections = np.append(all_sections, [max_iters])
    return all_sections


def run_sampling(
    dimension: int,
    num_iters: int,
    stepsize: float,
    run_index: int,
    num_particles: int,
    progress_file: str,
    loss_progress_file: str,
    debug: bool = False,
    show_tqdm: bool = False,
):
    # create a random seed based on the values
    random_seed = int(dimension * 1729 + run_index)
    # same random_seed for a specification of stepsize
    # this is to better judge the effects of these on the same sampling problem.
    # same random_seed for a specification of num_particles as well.

    # add 1 because the dimensionality of Dirichlet is (d + 1)
    alpha = torch.full((dimension + 1,), 3.0)  # not 2, but 3

    # reseed the generator
    torch.manual_seed(random_seed)

    barrier = SimplexBarrier(dimension=dimension)
    potential = DirichletPotential(alpha=alpha)
    # initialise from [1/2d, 1/2d ....] +/- epsilon * [1/24d, 1/24d, ....]
    initial_particles = torch.full(
        size=(num_particles, dimension), fill_value=1 / (2 * dimension)
    )
    initial_particles += torch.rand(num_particles, dimension) / (12 * dimension) - 1 / (
        24 * dimension
    )
    require_losses = loss_progress_file != "NA"

    dirichlet_sampler = GeneralMAMLASampler(
        barrier=barrier, potential=potential, num_samples=num_particles
    )
    dirichlet_sampler.set_initial_particles(initial_particles)
    time_start = time()
    particles, rejection_masks = dirichlet_sampler.mix(
        num_iters=num_iters,
        stepsize=stepsize,
        return_particles=debug or require_losses,
        no_progress=not show_tqdm,
    )
    time_end = time()
    total_time = time_end - time_start
    acceptance_full = num_particles - torch.sum(rejection_masks, dim=-1)

    # compute avg_acceptance after the first tenth of iterations
    avg_acceptance = torch.mean(acceptance_full[int(0.1 * num_iters) :] / num_particles)

    # construct last coordinate for convenience
    particles_sum = torch.sum(particles, dim=-1, keepdim=True)
    particles = torch.cat([particles, particles_sum.mul_(-1).add_(1.0)], dim=-1)

    # ground truth samples from PyTorch
    # note that our alpha is a bit different
    # the concentration parameter is 1 + alpha
    dir_dist = torch.distributions.Dirichlet(concentration=alpha + 1)
    ground_truth = dir_dist.sample((num_particles,))
    bias = ot_distance(ground_truth, ground_truth, bias=0)

    if require_losses:
        compute_ranges = generate_compute_ranges(num_iters)
        for itr in tqdm(compute_ranges, disable=not show_tqdm):
            itr_particles = particles[itr]
            loss = ot_distance(itr_particles, ground_truth, bias=bias)
            desc_string = f"{dimension},{itr},{stepsize},{run_index},{loss:.6f}"
            with open(loss_progress_file, "a") as f:
                f.write(desc_string + "\n")
    else:
        loss = ot_distance(particles, ground_truth, bias=bias)

    avg_acceptance = avg_acceptance.item()

    desc_string = (
        f"{dimension},{num_iters},"
        f"{stepsize},{run_index},{num_particles},"
        f"{total_time:.4f},{avg_acceptance:.6f},{loss:.6f}"
    )
    with open(progress_file, "a") as f:
        f.write(desc_string + "\n")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--dimension", type=int, help="Dimension")
    p.add_argument("--num_iters", type=int, help="Number of iterations")
    p.add_argument("--stepsize", type=float, help="Step size")
    p.add_argument("--run_index", type=int, help="Run index for multiple runs")
    p.add_argument("--num_particles", type=int, help="Number of samples")
    p.add_argument(
        "--show_tqdm", action="store_true", help="Toggle for showing progress bar"
    )
    p.add_argument(
        "--progress_file",
        default="progress.txt",
        type=str,
        help="Configuration to store results",
    )
    p.add_argument(
        "--loss_progress_file",
        default=None,
        type=str,
        help="Configuration to store losses computed every few iterations",
    )
    args = p.parse_args()
    args = vars(args)
    run_sampling(**args)
