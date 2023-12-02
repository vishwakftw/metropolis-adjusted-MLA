import numpy as np
import pandas as pd
import torch

torch.set_default_dtype(torch.float64)

from argparse import ArgumentParser
from time import time
from tqdm.auto import tqdm

from metromirrorlangevin.algorithms.mirror_algorithms import GeneralMAMLASampler
from metromirrorlangevin.barriers import SimplexBarrier
from metromirrorlangevin.baselines import MirrorLangevinSampler
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
    stepsize_mamla: float,
    stepsize_mla: float,
    run_index: int,
    num_particles: int,
    progress_file: str,
    loss_progress_file: str,
    debug: bool = False,
    show_tqdm: bool = False,
):
    # create a random seed based on the values
    random_seed = int(dimension * 3141 + run_index)
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

    # MAMLA first
    dirichlet_sampler = GeneralMAMLASampler(
        barrier=barrier, potential=potential, num_samples=num_particles
    )
    dirichlet_sampler.set_initial_particles(initial_particles)
    time_start = time()
    particles_mamla, rejection_masks = dirichlet_sampler.mix(
        num_iters=num_iters,
        stepsize=stepsize_mamla,
        return_particles=debug or require_losses,
        no_progress=not show_tqdm,
    )
    time_end = time()
    total_time_mamla = time_end - time_start
    acceptance_full = num_particles - torch.sum(rejection_masks, dim=-1)

    # compute avg_acceptance after the first tenth of iterations
    avg_acceptance = torch.mean(acceptance_full[int(0.1 * num_iters) :] / num_particles)

    # construct last coordinate for convenience
    particles_mamla_sum = torch.sum(particles_mamla, dim=-1, keepdim=True)
    particles_mamla = torch.cat(
        [particles_mamla, particles_mamla_sum.mul_(-1).add_(1.0)], dim=-1
    )

    # MLA second
    dirichlet_sampler = MirrorLangevinSampler(
        barrier=barrier, potential=potential, num_samples=num_particles
    )
    dirichlet_sampler.set_initial_particles(initial_particles)
    time_start = time()
    particles_mla = dirichlet_sampler.mix(
        num_iters=num_iters,
        stepsize=stepsize_mla,
        return_particles=debug or require_losses,
        no_progress=not show_tqdm,
    )
    time_end = time()
    total_time_mla = time_end - time_start

    # construct last coordinate for convenience
    particles_mla_sum = torch.sum(particles_mla, dim=-1, keepdim=True)
    particles_mla = torch.cat(
        [particles_mla, particles_mla_sum.mul_(-1).add_(1.0)], dim=-1
    )

    # ground truth samples from PyTorch
    # note that our alpha is a bit different
    # the concentration parameter is 1 + alpha
    dir_dist = torch.distributions.Dirichlet(concentration=alpha + 1)
    ground_truth = dir_dist.sample((num_particles,))
    bias = ot_distance(ground_truth, ground_truth, bias=0)

    for alg, particles, total_time, stepsize in zip(
        ["MAMLA", "MLA"],
        [particles_mamla, particles_mla],
        [total_time_mamla, total_time_mla],
        [stepsize_mamla, stepsize_mla],
    ):
        if require_losses:
            compute_ranges = generate_compute_ranges(num_iters)
            for itr in tqdm(
                compute_ranges, disable=not show_tqdm, desc=f"Computing for {alg}"
            ):
                itr_particles = particles[itr]
                loss = ot_distance(itr_particles, ground_truth, bias=bias)
                desc_string = (
                    f"{alg},{dimension},{itr},{stepsize},{run_index},{loss:.6f}"
                )
                with open(loss_progress_file, "a") as f:
                    f.write(desc_string + "\n")
        else:
            loss = ot_distance(particles, ground_truth, bias=bias)

        if alg == "MAMLA":
            avg_acceptance = avg_acceptance.item()
        else:
            avg_acceptance = 1.0

        desc_string = (
            f"{alg},{dimension},{num_iters},"
            f"{stepsize},{run_index},{num_particles},"
            f"{total_time:.4f},{avg_acceptance:.6f},{loss:.6f}"
        )
        with open(progress_file, "a") as f:
            f.write(desc_string + "\n")

    if debug:
        return desc_string, particles, avg_acceptance


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--dimension", type=int, help="Dimension")
    p.add_argument("--num_iters", type=int, help="Number of iterations")
    p.add_argument("--stepsize_mamla", type=float, help="Step size for MAMLA")
    p.add_argument("--stepsize_mla", type=float, help="Step size for MLA")
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
