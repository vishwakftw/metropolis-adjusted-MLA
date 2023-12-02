import numpy as np
import pandas as pd
import torch

torch.set_default_dtype(torch.float64)

from argparse import ArgumentParser
from time import time

from metromirrorlangevin.algorithms.mirror_algorithms import UniformMMRWSampler
from metromirrorlangevin.barriers import BoxBarrier, EllipsoidBarrier, SimplexBarrier
from metromirrorlangevin.utils import define_box, define_ellipsoid


def run_sampling(
    domain_type: str,
    dimension: int,
    condition_number: float,
    num_iters: int,
    stepsize: float,
    run_index: int,
    num_particles: int,
    progress_file: str,  # this will carry general information post-completion
    proportion_progress_file: str,  # this will be about the proportion of points outside
    debug: bool = False,
    show_tqdm: bool = False,
):
    # create a random seed based on the values
    random_seed = int(dimension * 1729 + condition_number * 10 + run_index)
    # same random_seed for a specification of stepsize
    # this is to better judge the effects of these on the same sampling problem.
    # same random_seed for a specification of num_particles as well.

    torch.manual_seed(random_seed)
    if domain_type == "ellipsoid":
        ellipsoid = define_ellipsoid(
            dimension=dimension,
            random_seed=random_seed,
            condition_number=condition_number,
        )
        barrier = EllipsoidBarrier(ellipsoid=ellipsoid)
    elif domain_type == "box":
        bounds = define_box(dimension=dimension, condition_number=condition_number)
        barrier = BoxBarrier(bounds=bounds)
    elif domain_type == "simplex":
        barrier = SimplexBarrier(dimension=dimension)
    else:
        raise ValueError(f"Invalid domain_type {domain_type}")

    if domain_type != "box":
        MAXCOST = 2**28
        batch_size = min(int(np.ceil(MAXCOST / dimension**2)), num_particles)
    else:
        batch_size = num_particles

    split_id = 0
    total_samples = 0
    total_time = 0
    acceptance_full = torch.zeros(num_iters)
    all_particles = []
    require_proportion = proportion_progress_file != "NA"
    if require_proportion:
        total_boundary_to_interior_half = torch.zeros(num_iters + 1)

    while total_samples < num_particles:
        num_samples = min(batch_size, num_particles - split_id * batch_size)
        total_samples += num_samples
        split_id += 1
        mmrw_sampler = UniformMMRWSampler(barrier=barrier, num_samples=num_samples)
        if domain_type == "box":
            # initialise within [-0.1 min_{b}, 0.1 min_{b}]
            initial_particles = (
                torch.rand(num_samples, dimension) * 0.2 / condition_number
                - 0.1 / condition_number
            )
        elif domain_type == "ellipsoid":
            # initialise within a small ball of radius 0.1 / condition_number around the origin
            initial_particles = torch.randn(num_samples, dimension)
            initial_particles /= torch.linalg.norm(
                initial_particles, dim=-1, keepdim=True
            )  # norm 1
            initial_particles *= (
                torch.rand(num_samples, 1) * 0.1 / condition_number
            )  # norm at most 1 / (10 * k)
        else:
            # initialise close to the origin
            initial_particles = torch.rand(num_samples, dimension) / (
                5 * dimension
            ) + 1 / (20 * dimension)

        mmrw_sampler.set_initial_particles(initial_particles)

        time_start = time()
        particles, rejection_masks = mmrw_sampler.mix(
            num_iters=num_iters,
            stepsize=stepsize,
            return_particles=require_proportion,
            no_progress=not show_tqdm,
        )
        time_end = time()
        total_time += time_end - time_start
        if require_proportion:
            boundary_to_interior_half = barrier.boundary_to_interior_half(particles)
            # (num_iters + 1) x num_particles
            total_boundary_to_interior_half.add_(boundary_to_interior_half.sum(dim=-1))
        if debug:
            all_particles.append(particles)
        acceptance_full.add_(num_samples - torch.sum(rejection_masks, dim=-1))

    # compute avg_acceptance after the first tenth of iterations
    avg_acceptance = torch.mean(acceptance_full[int(0.1 * num_iters) :] / num_particles)

    desc_string = (
        f"{dimension},{condition_number},{num_iters},"
        f"{stepsize},{run_index},{num_particles},"
        f"{total_time:.4f},{avg_acceptance.item():.6f}"
    )
    if debug:
        return desc_string, all_particles, avg_acceptance

    with open(progress_file, "a") as f:
        f.write(desc_string + "\n")

    if require_proportion:
        # compute proportion of points between boundary and interior half
        total_boundary_to_interior_half.div_(num_particles)

        for idx in range(num_iters + 1):
            # do this so that there are no writing problems from multiple tasks
            proportion = total_boundary_to_interior_half[idx].item()
            desc_string = (
                f"{dimension},{condition_number},{idx},"
                f"{stepsize},{run_index},{proportion:.4f}"
            )
            with open(proportion_progress_file, "a") as f:
                f.write(desc_string + "\n")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "--domain_type",
        type=str,
        choices=["ellipsoid", "box", "simplex"],
        help="Domain type",
    )
    p.add_argument("--dimension", type=int, help="Dimension")
    p.add_argument("--condition_number", type=float, help="Condition Number")
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
        "--proportion_progress_file",
        default="proportion_progress.txt",
        type=str,
        help="Configuration to store proportion results",
    )
    args = p.parse_args()
    args = vars(args)
    run_sampling(**args)
