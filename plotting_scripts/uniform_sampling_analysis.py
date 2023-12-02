import numpy as np
import pandas as pd
import scipy.stats as sts
import torch

torch.set_default_dtype(torch.float64)

from argparse import ArgumentParser
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from typing import List

from metromirrorlangevin.algorithms.mirror_algorithms import UniformMMRWSampler
from metromirrorlangevin.barriers import EllipsoidBarrier
from metromirrorlangevin.utils import define_ellipsoid, draw_ellipsoid_boundary

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 24


def get_mixing_time(all_data, c_mult):
    # first filter out all rows which do not meet criterion
    mixed = all_data[all_data["proportion"] >= 0.45]
    # sort the data based on dimension, condition_number, itr, stepsize, run_index
    mixed = mixed.sort_values(
        ["dimension", "condition_number", "run_index", "stepsize", "itr"]
    )
    # groupby the above attributes (except ", and pick the first occurrence
    mixed = mixed.groupby(
        ["dimension", "condition_number", "run_index", "stepsize"]
    ).first()
    # reset_index
    mixed = mixed.reset_index()
    # create column for d_scaling
    mixed["d_scaling"] = np.round(
        np.log(mixed["stepsize"] / c_mult) / np.log(mixed["dimension"]), decimals=2
    )
    # create column for i (the sequence type)
    mixed["seq_type"] = 2 - np.isclose(
        mixed["condition_number"], mixed["dimension"] ** 2 / 4
    )
    return mixed


def generate_plot_small_dimensions_mixing_time(
    proportion_progress_file: str,
    save_location: str,
    stepsize_c_mult: float,
):
    df = pd.read_csv(
        proportion_progress_file,
        names=[
            "dimension",
            "condition_number",
            "itr",
            "stepsize",
            "run_index",
            "proportion",
        ],
    )
    # This follows from the definition in CDWY18
    mixed = get_mixing_time(df, stepsize_c_mult)
    scalings_list = mixed["d_scaling"].unique()

    # After this, we can technically aggregate data for all condition numbers
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if len(mcolors.TABLEAU_COLORS) < len(scalings_list):
        print("WARN: Fewer colours than scalings")

    for scaling, colour in zip(scalings_list, list(mcolors.TABLEAU_COLORS)):
        tmp = mixed[mixed["d_scaling"] == scaling]
        result = sts.linregress(x=np.log(tmp["dimension"]), y=np.log(tmp["itr"]))
        dimension_list = tmp["dimension"].unique()
        x = [
            tmp[tmp["dimension"] == dimension]["itr"].to_numpy()
            for dimension in dimension_list
        ]

        # also plot
        w_by_2 = 0.05 / 2
        log_dim_list = np.log10(dimension_list)
        widths = 10 ** (log_dim_list + w_by_2) - 10 ** (log_dim_list - w_by_2)
        box_plot = ax.boxplot(
            x,
            positions=dimension_list,
            widths=widths,
            showfliers=False,
            patch_artist=True,
        )
        for median in box_plot["medians"]:
            median.set_color("black")
        for box in box_plot["boxes"]:
            box.set_facecolor(colour)

        line_x = np.linspace(dimension_list.min(), dimension_list.max(), 100)
        line_y = np.exp(result.intercept) * np.power(line_x, result.slope)
        ax.plot(
            line_x,
            line_y,
            alpha=0.75,
            color=colour,
            linestyle="--",
            label=f"$\\widehat{{\\tau}}_{{\mathrm{{mix}}}} \propto d^{{{result.slope:.3f}}}$",
            linewidth=3.0,
        )

    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Empirical mixing time $\widehat{\\tau}_{\mathrm{mix}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(save_location)


def generate_plot_large_dimensions_acceptance_rate(
    progress_file: str,
    save_location: str,
    stepsize_c_mult: float,
):
    df = pd.read_csv(
        progress_file,
        names=[
            "dimension",
            "condition_number",
            "num_iters",
            "stepsize",
            "run_index",
            "num_particles",
            "time_elapsed",
            "avg_acceptance",
        ],
    )

    df["d_scaling"] = np.round(
        np.log(df["stepsize"] / stepsize_c_mult) / np.log(df["dimension"]),
        decimals=2,
    )
    df = df.drop(
        columns=[
            "condition_number",
            "num_iters",
            "stepsize",
            "num_particles",
            "time_elapsed",
        ]
    )
    # df has three columns: dimension, d_scaling, avg_acceptance
    scalings_list = sorted(df["d_scaling"].unique())
    dimension_list = sorted(df["dimension"].unique())

    fig, ax = plt.subplots(1, 1, figsize=(6, 7))

    if len(mcolors.TABLEAU_COLORS) < len(scalings_list):
        print("WARN: Fewer colours than scalings")

    for scaling, colour, marker in zip(
        scalings_list, list(mcolors.TABLEAU_COLORS), ["o", "x", "*", "s"]
    ):
        tmp = df[df["d_scaling"] == scaling]
        x = [
            tmp[tmp["dimension"] == dimension]["avg_acceptance"].to_numpy()
            for dimension in dimension_list
        ]

        mean_x = [np.mean(xi) for xi in x]
        ax.plot(
            dimension_list,
            mean_x,
            color=colour,
            alpha=0.9,
            linestyle="-",
            marker="o",
            markerfacecolor=None,
            label=f"$\gamma = {-scaling}$",
            linewidth=3.0,
        )

    lgd = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fancybox=True,
        fontsize=18,
    )
    max_dim = max(dimension_list)
    ax.set_xlim(0, max_dim + 2)
    ax.set_ylim(-0.02, 1)
    if max_dim < 500:
        increments = 75
    else:
        increments = 100
    ax.set_xticks(np.arange(0, max_dim // increments * (increments + 1), increments))
    ax.set_xticklabels(
        np.arange(0, max_dim // increments * (increments + 1), increments)
    )

    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Average acceptance rate")
    ax.grid()
    plt.tight_layout()
    plt.savefig(save_location, bbox_extra_artists=(lgd,))


def generate_transition_plots(num_transitions: int, save_location: str):
    # Condition number 4 will look nice
    # Ellipsoid
    H = 0.25 / np.sqrt(8)
    T = 60
    N = 500
    DIM = 2
    COND = 4

    torch.manual_seed(0)
    ellipsoid = define_ellipsoid(dimension=DIM, random_seed=0, condition_number=COND)
    barrier = EllipsoidBarrier(ellipsoid=ellipsoid)
    initial_particles = torch.randn(N, 2) * 0.01
    assert torch.all(barrier.feasibility(initial_particles)), "Invalid particles"

    sampler = UniformMMRWSampler(barrier=barrier, num_samples=N)
    sampler.set_initial_particles(initial_particles)
    particles, _ = sampler.mix(
        num_iters=T, stepsize=H, no_progress=False, return_particles=True
    )

    fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(24, 6))
    # add 1 because num_transitions gives you
    # one extra plot
    iterates_to_plot = np.linspace(0, T, num_transitions + 1).astype(int)
    # initialisation
    for itr, ax in zip(iterates_to_plot, axs.flat):
        curr_particles = particles[itr]
        mask = barrier.boundary_to_interior_half(curr_particles)
        satisfied_proportion = torch.sum(mask) / N
        ax.scatter(
            curr_particles[mask, 0],
            curr_particles[mask, 1],
            c="g",
            label=f"Prop: {satisfied_proportion:.2f}",
        )
        ax.scatter(
            curr_particles[~mask, 0],
            curr_particles[~mask, 1],
            c="b",
            label=f"Prop: {1 - satisfied_proportion:.2f}",
        )
        draw_ellipsoid_boundary(ellipsoid=ellipsoid, ax=ax)
        if itr == 0:
            title = "Initialisation"
        else:
            title = f"After {itr} iterations"
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_location)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "--box_progress_file", type=str, help="Box progress file", default=None
    )
    p.add_argument(
        "--ellipsoid_progress_file",
        type=str,
        help="Ellipsoid progress file",
        default=None,
    )
    p.add_argument(
        "--simplex_progress_file",
        type=str,
        help="Simplex progress file",
        default=None,
    )
    p.add_argument(
        "--box_proportion_progress_file",
        type=str,
        help="Box proportion progress file",
        default=None,
    )
    p.add_argument(
        "--ellipsoid_proportion_progress_file",
        type=str,
        help="Ellipsoid proportion progress file",
        default=None,
    )
    p.add_argument(
        "--simplex_proportion_progress_file",
        type=str,
        help="Simplex proportion progress file",
        default=None,
    )
    p.add_argument(
        "--mixing_time_box_location",
        type=str,
        help="Save location for mixing time (Box)",
        default=None,
    )
    p.add_argument(
        "--mixing_time_ellipsoid_location",
        type=str,
        help="Save location for mixing time (Ellipsoid)",
        default=None,
    )
    p.add_argument(
        "--mixing_time_simplex_location",
        type=str,
        help="Save location for mixing time (Simplex)",
        default=None,
    )
    p.add_argument(
        "--accept_rate_box_location",
        type=str,
        help="Save location for acceptance rate (Box)",
        default=None,
    )
    p.add_argument(
        "--accept_rate_ellipsoid_location",
        type=str,
        help="Save location for acceptance rate (Ellipsoid)",
        default=None,
    )
    p.add_argument(
        "--accept_rate_simplex_location",
        type=str,
        help="Save location for acceptance (Simplex)",
        default=None,
    )
    p.add_argument(
        "--transition_plot_location",
        type=str,
        help="Save location for transition plot (Ellipsoid)",
        default=None,
    )
    args = p.parse_args()
    args = vars(args)

    domains = ["box", "ellipsoid", "simplex"]
    c_mults = [0.25, 0.05, 0.1]

    for domain, c_mult in zip(domains, c_mults):
        if args[f"{domain}_progress_file"] is not None:
            if args[f"accept_rate_{domain}_location"] is None:
                raise ValueError(
                    f"not sure where to save the dimension vs accept rate plot ({domain})"
                )
            generate_plot_large_dimensions_acceptance_rate(
                progress_file=args[f"{domain}_progress_file"],
                save_location=args[f"accept_rate_{domain}_location"],
                stepsize_c_mult=c_mult,
            )

        if args[f"{domain}_proportion_progress_file"] is not None:
            if args[f"mixing_time_{domain}_location"] is None:
                raise ValueError(
                    f"not sure where to save the mixing time plot ({domain})"
                )
            generate_plot_small_dimensions_mixing_time(
                proportion_progress_file=args[f"{domain}_proportion_progress_file"],
                save_location=args[f"mixing_time_{domain}_location"],
                stepsize_c_mult=c_mult,
            )

    if args["transition_plot_location"] is not None:
        generate_transition_plots(
            num_transitions=3, save_location=args["transition_plot_location"]
        )
