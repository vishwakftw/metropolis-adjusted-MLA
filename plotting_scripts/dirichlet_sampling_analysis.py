import numpy as np
import pandas as pd
import scipy.stats as sts
import ternary
import torch

from argparse import ArgumentParser
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

from metromirrorlangevin.algorithms.mirror_algorithms import GeneralMAMLASampler
from metromirrorlangevin.barriers import SimplexBarrier
from metromirrorlangevin.potentials import DirichletPotential


plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 24


def get_mixing_time(all_data, c_mult):
    # first filter out all rows which do not meet criterion
    mixed = all_data[all_data["loss"] <= 0.01]
    # sort the data based on dimension, condition_number, itr, stepsize, run_index
    mixed = mixed.sort_values(["dimension", "run_index", "stepsize", "itr"])
    # groupby the above attributes (except ", and pick the first occurrence
    mixed = mixed.groupby(["dimension", "run_index", "stepsize"]).first()
    # reset_index
    mixed = mixed.reset_index()
    # create column for d_scaling
    mixed["d_scaling"] = np.round(
        np.log(mixed["stepsize"] / c_mult) / np.log(mixed["dimension"]), decimals=2
    )
    return mixed


def generate_plot_small_dimensions_mixing_time(
    loss_progress_file: str,
    save_location: str,
    stepsize_c_mult: float,
):
    df = pd.read_csv(
        loss_progress_file,
        names=[
            "dimension",
            "itr",
            "stepsize",
            "run_index",
            "loss",
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
        ax.scatter(
            dimension_list,
            [np.mean(xi) for xi in x],
            marker="s",
            s=40.0,
            edgecolor="k",
            facecolor=colour,
        )

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
    ax.xaxis.set_tick_params(which="minor", labelbottom=False)

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
            "num_iters",
            "stepsize",
            "run_index",
            "num_particles",
            "time_elapsed",
            "avg_acceptance",
            "loss",
        ],
    )

    df["d_scaling"] = np.round(
        np.log(df["stepsize"] / stepsize_c_mult) / np.log(df["dimension"]),
        decimals=2,
    )
    df = df.drop(
        columns=[
            "num_iters",
            "stepsize",
            "num_particles",
            "time_elapsed",
            "loss",
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
    increments = 50
    ax.set_xticks(np.arange(0, max_dim // increments * (increments + 1), increments))
    ax.set_xticklabels(
        np.arange(0, max_dim // increments * (increments + 1), increments)
    )

    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel("Average acceptance rate")
    ax.grid()
    plt.tight_layout()
    plt.savefig(save_location, bbox_extra_artists=(lgd,))


def generate_plots_comparison(
    loss_progress_file: str,
    save_location_folder: str,
):
    df = pd.read_csv(
        loss_progress_file,
        names=[
            "alg",
            "dimension",
            "itr",
            "stepsize",
            "run_index",
            "loss",
        ],
    )
    dimension_list = sorted(df["dimension"].unique())

    for dimension in dimension_list:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for alg, colour in zip(["MAMLA", "MLA"], mcolors.TABLEAU_COLORS):
            tmp = df[(df["alg"] == alg) & (df["dimension"] == dimension)]
            tmp = tmp.drop(columns=["stepsize", "alg", "run_index", "dimension"])
            mat = (
                tmp.groupby(["itr"])
                .mean()
                .reset_index()
                .sort_values(["itr"])
                .to_numpy()
            )
            ax.plot(
                mat[:, 0],
                mat[:, 1],
                c=colour,
                lw=3.0,
                label=alg,
                marker="o",
                markevery=[0, -1],
            )

        ax.set_xlabel("Iterations")
        ax.set_ylabel("$\widetilde{W_{2}^{2}}$")
        ax.legend()
        ax.grid()
        plt.savefig(
            f"{save_location_folder}/comp_MLA_MAMLA_dim={dimension}.pdf",
            bbox_inches="tight",
        )


def generate_transition_plots(num_transitions: int, save_location: str):
    # Condition number 4 will look nice
    # Ellipsoid
    H = 0.25 / np.sqrt(8)
    T = 15
    N = 2000
    DIM = 2

    torch.manual_seed(1)
    barrier = SimplexBarrier(dimension=2)
    initial_particles = torch.full(size=(N, DIM), fill_value=1 / (4 * DIM))
    initial_particles += torch.rand(N, DIM) / (12 * DIM) - 1 / (24 * DIM)
    assert torch.all(barrier.feasibility(initial_particles)), "Invalid particles"

    alpha = torch.tensor([6.0, 6.0, 6.0])
    potential = DirichletPotential(alpha=alpha)
    sampler = GeneralMAMLASampler(barrier=barrier, potential=potential, num_samples=N)
    sampler.set_initial_particles(initial_particles)
    particles, _ = sampler.mix(
        num_iters=T, stepsize=H, no_progress=False, return_particles=True
    )

    fig, axs = plt.subplots(
        nrows=1, ncols=num_transitions + 2, sharex=True, sharey=True, figsize=(24, 6)
    )
    # add 2 because num_transitions gives you
    # one extra plot and one extra plot for the groundtruth
    iterates_to_plot = np.linspace(0, int(T * 0.3), num_transitions).astype(int)
    iterates_to_plot = np.append(iterates_to_plot, [T])
    # initialisation
    for itr, ax in zip(iterates_to_plot, axs.flat[:-1]):
        curr_particles = particles[itr]
        curr_particles_sum = torch.sum(curr_particles, dim=-1, keepdim=True)
        curr_particles = torch.cat(
            [curr_particles, curr_particles_sum.mul_(-1).add_(1.0)], dim=-1
        )
        tax = ternary.TernaryAxesSubplot(ax=ax)
        tax.boundary(lw=2.0)
        tax.gridlines(color="black")
        tax.scatter(curr_particles.numpy(), color="b")
        if itr == 0:
            title = "Initialisation"
        else:
            title = f"After {itr} iterations"
        ax.set_title(title)

    ax = axs.flat[-1]
    tax = ternary.TernaryAxesSubplot(ax=ax)
    tax.boundary(lw=2.0)
    tax.gridlines(color="black")
    tax.scatter(
        torch.distributions.Dirichlet(alpha + 1).sample((N,)).numpy(), color="r"
    )
    ax.set_title("Ground Truth")

    plt.tight_layout()
    plt.savefig(save_location)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "--dirichlet_loss_progress_file",
        type=str,
        help="Dirichlet loss progress file",
        default=None,
    )
    p.add_argument(
        "--mixing_time_location",
        type=str,
        help="Dirichlet mixing time save location",
        default=None,
    )
    p.add_argument(
        "--dirichlet_progress_file",
        type=str,
        help="Dirichlet progress file",
        default=None,
    )
    p.add_argument(
        "--accept_rate_location",
        type=str,
        help="Dirichlet acceptance rate save location",
        default=None,
    )
    p.add_argument(
        "--dirichlet_comparison_progress_file",
        type=str,
        help="Dirichlet comparison progress file",
        default=None,
    )
    p.add_argument(
        "--comparison_save_location",
        type=str,
        help="Dirichlet save location for comparison",
        default=None,
    )
    p.add_argument(
        "--transition_plot_location",
        type=str,
        help="Save location for transition plot (Dirichlet)",
        default=None,
    )

    args = p.parse_args()
    args = vars(args)

    if args["dirichlet_loss_progress_file"] is not None:
        if args["mixing_time_location"] is None:
            raise ValueError("Not sure where to save mixing time plot")

        generate_plot_small_dimensions_mixing_time(
            args["dirichlet_loss_progress_file"], args["mixing_time_location"], 0.25
        )

    if args["dirichlet_progress_file"] is not None:
        if args["accept_rate_location"] is None:
            raise ValueError("Not sure where to accept rate plot")

        generate_plot_large_dimensions_acceptance_rate(
            args["dirichlet_progress_file"], args["accept_rate_location"], 0.25
        )

    if args["dirichlet_comparison_progress_file"] is not None:
        if args["comparison_save_location"] is None:
            raise ValueError("Not sure where to save the comparison plot")

        generate_plots_comparison(
            args["dirichlet_comparison_progress_file"], args["comparison_save_location"]
        )

    if args["transition_plot_location"] is not None:
        generate_transition_plots(
            num_transitions=2, save_location=args["transition_plot_location"]
        )
