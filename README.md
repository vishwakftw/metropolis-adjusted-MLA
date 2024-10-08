# metropolis-adjusted-MLA
Code for the experiments in the paper [_Fast sampling from constrained spaces using the Metropolis-adjusted Mirror Langevin algorithm_](https://arxiv.org/abs/2312.08823).

### Required packages

To use the package `metromirrorlangevin`, you will require
- PyTorch (version 2.0.0)
- NumPy (version 1.24.3)
- SciPy (version 1.9.1)
- python-OT / POT (version 0.9.1)
- Matplotlib (version 3.7.1)
- tqdm (version 4.65.0)

To run the scripts (both in `scripts/` and `plotting_scripts/`), you will additionally require
- Pandas (version 2.0.1)
- python-ternary (version 1.0.8)

### How to use `metromirrorlangevin`?

Perform `pip install .` or `pip install -e .` if you would like to an editable installation.
After installation, you can use it like so.

```python
import torch

from metromirrorlangevin import algorithms, barriers

# Define a log-barrier for a 2D box [-0.01, 0.01] x [1, 1]
barrier = barriers.BoxBarrier(bounds=torch.tensor([0.01, 1]))

# Define the sampler instance, with number of samples = 500
sampler = algorithms.mirror_algorithms.UniformSamplerMMRW(
    barrier=barrier,
    num_samples=500
)

# Initialise the particles
# in the smaller box [-0.001, 0.001] x [-0.001, 0.001]
sampler.set_initial_particles(torch.rand(500, 2) * 0.002 - 0.001

# Perform the mixing for 1000 iterations, with step size 0.05
# particles is of shape (num_iters, num_samples, dimension)
# rejects is of shape (num_iters, num_samples)
particles, rejects = sampler.mix(
    num_iters=1000,
    stepsize=0.05,
    return_particles=True,
    no_progress=False
)
```

### Citation

```
@InProceedings{srinivasan24a,
  title = 	 {Fast sampling from constrained spaces using the Metropolis-adjusted Mirror Langevin algorithm},
  author =       {Srinivasan, Vishwak and Wibisono, Andre and Wilson, Ashia},
  booktitle = 	 {Proceedings of Thirty Seventh Conference on Learning Theory},
  pages = 	 {4593--4635},
  year = 	 {2024},
  editor = 	 {Agrawal, Shipra and Roth, Aaron},
  volume = 	 {247},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {30 Jun--03 Jul},
  publisher =    {PMLR},
}
```
