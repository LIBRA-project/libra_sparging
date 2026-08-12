# libra_sparging

A 1D advection–reaction–dispersion model of gas sparging for tritium extraction from a static
molten salt. Bubbles of inert gas rise through the salt, tritium transfers across the
gas–liquid interface and is carried out of the tank; the model predicts how fast the salt is
stripped, and under which conditions a closed-form solution is enough.

Developed for the LIBRA Pi tritium breeding experiment, but not specific to it: the species,
the salt and the column geometry are all inputs.

## Install

```bash
conda env create -f environment.yml
conda activate libra_sparging
pip install -e .
```

The finite element discretisation uses [FEniCSx/DOLFINx](https://fenicsproject.org/), the
closure-relation resolution [NetworkX](https://networkx.org/), and every quantity carries its
units through [Pint](https://pint.readthedocs.io/).

## A first run

```python
from sparging import get_sim_input_LIBRA_Pi, Simulation, ureg

sim_input = get_sim_input_LIBRA_Pi()          # a ready-made case
print(sim_input.get_tau().to("hour"))          # analytical extraction time
print(sim_input.get_Pi_ave())                  # saturation number

results = Simulation(sim_input, t_final=5 * sim_input.get_tau()).solve()
```

`examples/sparging101.py` walks through the same thing more slowly.

## Layout

| path | what it is |
|---|---|
| `src/sparging/` | the package |
| `data/` | one directory per study, each with its `metadata.json` |
| `generate_data.py` | one function per dataset in `data/`, reruns it from scratch |
| `make_plots.ipynb` | turns `data/` into the thesis figures |
| `make_*.py` | the three figures whose plotting is heavy enough to live outside the notebook |
| `examples/` | short standalone scripts |
| `test/` | `pytest` suite: input resolution, solver against known solutions, serialisation |

Inside the package:

| module | responsibility |
|---|---|
| `simulation_input.py` | `SimulationInput`, and the graph search that resolves the closure relations |
| `ard_model.py` | `Simulation`, the finite element solver, and `SimulationResults` |
| `closure.py` | every closure relation, each carrying its source and validity range |
| `example_cases.py` | pre-built cases (LIBRA Pi, LIBRA 1L, a generic standard case) |
| `postprocess.py` | exponential fits, extraction times, decay diagnostics |
| `config.py` | unit registry, physical constants, provenance helpers |

## Datasets

Each directory in `data/` records the git commit it was produced at. A `-dirty` suffix means
the working tree had uncommitted changes at the time, so the run is not reproducible from that
commit alone.

| dataset | what it supports |
|---|---|
| `verification/` | the numerical solution against the analytical one, in both partial pressure regimes |
| `discretisation_convergence/` | mesh and time step convergence, Richardson extrapolation |
| `design_space/` | 300 samples over the three governing dimensionless groups: where the analytical solution holds |
| `non_exponential_sample/` | the sample whose inventory decay is super-exponential |
| `libra_pi_sobol_optimistic/`, `libra_pi_sobol_pessimistic/` | 1536-sample Saltelli designs over the four operating parameters, one per transport-property scenario |
| `libra_pi_corner_*/` | four independent corner runs, used to check the surrogate |

The two scenarios bracket the tritium transport properties of ClLiF, which have not been
measured: `pessimistic` uses the Calderoni solubility and diffusivity, `optimistic` the
Malinauskas solubility and the Fukada diffusivity.

## Reproducing

```bash
python generate_data.py          # regenerates data/ (hours, parallel over 6 workers)
jupyter lab make_plots.ipynb     # data/ -> figures
python make_sobol_figure.py      # the three heavier figures
python make_design_map_figures.py
python make_operating_map.py
```

## License

MIT, see `LICENSE`.
