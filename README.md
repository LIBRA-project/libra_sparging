# libra_sparging

To run the model:

1. Create the right conda environment

> [!NOTE]
> Requires and conda to be installed

> [!NOTE]
> This uses `dolfinx` which doesn't run on Linux. For windows users, consider using Windows Subsystem for Linux (WSL)

To simply use the library, use the libra_sparging environment:
```
conda env create -f environment.yml
conda activate libra_sparging
```

To also train a surrogate model (for sensitivity analysis, for example), you need autoemulate, use the autoemulate environment:
```
conda env create -f autoemulate_env.yml
conda activate autoemulate_env
```

```
python -m pip install -e .[dev]
```

## How to run tests

```
python -m pytest test
```