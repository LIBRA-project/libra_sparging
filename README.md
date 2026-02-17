# libra_sparging

To run the model:

1. Create the right conda environment

> [!NOTE]
> Requires and conda to be installed

> [!NOTE]
> This uses `dolfinx` which doesn't run on Linux. For windows users, consider using Windows Subsystem for Linux (WSL)

```
conda env create -f environment.yml
conda activate libra_sparging
```


```
python model.py
```