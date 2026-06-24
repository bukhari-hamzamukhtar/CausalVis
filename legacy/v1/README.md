# CausalVis

## Setup

This repository requires Python 3.9 for the working environment. The current system Python 3.14 installation is not compatible with the required PyTorch wheel and package setup used here.

1. Create and activate the dedicated venv:

```powershell
py -3.9 -m venv .venv39
.venv39\Scripts\Activate.ps1
```

2. Upgrade `pip`:

```powershell
python -m pip install --upgrade pip
```

3. Install the required dependencies:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu
python -m pip install -r requirements.txt
python -m pip install torch_geometric
```

## What was fixed

- Added a shared helper file: `pkg_paths.py`
- Refactored `src/reasoning/composer.py` and `src/reasoning/counterfactual.py` to use package imports via the repository root
- Removed fragile local `sys.path` hacks and local module import assumptions
- Ensured both scripts can run from the repo root using the same environment

## Running the scripts

From the repo root with `.venv39` activated:

```powershell
python src/reasoning/composer.py
python src/reasoning/counterfactual.py
```

## Notes

- `composer.py` uses `src.data.loader.load_scene` and `src.data.build_dataset.process_video` to read the dataset JSON files.
- `counterfactual.py` uses the same data loaders and `src.models.gnn.CausalGNN` to generate counterfactual outcomes.
- Both scripts now load `causal_gnn_weighted.pt` from `src/models` via the repository root path.
