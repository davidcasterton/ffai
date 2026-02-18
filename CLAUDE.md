# Claude Behavior Guidelines for ffai

## Python Virtual Environment

This repo uses a virtual environment at `.venv/`. Always use it when running Python commands:

```bash
.venv/bin/python   # instead of python / python3
.venv/bin/pip      # instead of pip
```

The package is installed in editable mode (`pip install -e ffai/`), so source changes in `ffai/src/` take effect immediately without reinstalling.

PufferLib is installed from source at `/home/dave/code/public/PufferLib/` (required for PyTorch ABI compatibility — do not install from PyPI).

## Plans

Whenever a plan is created for this project, save it as a markdown file in the `plans/` directory using the format:

```
plans/YYYY-MM-DD_{short description}.md
```

Example: `plans/2026-02-18_rl-training-refactor.md`
