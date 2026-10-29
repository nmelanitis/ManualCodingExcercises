# Data directories

- `raw/`: original input files. Do not edit them in place.
- `interim/`: temporary outputs from cleaning or feature preparation.
- `processed/`: analysis-ready datasets.
- `sample/`: small, non-sensitive datasets that are safe to keep in Git.

Large, private, or licensed datasets should remain outside Git. The root
`.gitignore` keeps generated raw/interim/processed files untracked while retaining
the directory structure.

