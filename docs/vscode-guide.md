# VS Code Guide for Manual Data Science Practice

## One-time setup

1. Open the repository folder, not an individual file.
2. Create `.venv` using the commands in the root `README.md`.
3. Open the Command Palette and run **Python: Select Interpreter**.
4. Choose the interpreter inside `.venv`.
5. Open Extensions and install the workspace recommendations when prompted.

The project recommends a deliberately small extension set:

- **Python** (`ms-python.python`) for running, debugging, environment selection,
  tests, and Python language integration.
- **Jupyter** (`ms-toolsai.jupyter`) for `# %%` cells, the Interactive Window,
  notebook kernels, the Variables view, and the Data Viewer.
- **Data Wrangler** (`ms-toolsai.datawrangler`) as an optional alternative for
  visual DataFrame profiling and previewing cleaning operations with generated
  pandas code. It does not replace the pandas/Jupyter workflow.
- **Ruff** (`charliermarsh.ruff`) for fast linting and formatting.
- **Rainbow CSV** (`mechatroner.rainbow-csv`) for easier inspection of CSV files.

Optional additions: **GitLens** is useful for understanding file history, while
**Error Lens** makes diagnostics more visible inline. Add extensions slowly;
every installed extension runs with meaningful access to your editor environment.

## The daily workflow

1. Open or copy a worksheet ending in `.py`.
2. Confirm the `.venv` interpreter in the VS Code status bar.
3. Work from the first `# %%` cell downward.
4. Use `Shift+Enter` to run the current line/selection and move forward, or use
   the **Run Cell** link displayed above a cell.
5. Inspect DataFrames in the Interactive Window's Variables/Data Viewer rather
   than printing every row. Open Data Wrangler only when an optional visual view
   would help.
6. Add breakpoints when a result surprises you; inspect values instead of adding
   many temporary `print` calls.
7. Finish with the worksheet's reflection prompts.

If cell execution uses the wrong environment, run **Jupyter: Select Notebook
Kernel** or **Python: Select Interpreter** from the Command Palette.

## Optional Data Wrangler workflow

The normal workflow remains pandas plus the Jupyter Variables/Data Viewer. As an
alternative, Data Wrangler can help at three points in an exercise:

1. **Initial profile:** run a cell that outputs `df`, then select **Open `df` in
   Data Wrangler**. Inspect column types, missingness, distributions, and unique
   values.
2. **Investigate anomalies:** filter or sort suspicious columns in Data Wrangler,
   then reproduce the important check with pandas in the worksheet.
3. **Preview cleaning:** optionally try a transformation in Data Wrangler's
   editing mode and inspect the generated pandas code. Return to the worksheet
   and manually adapt the parts you understand and need.

Data Wrangler works in a sandbox: exploration does not change the original data
until changes are explicitly exported. Using it is never required to complete an
exercise. Continue treating `data/raw/` as immutable and write cleaned results to
`data/interim/` or `data/processed/`.

## High-value shortcuts

| Action | macOS | Windows/Linux |
|---|---|---|
| Command Palette | `Cmd+Shift+P` | `Ctrl+Shift+P` |
| Quick Open file | `Cmd+P` | `Ctrl+P` |
| Search project | `Cmd+Shift+F` | `Ctrl+Shift+F` |
| Run line/selection and advance | `Shift+Enter` | `Shift+Enter` |
| Run current `# %%` cell | `Ctrl+Enter` | `Ctrl+Enter` |
| Toggle terminal | `Ctrl+backtick` | `Ctrl+backtick` |
| Toggle comment | `Cmd+/` | `Ctrl+/` |
| Quick Fix | `Cmd+.` | `Ctrl+.` |
| Rename symbol | `F2` | `F2` |
| Go to definition | `F12` | `F12` |
| Find references | `Shift+F12` | `Shift+F12` |
| Move line | `Option+Up/Down` | `Alt+Up/Down` |
| Copy line | `Shift+Option+Up/Down` | `Shift+Alt+Up/Down` |
| Add next match to selection | `Cmd+D` | `Ctrl+D` |
| Start/continue debugging | `F5` | `F5` |
| Toggle breakpoint | `F9` | `F9` |
| Open shortcut editor | `Cmd+K Cmd+S` | `Ctrl+K Ctrl+S` |

On compact Mac keyboards you may need to hold `Fn` for an `F` key. Shortcuts
can also vary with keyboard layout; the Shortcut Editor shows the effective keys.

## Features worth practicing

### Multi-cursor editing

Use `Option+Click` (`Alt+Click` on Windows/Linux) to place extra cursors. It is
excellent for changing repeated column names or adding similar arguments without
copy/paste errors.

### Rename instead of search/replace

Put the cursor on a Python variable and press `F2`. Symbol-aware rename changes
the relevant references and is safer than replacing text across the project.

### Debug data transformations

Click beside a line number to add a breakpoint, then press `F5`. Step through a
short transformation and watch DataFrame shapes, dtypes, or tensor dimensions.
Use `F10` to step over a line and `F11` to step into a function.

### Use the Problems panel

Open **View: Toggle Problems** from the Command Palette. Fix actual errors first,
then warnings that reveal ambiguous types, unused imports, or likely mistakes.
Do not blindly silence a diagnostic without understanding it.

### Format after the idea works

The workspace formats Python on save. During a difficult exercise, focus first on
the behavior; then save and study what the formatter changed. Ruff can also apply
a focused Quick Fix with `Cmd+.` / `Ctrl+.`.

## Official references

- [Python in Visual Studio Code](https://code.visualstudio.com/docs/languages/python)
- [Python Interactive Window and `# %%` cells](https://code.visualstudio.com/docs/python/jupyter-support-py)
- [Getting started with Data Wrangler](https://code.visualstudio.com/docs/datascience/data-wrangler)
- [Default keyboard shortcuts](https://code.visualstudio.com/docs/reference/default-keybindings)
- [Extension security](https://code.visualstudio.com/docs/configure/extensions/extension-runtime-security)
