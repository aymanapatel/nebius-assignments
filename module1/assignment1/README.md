# LLM Architectures - Hometask 1

This repository contains the solution for the first module's hometask on LLM Architectures.

## Environment Setup

The project uses `uv` for lightning-fast Python package management and requires **Python 3.12+**.

### Installation
To set up the virtual environment with Python 3.12 and install all dependencies:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install datasets torch numpy matplotlib ipywidgets ipykernel tqdm pandas nbdime
```

### Jupyter Kernel
To use the virtual environment in your Jupyter notebooks:
```bash
uv run python -m ipykernel install --user --name=nebius_hometask --display-name "Python (Nebius Hometask)"
```

---

## 📊 Viewing Notebook Diffs with `nbdime`

To avoid JSON noise in `.ipynb` diffs, we use `nbdime`.

| Task | Command | Description |
| :--- | :--- | :--- |
| **Initial Setup** | `uv run nbdime config-git --enable --global` | One-time setup to integrate nbdime with Git. |
| **Terminal Diff** | `git diff` | Standard git command now shows a clean notebook diff. |
| **Direct Diff** | `uv run nbdiff notebook.ipynb` | Compare files directly without using Git. |
| **Web Diff (All)** | `uv run nbdime diff-web` | Open a **side-by-side visual comparison** in your browser. |
| **Web Diff (File)** | `uv run nbdime diff-web <filename>` | Visual comparison for a specific notebook. |
| **Visual Merge** | `uv run nbdime mergetool` | Resolve notebook merge conflicts visually. |

---

## 💡 Troubleshooting
If you encounter an `f-string: unmatched '['` SyntaxError, ensure your nested quotes aren't conflicting:
- **❌ Incorrect:** `f'{data['key']}'`
- **✅ Correct:** `f"{data['key']}"` or `f'{data["key"]}'`
