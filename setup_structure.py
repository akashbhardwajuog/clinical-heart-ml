from pathlib import Path

directories = [
    "configs", "data/raw", "data/interim", "data/processed",
    "notebooks", "src/heartml", "scripts", "tests", "models",
    "reports/figures", "reports/tables", "docs",
]

files = [
    ".gitignore", "README.md", "pyproject.toml",
    "configs/default.json", "docs/project_plan.md",
    "src/heartml/__init__.py", "src/heartml/common.py",
    "src/heartml/prepare.py", "src/heartml/splits.py",
    "src/heartml/train_sklearn.py", "src/heartml/train_torch.py",
    "scripts/download_data.py", "tests/test_data.py"
]

for d in directories:
    Path(d).mkdir(parents=True, exist_ok=True)
for f in files:
    Path(f).touch(exist_ok=True)
print("Project structure created.")