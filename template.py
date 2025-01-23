import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = "actionDetection"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_loader.py",
    f"src/{project_name}/components/preprocess.py",
    f"src/{project_name}/components/trainer.py",
    f"src/{project_name}/components/inferencer.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/visualization.py",
    f"src/{project_name}/utils/logging.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/run_pipeline.py",
    f"src/{project_name}/evaluation/__init__.py",
    f"src/{project_name}/evaluation/metrics.py",
    f"config/config.yaml",
    f"config/constants.py",
    "logs/running_logs.log",
    "artifacts/data/.gitkeep",
    "artifacts/models/.gitkeep",
    "artifacts/results/.gitkeep",
    "research/.gitkeep",
    "tests/test_preprocessing.py",
    "Dockerfile",
    "app.py",
    "main.py",
    "requirements.txt",
    "README.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
