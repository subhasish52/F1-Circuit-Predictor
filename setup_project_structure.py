# -*- coding: utf-8 -*-
import os

# Use current directory (no nested folder creation)
ROOT_DIR = "."

# Folder structure
structure = [
    "data/raw",
    "data/processed",
    "data/external",
    "notebooks",
    "src",
    "api",
    "dashboard",
    "utils",
    "models",
    "reports",
    "tests",
]

# Files with initial content
files = {
    "README.md": "# 🏁 F1 Circuit Predictor\n\nProject to predict F1 race outcomes using ML.",
    "requirements.txt": "# Add your pip packages here\npandas\nnumpy\nscikit-learn\nxgboost\nfastf1\nstreamlit\nfastapi\nuvicorn",
    "Dockerfile": "FROM python:3.10\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"python\", \"src/model.py\"]",
    ".gitignore": "__pycache__/\n*.pkl\n.env\n.ipynb_checkpoints/\n",
    "src/__init__.py": "",
    "src/config.py": "# Config variables (e.g., file paths, API keys)",
    "src/data_ingestion.py": "# Load data from APIs or files",
    "src/preprocess.py": "# Clean and transform data",
    "src/features.py": "# Feature engineering logic",
    "src/model.py": "# Train and save the ML model",
    "src/predict.py": "# Load model and make predictions",
    "api/main.py": "# FastAPI app for prediction endpoint",
    "api/schema.py": "# Request and response schemas",
    "dashboard/app.py": "# Streamlit dashboard code here",
    "utils/telemetry.py": "# Lap time & telemetry analysis",
    "utils/weather.py": "# Weather API integration",
    "utils/helpers.py": "# Miscellaneous helper functions",
    "reports/model_eval.md": "# Model Evaluation Reports",
    "tests/test_features.py": "# Unit test for features",
    "tests/test_model.py": "# Unit test for model logic",
}

# Create folders
for path in structure:
    full_path = os.path.join(ROOT_DIR, path)
    os.makedirs(full_path, exist_ok=True)
    print(f"Created: {full_path}")

# Create files with UTF-8 encoding
for file_name, content in files.items():
    full_path = os.path.join(ROOT_DIR, file_name)
    dir_path = os.path.dirname(full_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created file: {full_path}")

print("\n✅ Project structure created successfully in current folder.")
