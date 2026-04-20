# XGBoost from Scratch for Wildfire Prediction

<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![MLflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=flat&logo=mlflow&logoColor=blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

</div>

This repository contains a complete implementation of the **XGBoost algorithm from scratch**, originally developed in the context of the *Advanced Machine Learning* course taught at ENSAE. 

The project has now transitioned into an **MLOps pipeline**, focusing not only on understanding the theoretical foundations of gradient boosting, but also on encapsulating the resulting model into a deployable infrastructure.

## Project Objectives

The main objectives of this project are twofold:

**1. Machine Learning Core:**
- To implement the XGBoost algorithm from scratch, without relying on existing ML libraries.
- To closely follow the original mathematical formulation (second-order optimization, regularization, greedy split selection, and approximate split finding).
- To apply the model to the US wildfire dataset and evaluate its predictive performance against standard implementations.

**2. MLOps & Industrialization:**
- To structure the project following the Data Science Cookiecutter standard.
- To separate code from data by hosting datasets on an **S3 MinIO** bucket.
- To track experiments and model registries using **MLflow**.
- To containerize the application and serve predictions via a **FastAPI** interface.
- To automate deployment on a **Kubernetes** cluster using a **GitOps** approach (ArgoCD).

## Repository Structure

The repository follows standard MLOps practices to ensure reproducibility and clean collaboration:

```text
.
├── data/
├── k8s/
├── requirements.txt
├── notebooks/
│   └── main.ipynb
└── src/
    ├── api/
    ├── data/
    └── models/
```

## Dataset

The experiments are based on the **US Wildfire Dataset (2014–2025)**, publicly available on [Kaggle](https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset). It combines wildfire ignition records with meteorological and environmental variables derived from the GRIDMET climatology database.

**Important:** To keep the repository lightweight, the aggregated dataset (`df_aggregated.parquet`) is hosted on our S3 MinIO bucket. It is automatically fetched by the scripts located in `src/data/` using the credentials defined in your `.env` file.

## Git Workflow & Branching Convention

To avoid conflicts and maintain a clean history, **pushing directly to the `main` branch is strictly disabled**. All changes must go through a Pull Request (PR) and require at least one approval.

**Branch Naming Convention:**
Please name your branches using the following format: `type/firstname-action`
- `feat/` : New feature 
- `fix/` : Bug fix 
- `docs/` : Documentation updates 
- `chore/` : Maintenance or configuration 
- `refactor/`: Code reorganization without changing functionality

## Methodology & References

Our implementation is primarily inspired by the original XGBoost paper:
> **Tianqi Chen and Carlos Guestrin** > *XGBoost: A Scalable Tree Boosting System*

A detailed description of the theoretical foundations, implementation choices, and experimental results can be found in our accompanying academic report:
- [Overleaf Report Link](https://overleaf.enst.fr/project/6901e9ac9912eb0d35ba0d1d)

This document includes a rigorous derivation of the XGBoost objective, an analysis of the dataset, and a comparison with existing models from the literature.

## Installation & Usage

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A `.env` file based on `.env.example`

### Setup
```bash
git clone https://github.com/plemvan/WildFirePrediction
cd WildFirePrediction
uv sync
```

### Train the model
```bash
PYTHONPATH=. python src/models/train.py --n-iter 20 --n-splits 5
```

### Run the API locally
```bash
PYTHONPATH=. uvicorn src.api.main:app --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## Deployment

### API
The API is live at **[wildfire-api.user.lab.sspcloud.fr/docs](https://wildfire-api.user.lab.sspcloud.fr/docs)** — try it directly from your browser.

- `GET /health` — liveness check
- `POST /predict` — wildfire probability prediction
- `GET /docs` — interactive Swagger documentation

### Kubernetes & ArgoCD
The API is deployed on SSP Cloud (Onyxia) via Kubernetes.
ArgoCD is configured to watch the `k8s/` folder of this repository,
with automatic sync policy on the `develop` branch.

To apply manifests manually:
```bash
kubectl apply -f k8s/
```

### Monitoring & Logs
Grafana and Prometheus are not available on SSP Cloud.
Monitoring is done via `kubectl logs`:

```bash
# Stream live logs from the API pod
kubectl logs -f deployment/wildfire-api -n <namespace>

# Each prediction is logged with inputs and output:
# PREDICT | inputs={...} | wildfire=0 | probability=0.3032
```

### MLflow
Model training and experiment tracking are managed via MLflow.
The production model is registered under `wildfire-xgboost-classifier`
in the MLflow Model Registry.
