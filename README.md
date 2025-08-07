# Reduced-Order Modeling of Turbulent and Ocean Flows and Lagrangian Data Assimilation with Bayesian Optimization 

This repository contains code and data for reduced-order modeling (ROM) and Lagrangian data assimiliation with Bayesian optimization for two dynamical systems: **Kolmogorov flow** and **ocean flow from the Finite Volume Community Ocean Model**. The goal is to develop compact representations of complex flow fields and efficiently assimilate trajectories using machine learning methods.

## Directory Structure

```text

├── data/                  # Preprocessed datasets  
│   ├── fvcom/             # FVCOM ocean model data  
│   ├── kolmogorov/        # 2D Kolmogorov flow velocity and vorticity  
│   └── massachusetts/     # GIS shapefiles for plotting
├── examples/              # Jupyter notebooks and python scripts demonstrating usage  
│   ├── fvcom/  
│   └── kolmogorov/  
├── figures/               # Saved plots and visualizations  
├── results/               # Saved plots and visualizations  
├── src/                   # Source code  
│   ├── models/            # PCA, MLP, CNN architectures and saved checkpoints  
│   ├── utils.py           # Utility functions  
│   ├── optimizer.py       # BO routines  
│   ├── inputs.py          # Wrapper class for Gaussian inputs  
├── environment.yml        # Conda environment file  
├── README.md              # This file  
```

## Getting Started

### Installation

To set up the environment, use `conda`:
```bash
conda env create -f environment.yml
conda activate rom-flow
```

Alternatively, install dependencies using pip:
```bash
pip install -r requirements.txt
```

### Data Overview

- `data/fvcom/`: Velocity fields and grid data from the FVCOM ocean model  
- `data/kolmogorov/`: Synthetic 2D turbulence (u, v, vorticity)  
- `data/massachusetts/`: GIS shapefiles for regional mapping and domain boundaries

These files can be accessed here (links coming soon).

### Reduced-Order Models

Implemented reduced-order models include:

- PCA: Principal Component Analysis for linear dimensionality reduction
- MLP Autoencoders: Fully connected neural networks for nonlinear latent representations
- CNN Autoencoders: Convolutional architectures suited for spatially structured flow data

Model checkpoints and training logs are saved in `src/models/checkpoints/`.

### Examples

The `examples/` directory contains organized Jupyter notebooks for:

- Training and evaluating reduced-order models  
- Computing characteristic timescales using autocorrelation  
- Performing trajectory optimization using Bayesian Optimization (BO)  

Each subdirectory (`fvcom/`, `kolmogorov/`) contains examples for its respective system.

### Results and Figures

The `results/` and `figures/` directories contain the saved datasets and figures produced by running the examples in (`fvcom/`, `kolmogorov/`). These files can be accessed here (links coming soon).

### Citation
If you use this repository in your research, please cite the appropriate publications (citation coming soon).