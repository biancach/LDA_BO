# Reduced-Order Modeling and Optimization of Geophysical Flows

This repository contains code and data for reduced-order modeling (ROM) and Bayesian optimization of two dynamical systems: **Kolmogorov flow** and **oceanic flow from the FVCOM model**. The goal is to develop compact representations of complex flow fields and efficiently optimize trajectories or model parameters using machine learning methods.

## Directory Structure

```text
.
├── data/                  # Preprocessed datasets  
│   ├── fvcom/             # FVCOM ocean model data  
│   ├── kolmogorov/        # 2D Kolmogorov flow velocity and vorticity  
│   └── massachusetts/     # GIS shapefiles for plotting or domain extraction  
├── environment.yml        # Conda environment file  
├── examples/              # Jupyter notebooks demonstrating usage  
│   ├── fvcom/  
│   └── kolmogorov/  
├── figures/               # Saved plots and visualizations  
├── requirements.txt       # Python package dependencies  
├── src/                   # Source code  
│   ├── models/            # PCA, MLP, CNN architectures and saved checkpoints  
│   ├── utils.py           # Utility functions  
│   ├── optimizer.py       # BO routines  
│   ├── inputs.py          # Input pipelines or parameter setup  
├── README.md              # This file  


## Getting Started

### Installation

To set up the environment, use `conda`:

```bash
conda env create -f environment.yml
conda activate rom-flow

Alternatively, install dependencies using pip:

pip install -r requirements.txt

### Data Overview

- `data/fvcom/`: Velocity fields and grid data from the FVCOM ocean model  
- `data/kolmogorov/`: Synthetic 2D turbulence (u, v, vorticity)  
- `data/massachusetts/`: GIS shapefiles for regional mapping and domain boundaries


### Models

Implemented reduced-order models include:

- PCA: Principal Component Analysis for linear dimensionality reduction
- MLP Autoencoders: Fully connected neural networks for nonlinear latent representations
- CNN Autoencoders: Convolutional architectures suited for spatially structured flow data

Model checkpoints and training logs are saved in src/models/checkpoints/.

### Examples

The `examples/` directory contains organized Jupyter notebooks for:

- Training and evaluating reduced-order models  
- Computing characteristic timescales using autocorrelation  
- Performing trajectory optimization using Bayesian Optimization (BO)  

Each subdirectory (`fvcom/`, `kolmogorov/`) contains examples for its respective system.

### Citation
If you use this repository in your research, please cite the appropriate publications (citation coming soon).