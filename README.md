# Spike-MPPI for Franka Manipulation

This project studies **Franka Manipulation with Model Predictive Path Integral control (MPPI)** in MuJoCo.

The main novelty of this repository is **Spike-MPPI**, a motoneuron-inspired MPPI sampling method. Instead of directly sampling Gaussian actuator perturbations, Spike-MPPI generates perturbations through stochastic common drive, ordered motor-unit recruitment, discharge dynamics, and heterogeneous causal twitch responses.

Paper is available at: https://arxiv.org/abs/2609.28325

Detailed code and install for a locomotion task and Ant robot is available at: https://github.com/fleurssauvages/spikeMPPI

## Demo

<p align="center">
  <strong>Spike-MPPI</strong>
</p>

<p align="center">
  <img src="manipulation/results/spike_reaching.gif"
       width="800"
       alt="Spike-MPPI reaching">
</p>

<p align="center">
  <strong>Standard MPPI</strong>
</p>

<p align="center">
  <img src="manipulation/results/standard_reaching.gif"
       width="800"
       alt="Standard MPPI reaching">
</p>

## Results on 100 runs

<h3 align="center">Normalized Reaching Progress</h3>

<p align="center">
  <img src="manipulation/results/reaching_normalized_distance.png"
       width="450"
       alt="Normalized reaching progress">
</p>

<h3 align="center">Performances</h3>

<p align="center">
  <img src="manipulation/results/reaching_violins.png"
       width="900"
       alt="Performances">
</p>


## Install

Requires Python 3.10+, MuJoCo 3.3+, and a C++17 compiler for the optional native rollout extension.

```bash
git clone https://github.com/fleurssauvages/Franka_MPPI.git
cd Franka_MPPI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python manipulation/setup_native.py build_ext --inplace
```

## Run

```bash
python -m manipulation.franka_reaching.run --sampling spike
```

Use `--sampling standard` or `--sampling poisson` for comparison.
