# Adversarial State-Space Models (AdvSSM)

Project focusing on adversarial attacks and robustness in State-Space Models (SSMs), specifically using Kalman Filters.

## Project Structure

- `AdvSSM/`: Contains scripts for SSM-based adversarial attacks, optimization, and visualizations.
  - `output/`: Generated plots and results for SSM experiments.
- `RL/`: Contains Reinforcement Learning experiments and results.
  - `experiments/`: RL training and evaluation scripts.
  - `results/`: Plots and data from RL simulations.
  - `saved_models/`: Trained RL policies.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate it:
   - Windows: `.venv\Scripts\activate`
   - Linux/macOS: `source .venv/bin/activate`
3. Install dependencies (if requirements.txt is provided, or manually).

## Usage

Scripts in `AdvSSM/` can be run independently to explore Kalman Filter properties and KKT-based attacks.
Scripts in `RL/` explore the impact of adversarial observations on trained RL agents.
