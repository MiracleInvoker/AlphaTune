# AlphaTune

AlphaTune is a Python-based optimization tool designed to automate the tuning of simulation parameters for financial Alphas (trading strategies). It leverages [Optuna](https://optuna.org/) to find the most effective configuration for a given Alpha Expression.

## ⚡ Features

- **Automated Tuning:** Uses Optuna's `TPESampler` to explore parameter spaces efficiently.
- **Parallel Execution:** Runs multiple simulation trials concurrently to speed up the optimization process.
- **Custom Objective:** Optimizes based on custom metrics to avoid overfitting.
- **Simulation Control:** Automatically varies key simulation settings, including:

  - **Universe**
  - **Delay**
  - **Neutralization**
  - **MaxTrade**

- **Duplicate Handling:** Prevents redundant simulations by hashing and tracking visited parameter sets.

## 🚀 Usage

1. Ensure you have the required dependencies installed (`optuna`, etc.) and the `brain.py` module configured for your environment.
2. Define your `alpha_expression` and `region` in `tune.py`.
3. Run the script:

```bash
python tune.py

```

4. The script will execute 100 trials (by default) and output the best simulation parameters and metrics upon completion.
