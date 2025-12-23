import hashlib
import json
import threading

import optuna
from optuna.samplers import TPESampler

import brain
import const

study_name = "lev"
region = "USA"
alpha_expression = "liabilities / assets"

session = brain.re_login()

visited_params = set()
lock = threading.Lock()


def get_params_hash(params):
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode("utf-8")).hexdigest()


def objective(trial):
    universe = trial.suggest_categorical("universe", const.regions[region]["Universe"])
    delay = trial.suggest_categorical("delay", const.regions[region]["Delay"])
    neutralization = trial.suggest_categorical(
        "neutralization", const.regions[region]["Neutralization"]
    )
    max_trade = trial.suggest_categorical("maxTrade", ["ON", "OFF"])

    current_params = trial.params
    param_hash = get_params_hash(current_params)

    with lock:
        if param_hash in visited_params:
            raise optuna.TrialPruned("Duplicate Parameters.")
        visited_params.add(param_hash)

    simulation_data = {
        "type": "REGULAR",
        "settings": {
            "instrumentType": "EQUITY",
            "region": region,
            "universe": universe,
            "delay": delay,
            "decay": 0,
            "neutralization": neutralization,
            "truncation": 0.08,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "ON",
            "maxTrade": max_trade,
            "language": "FASTEXPR",
            "visualization": False,
            "testPeriod": "P2Y",
        },
        "regular": alpha_expression,
    }

    alpha_id = brain.Alpha.simulate(session, simulation_data)
    result_json = brain.Alpha.simulation_result(session, alpha_id)

    insample = result_json["is"]
    sharpe = insample["sharpe"]
    fitness = insample["fitness"]

    return round(sharpe * fitness, 2)


if __name__ == "__main__":
    sampler = TPESampler(constant_liar=True, multivariate=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )

    print("Starting Optuna Optimization...")

    study.enqueue_trial(
        {
            "universe": "TOP3000",
            "delay": 1,
            "neutralization": "SUBINDUSTRY",
            "maxTrade": "OFF",
        }
    )

    try:
        study.optimize(objective, n_trials=100, n_jobs=5)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    print()
    print("Optimization Completed!")

    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Score: {study.best_value}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    print("Best Trial Metrics:")
    print(f"    Sharpe: {study.best_trial.user_attrs.get('sharpe')}")
    print(f"    Turnover: {study.best_trial.user_attrs.get('turnover')}")
