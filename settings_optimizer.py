import copy
import datetime
import json
import math

import optuna
from optuna.samplers import GridSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import brain
import reward

with open("regions.json", "r") as f:
    regions = json.load(f)  # Region Constants
with open("alpha.json", "r") as f:
    alpha = json.load(f)

now = datetime.datetime.now()
study_file_name = now.strftime("%Y-%m-%d_%H-%M-%S")

region = alpha["settings"]["region"]
region_consts = regions[region]
region_consts["Neutralization"].remove(
    "NONE"
)  # No Point in having Neutralization as NONE

brain_session = brain.re_login()

search_space = {
    "maxTrade": ["ON", "OFF"],
    "universe": region_consts["Universe"],
    "neutralization": region_consts["Neutralization"],
}

# Conditional Search Space: Might not always be necessary to Optimize Delay
if alpha["settings"]["delay"] is None:
    search_space["delay"] = region_consts["Delay"]
else:
    print(
        f"Delay is already set to {alpha['settings']['delay']}. Skipping optimization for delay."
    )
    search_space["delay"] = [alpha["settings"]["delay"]]


def objective(trial):

    # Make a copy of the Global Variable "alpha" so that the objective function is thread safe
    trial_alpha = copy.deepcopy(alpha)

    p_delay = trial.suggest_categorical("delay", search_space["delay"])
    p_universe = trial.suggest_categorical("universe", search_space["universe"])
    p_neutralization = trial.suggest_categorical(
        "neutralization", search_space["neutralization"]
    )
    p_maxTrade = trial.suggest_categorical("maxTrade", search_space["maxTrade"])

    # Handling the Concurrency Race Condition which occurs at the end of the study
    for t in trial.study.trials:

        if t.number != trial.number and t.params == trial.params:
            raise optuna.TrialPruned("Duplicate Trial Detected!")

    trial_alpha["settings"]["delay"] = p_delay
    trial_alpha["settings"]["universe"] = p_universe
    trial_alpha["settings"]["neutralization"] = p_neutralization
    trial_alpha["settings"]["maxTrade"] = p_maxTrade

    alpha_id = brain.Alpha.simulate(brain_session, trial_alpha)
    result = brain.Alpha.simulation_result(brain_session, alpha_id)

    insample = result["is"]

    trial.set_user_attr("alpha_id", result["id"])
    trial.set_user_attr("sharpe", insample["sharpe"])
    trial.set_user_attr("fitness", insample["fitness"])
    trial.set_user_attr("turnover", insample["turnover"])
    trial.set_user_attr("returns", insample["returns"])
    trial.set_user_attr("drawdown", insample["drawdown"])

    failed_checks = [
        check["name"] for check in insample["checks"] if check["result"] == "FAIL"
    ]
    trial.set_user_attr(
        "failed_checks", ",".join(failed_checks) if failed_checks else "NONE"
    )

    passed_checks = [
        check["name"] for check in insample["checks"] if check["result"] == "PASS"
    ]
    trial.set_user_attr(
        "passed_checks", ",".join(passed_checks) if passed_checks else "NONE"
    )

    warnings = [
        check["name"] for check in insample["checks"] if check["result"] == "WARNING"
    ]
    trial.set_user_attr("warnings", ",".join(warnings) if warnings else "NONE")

    score = reward.net_calmar_ratio(insample)
    return round(score, 2)


"""
Journal Storage is better to use than SQLite when considering multiple workers
but unfortunately have to use Administrative Privileges in Windows to run the script
sudo comes in clutch though
"""
storage = JournalStorage(JournalFileBackend(f"./studies/{study_file_name}.log"))
sampler = GridSampler(search_space)
study = optuna.create_study(
    study_name="settings_optimizer",
    direction="maximize",
    storage=storage,
    sampler=sampler,
)

total_combinations = math.prod([len(v) for v in search_space.values()])
print(f"Starting Search: {total_combinations} combinations.")

start_time = datetime.datetime.now()

try:
    study.optimize(objective, n_jobs=5)  # Maximum Concurrent Simulations on BRAIN is 5
except KeyboardInterrupt:
    print("Study interrupted by user...")

elapsed = datetime.datetime.now() - start_time
print(f"Total study time: {elapsed}")
input()
