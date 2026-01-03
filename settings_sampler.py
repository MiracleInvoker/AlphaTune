import argparse
import copy
import datetime
import json
from math import prod
from os import makedirs

import optuna
from optuna.samplers import GridSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import ace_lib as ace
import reward
from ace_extensions import disable_progress_bar, get_datafield, get_stored_session
from utils import extract_datafields

with open("region_consts.json", "r") as f:
    region_consts = json.load(f)  # Region Constants

with open("alpha.json", "r") as f:
    alpha = json.load(f)


region = alpha["settings"]["region"]
neutralizations = region_consts[region]["neutralization"]
neutralizations.remove("NONE")  # No Point in having Neutralization as NONE

datafields = extract_datafields(alpha["regular"])
brain_session = get_stored_session(duration=7200)
disable_progress_bar()


universes = None
delays = None
for datafield in datafields:
    datafield = get_datafield(brain_session, datafield)
    data = datafield["data"]

    current_universes = {item["universe"] for item in data if item["region"] == region}
    current_delays = {item["delay"] for item in data if item["region"] == region}

    if universes is None:
        universes = current_universes
        delays = current_delays

    else:
        universes.intersection_update(current_universes)
        delays.intersection_update(current_delays)


search_space = {
    "universe": sorted(universes),
    "delay": sorted(delays),
    "neutralization": neutralizations,
    "maxTrade": ["OFF", "ON"],
}


def objective(trial):

    # Make a copy of the Global Variable "alpha" so that the objective function is thread safe
    trial_alpha = copy.deepcopy(alpha)

    p_universe = trial.suggest_categorical("universe", search_space["universe"])
    p_delay = trial.suggest_categorical("delay", search_space["delay"])
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

    simulation_response = ace.simulate_single_alpha(brain_session, trial_alpha)
    alpha_id = simulation_response["alpha_id"]
    simulation_result = ace.get_simulation_result_json(brain_session, alpha_id)

    insample = simulation_result["is"]
    checks = insample["checks"]

    trial.set_user_attr("alpha_id", alpha_id)
    trial.set_user_attr("sharpe", insample["sharpe"])
    trial.set_user_attr("fitness", insample["fitness"])
    trial.set_user_attr("turnover", insample["turnover"])
    trial.set_user_attr("returns", insample["returns"])
    trial.set_user_attr("drawdown", insample["drawdown"])

    failed_checks = [check["name"] for check in checks if check["result"] == "FAIL"]
    trial.set_user_attr(
        "failed_checks", ",".join(failed_checks) if failed_checks else "NONE"
    )

    passed_checks = [check["name"] for check in checks if check["result"] == "PASS"]
    trial.set_user_attr(
        "passed_checks", ",".join(passed_checks) if passed_checks else "NONE"
    )

    warnings = [check["name"] for check in checks if check["result"] == "WARNING"]
    trial.set_user_attr("warnings", ",".join(warnings) if warnings else "NONE")

    score = reward.net_calmar_ratio(insample)
    return round(score, 2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Optimize Alpha Simulation Settings using Optuna GridSampler."
    )

    parser.add_argument(
        "--simulations",
        "-s",
        type=int,
        default=8,
        help="Number of concurrent simulations (n_jobs). Must be 1...8. Default: 8",  # Maximum Concurrent Simulations on BRAIN is 8
    )

    now = datetime.datetime.now()
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=now.strftime("%Y-%m-%d_%H-%M-%S"),
        help="Name of File under studies/ folder. Default: %Y-%m-%d_%H-%M-%S",
    )

    args = parser.parse_args()

    if not (1 <= args.simulations <= 8):
        parser.error("--simulations must be an integer between 1 and 8")

    simulations = args.simulations
    study_file_name = args.name

    """
    Journal Storage is better to use than SQLite when considering multiple workers
    but unfortunately have to use Administrative Privileges in Windows to run the script
    sudo comes in clutch though
    """
    makedirs(f"./studies", exist_ok=True)
    storage = JournalStorage(JournalFileBackend(f"./studies/{study_file_name}.log"))
    sampler = GridSampler(search_space)
    study = optuna.create_study(
        study_name=f"{region}_settings_sampler",
        direction="maximize",
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )

    total_combinations = prod([len(values) for values in search_space.values()])
    print(f"Total Combinations: {total_combinations}")
    print(f"Starting Search.")

    start_time = datetime.datetime.now()

    try:
        study.optimize(objective, n_jobs=simulations)
    except KeyboardInterrupt:
        print("Study interrupted by user...")

    elapsed = datetime.datetime.now() - start_time
    print(f"Total Study Time: {elapsed}")
    input("Press Enter to exit...")
