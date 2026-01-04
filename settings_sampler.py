import argparse
import copy
import datetime
import json
from math import prod
from os import makedirs

import optuna
from optuna.samplers import BruteForceSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

import reward
from ace_extensions import get_datafield, get_stored_session, simulate_single_alpha
from utils import extract_datafields, get_common_regions

with open("region_consts.json", "r") as f:
    region_consts = json.load(f)  # Region Constants

with open("alpha.json", "r") as f:
    alpha = json.load(f)


datafields = extract_datafields(alpha["regular"])
brain_session = get_stored_session(duration=7200)

data_list = []
for datafield in datafields:
    datafield = get_datafield(brain_session, datafield)
    data = datafield["data"]

    data_list.append(data)


common_regions = get_common_regions(data_list)

region = alpha["settings"]["region"]

neutralizations = region_consts[region]["neutralization"]
current_region = common_regions[region]

neutralizations.remove("NONE")  # No Point in having Neutralization as NONE


def objective(trial):

    # Make a copy of the Global Variable "alpha" so that the objective function is thread safe
    trial_alpha = copy.deepcopy(alpha)

    p_delay = trial.suggest_categorical("delay", list(current_region.keys()))
    p_universe = trial.suggest_categorical("universe", current_region[p_delay])
    p_neutralization = trial.suggest_categorical("neutralization", neutralizations)
    p_maxTrade = trial.suggest_categorical("maxTrade", ["ON", "OFF"])

    # Handling the Concurrency Race Condition which occurs at the end of the study
    for t in trial.study.trials:
        if t.number == trial.number:
            continue

        if t.params == trial.params and t.state != TrialState.FAIL:
            raise optuna.TrialPruned("Duplicate Trial Detected!")

    trial_alpha["settings"]["delay"] = p_delay
    trial_alpha["settings"]["universe"] = p_universe
    trial_alpha["settings"]["neutralization"] = p_neutralization
    trial_alpha["settings"]["maxTrade"] = p_maxTrade

    simulation_result = simulate_single_alpha(brain_session, trial_alpha)

    """
    Handles this Error:
    Your simulation has been running too long. If you are running simulations in batch, consider to reduce number of concurrent simulations or input data.
    """
    if "id" not in simulation_result:
        simulation_result = simulate_single_alpha(brain_session, trial_alpha)

    alpha_id = simulation_result["id"]
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
        description="Optimize Alpha Simulation Settings using Optuna BruteForceSampler."
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

    study_name = f"{region}_settings_sampler"

    """
    Journal Storage is better to use than SQLite when considering multiple workers
    but unfortunately have to use Administrative Privileges in Windows to run the script
    sudo comes in clutch though
    """
    makedirs(f"./studies", exist_ok=True)
    storage = JournalStorage(JournalFileBackend(f"./studies/{study_file_name}.log"))
    sampler = BruteForceSampler()
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )

    failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]
    if len(failed_trials) > 0:
        print(f"Found {len(failed_trials)} failed trials. Re-enqueueing them...")
        for t in failed_trials:
            study.enqueue_trial(t.params)

    print(f"Starting Search.")

    start_time = datetime.datetime.now()

    try:
        study.optimize(objective, n_jobs=simulations)

    except KeyboardInterrupt:
        print("Study interrupted by User...")

    elapsed = datetime.datetime.now() - start_time
    print(f"Study Name: {study_file_name}/{study_name}")
    print(f"Total Study Time: {elapsed}")
    input("Press Enter to Exit...")
