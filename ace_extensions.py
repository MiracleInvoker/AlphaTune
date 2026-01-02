import os
import pickle
import time

import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import ace_lib as ace


def get_stored_session(duration=1800):

    brain_session = ace.SingleSession()

    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    brain_session.mount("https://", adapter)
    brain_session.mount("http://", adapter)

    if os.path.exists("session.pkl"):
        try:
            with open("session.pkl", "rb") as f:
                session_data = pickle.load(f)
                brain_session.cookies.update(session_data["cookies"])
                brain_session.headers.update(session_data["headers"])

            print("Loaded stored session from disk.")

        except Exception as e:
            print(f"Failed to load session: {e}")

    time_to_live = ace.check_session_timeout(brain_session)

    if time_to_live >= duration:
        print(f"Session is valid. Expires in {time_to_live} seconds.")
        return brain_session

    print("Session expired or missing. Logging in...")
    brain_session = ace.start_session()

    with open("session.pkl", "wb") as f:
        pickle.dump(
            {"cookies": brain_session.cookies, "headers": brain_session.headers}, f
        )
    print("New session saved to disk.")

    return brain_session


def disable_progress_bar():

    class _DummyTqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total", None)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, n=1):
            pass

        def set_description(self, *a, **kw):
            pass

        def close(self):
            pass

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    ace.tqdm.tqdm = _DummyTqdm


def get_power_pool_corr(s: ace.SingleSession, alpha_id: str) -> pd.DataFrame:
    """
    Retrieve the power pool correlation data for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.

    Returns:
        pandas.DataFrame: A DataFrame containing the power pool correlation data.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """

    while True:
        result = s.get(
            ace.brain_api_url + "/alphas/" + alpha_id + "/correlations/power-pool"
        )
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))

        else:
            break

    if result.json().get("records", 0) == 0:
        ace.logger.warning(
            f"Failed to get power pool correlation for alpha_id {alpha_id}. {result.json()}"
        )
        return pd.DataFrame()

    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    power_pool_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(
        alpha_id=alpha_id
    )
    power_pool_corr_df["alpha_max_power_pool_corr"] = result.json()["max"]
    power_pool_corr_df["alpha_min_power_pool_corr"] = result.json()["min"]

    return power_pool_corr_df


def create_tag_list(s: ace.SingleSession, name: str, alpha_ids: list[str]) -> dict:

    response = s.post(
        ace.brain_api_url + "/tags",
        json={"type": "LIST", "name": name, "alphas": alpha_ids},
    )

    if response.status_code not in (200, 201):
        ace.logger.warning(
            f"Failed to create tag list '{name}'. "
            f"Status code: {response.status_code}, Response: {response.text}"
        )
        return {}

    return response.json()


def get_datafield(s: ace.SingleSession, datafield: str):
    """
    Retrieve Data Field.

    Args:
        s (SingleSession): An authenticated session object.
        data_field (str): Name of Data Field

    Returns:
        JSON: A JSON Object containing information about the Data Field.
    """

    resp = s.get(ace.brain_api_url + "/data-fields/" + datafield)
    resp_json = resp.json()

    return resp_json
