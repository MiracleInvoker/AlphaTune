import os
import pickle

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import ace_lib as ace


def get_stored_session():
    brain_session = ace.SingleSession()

    retry_strategy = Retry(
        total=5,  # Retry up to 5 times
        backoff_factor=1,  # Wait 1s, 2s, 4s... between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these errors
        allowed_methods=[
            "HEAD",
            "GET",
            "OPTIONS",
            "POST",
        ],  # Apply to GET (used by check_session_timeout)
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

    if time_to_live >= 7200:
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
