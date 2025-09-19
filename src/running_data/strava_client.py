"""Utilities for downloading activities and streams from the Strava API."""
from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import time
from typing import Dict, Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)

STRAVA_API_BASE = "https://www.strava.com/api/v3"


@dataclasses.dataclass
class StravaClient:
    """Thin wrapper around the Strava REST API.

    Parameters
    ----------
    access_token:
        OAuth access token with the required scopes (``activity:read_all``
        for private activities).
    request_timeout:
        Timeout applied to the underlying ``requests`` calls.
    max_retries:
        Number of retries performed when the API returns a temporary error
        such as HTTP 429 (rate limit) or 5xx responses.
    backoff_factor:
        Base factor for the exponential backoff between retries.
    """

    access_token: str
    request_timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 2.0

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{STRAVA_API_BASE}/{path.lstrip('/')}"
        headers = kwargs.pop("headers", {})
        headers.update(self._headers())

        for attempt in range(1, self.max_retries + 1):
            resp = requests.request(
                method,
                url,
                headers=headers,
                timeout=self.request_timeout,
                **kwargs,
            )

            if resp.status_code in {429, 500, 502, 503, 504}:
                wait = self.backoff_factor ** attempt
                logger.warning(
                    "Strava API returned %s. Retrying in %.1fs (attempt %s/%s)",
                    resp.status_code,
                    wait,
                    attempt,
                    self.max_retries,
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp

        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------
    def get_athlete(self) -> Dict:
        """Return metadata about the authenticated athlete."""
        return self._request("GET", "athlete").json()

    # ------------------------------------------------------------------
    def iter_activities(
        self,
        *,
        after: Optional[dt.datetime] = None,
        before: Optional[dt.datetime] = None,
        per_page: int = 30,
        max_pages: int = 10,
    ) -> Iterable[Dict]:
        """Yield activities for the authenticated athlete.

        Parameters
        ----------
        after, before:
            Optional UTC datetimes used to filter the activities.
        per_page:
            Number of activities requested per API call (max 200).
        max_pages:
            Safety limit that prevents accidental full history downloads.
        """

        params: Dict[str, object] = {"per_page": per_page}
        if after is not None:
            params["after"] = int(after.timestamp())
        if before is not None:
            params["before"] = int(before.timestamp())

        page = 1
        while page <= max_pages:
            params["page"] = page
            resp = self._request("GET", "athlete/activities", params=params)
            activities = resp.json()
            if not activities:
                break
            for act in activities:
                yield act
            page += 1

    # ------------------------------------------------------------------
    def get_activity_streams(
        self,
        activity_id: int,
        *,
        keys: Optional[List[str]] = None,
        key_by_type: bool = True,
        series_type: str = "time",
        resolution: str = "high",
    ) -> Dict[str, List]:
        """Download the specified stream keys for an activity.

        Parameters
        ----------
        activity_id:
            Identifier of the activity to query.
        keys:
            List of stream types to request. When ``None`` the default keys are
            ``["time", "heartrate", "cadence", "velocity_smooth", "grade_smooth"]``.
        key_by_type:
            Matches the Strava API parameter. When ``True`` the response is a
            dictionary keyed by stream type.
        series_type:
            One of ``time``, ``distance`` or ``altitude``. ``time`` is the most
            common when working with pace/HR data.
        resolution:
            ``low`` (11 samples), ``medium`` (51 samples) or ``high`` (max). The
            user supplied stream is interpolated to the requested resolution.
        """

        if keys is None:
            keys = ["time", "heartrate", "cadence", "velocity_smooth", "grade_smooth"]

        params = {
            "key_by_type": str(key_by_type).lower(),
            "series_type": series_type,
            "resolution": resolution,
        }
        if keys:
            params["keys"] = ",".join(keys)

        resp = self._request(
            "GET",
            f"activities/{activity_id}/streams",
            params=params,
        )
        return resp.json()


def infer_fcmax_from_stream(stream: Dict[str, List]) -> Optional[float]:
    """Return the maximum heart rate found in a stream dictionary."""

    hr_stream = stream.get("heartrate", {})
    if not hr_stream:
        return None
    data = hr_stream.get("data", [])
    return float(max(data)) if data else None
