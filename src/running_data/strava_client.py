"""Utilitats per descarregar activitats i fluxos de l'API de Strava."""
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
    """Client lleuger per a la API REST de Strava.

    Paràmetres
    ----------
    access_token:
        Testimoni OAuth amb els permisos necessaris (``activity:read_all``
        per a activitats privades).
    request_timeout:
        Temps màxim aplicat a les crides ``requests``.
    max_retries:
        Nombre de reintents quan l'API retorna un error temporal
        com ara HTTP 429 (limit de velocitat) o respostes 5xx.
    backoff_factor:
        Factor base per al backoff exponencial entre reintents.
    """

    access_token: str
    request_timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 2.0

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

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
                    "L'API de Strava ha retornat %s. Reintent en %.1fs (intent %s/%s)",
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

    def get_athlete(self) -> Dict:
        """Retorna les metadades de l'atleta autenticat."""
        return self._request("GET", "athlete").json()

    def iter_activities(
        self,
        *,
        after: Optional[dt.datetime] = None,
        before: Optional[dt.datetime] = None,
        per_page: int = 30,
        max_pages: int = 10,
    ) -> Iterable[Dict]:
        """Genera les activitats de l'atleta autenticat.

        Paràmetres
        ----------
        after, before:
            Dates UTC opcionals per filtrar les activitats.
        per_page:
            Nombre d'activitats sol·licitades per crida (màxim 200).
        max_pages:
            Límit de seguretat per evitar descarregar tot l'historial per error.
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

    def get_activity_streams(
        self,
        activity_id: int,
        *,
        keys: Optional[List[str]] = None,
        key_by_type: bool = True,
        series_type: str = "time",
        resolution: str = "high",
    ) -> Dict[str, List]:
        """Descarrega els fluxos especificats d'una activitat.

        Paràmetres
        ----------
        activity_id:
            Identificador de l'activitat a consultar.
        keys:
            Llista de tipus de flux a demanar. Quan és ``None`` els tipus
            per defecte són ``["time", "heartrate", "cadence", "velocity_smooth", "grade_smooth"]``.
        key_by_type:
            Reprodueix el paràmetre de l'API de Strava. Si és ``True`` la
            resposta és un diccionari indexat pel tipus de flux.
        series_type:
            Un dels valors ``time``, ``distance`` o ``altitude``. ``time`` és el
            més habitual per treballar amb ritme i freqüència cardíaca.
        resolution:
            ``low`` (11 mostres), ``medium`` (51 mostres) o ``high`` (màxim).
            El flux original es reinterpola a la resolució demanada.
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
    """Retorna la freqüència cardíaca màxima trobada en un diccionari de fluxos."""

    hr_stream = stream.get("heartrate", {})
    if not hr_stream:
        return None
    data = hr_stream.get("data", [])
    return float(max(data)) if data else None
