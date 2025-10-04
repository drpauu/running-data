# -*- coding: utf-8 -*-
# strava_to_csv.py
# Comentaris en català :)

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
ACCESS_TOKEN = os.getenv("STRAVA_ACCESS_TOKEN")
REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN")

API_BASE = "https://www.strava.com/api/v3"
TOKEN_URL = "https://www.strava.com/oauth/token"

def ensure_env():
    missing = [k for k, v in {
        "STRAVA_CLIENT_ID": CLIENT_ID,
        "STRAVA_CLIENT_SECRET": CLIENT_SECRET,
        "STRAVA_ACCESS_TOKEN": ACCESS_TOKEN,
        "STRAVA_REFRESH_TOKEN": REFRESH_TOKEN
    }.items() if not v]
    if missing:
        raise SystemExit(f"Falten variables d'entorn: {', '.join(missing)}")

def refresh_access_token(refresh_token: str) -> dict:
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    }
    r = requests.post(TOKEN_URL, data=payload, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"[REFRESH_FAIL] {r.status_code} - {r.text}")
    return r.json()

def get_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

def check_me(token: str):
    r = requests.get(f"{API_BASE}/athlete", headers=get_headers(token), timeout=15)
    return r

def get_activities(token: str):
    activities = []
    page = 1
    per_page = 200
    while True:
        r = requests.get(
            f"{API_BASE}/athlete/activities",
            headers=get_headers(token),
            params={"page": page, "per_page": per_page},
            timeout=30
        )
        if r.status_code == 429:
            time.sleep(60)
            continue
        if r.status_code == 401:
            raise PermissionError("TOKEN_EXPIRED_OR_INVALID")
        if r.status_code != 200:
            raise RuntimeError(f"[ACTIVITIES_FAIL] {r.status_code} - {r.text}")
        items = r.json()
        if not items:
            break
        activities.extend(items)
        page += 1
        time.sleep(0.2)
    return activities

def flatten(items: list) -> pd.DataFrame:
    rows = []
    for a in items:
        rows.append({
            "id": a.get("id"),
            "name": a.get("name"),
            "sport_type": a.get("sport_type") or a.get("type"),
            "start_date": a.get("start_date"),
            "distance_m": a.get("distance"),
            "moving_time_s": a.get("moving_time"),
            "elapsed_time_s": a.get("elapsed_time"),
            "elev_gain_m": a.get("total_elevation_gain"),
            "avg_speed_m_s": a.get("average_speed"),
            "max_speed_m_s": a.get("max_speed"),
            "avg_hr": a.get("average_heartrate"),
            "max_hr": a.get("max_heartrate"),
            "calories": a.get("calories"),
            "visibility": a.get("visibility"),
            "private": a.get("private"),
            "gear_id": a.get("gear_id"),
            "country": a.get("location_country"),
            "external_id": a.get("external_id"),
            "upload_id": a.get("upload_id"),
            "polyline": (a.get("map") or {}).get("summary_polyline")
        })
    return pd.DataFrame(rows)

def main():
    ensure_env()

    access = ACCESS_TOKEN
    refresh = REFRESH_TOKEN

    # 0) Prova el token actual amb /athlete
    r = check_me(access)
    if r.status_code == 401:
        print("Token vençut; refrescant…")
        try:
            td = refresh_access_token(refresh)
        except RuntimeError as e:
            # Mostra el missatge complet que envia Strava (motiu real del 401)
            raise SystemExit(
                "El refresh ha fallat. Causes típiques:\n"
                "- refresh_token caducat/rotat\n"
                "- client_secret canviat\n"
                "- app desautoritzada o sense scope\n"
                f"Detall Strava: {e}"
            )
        access = td["access_token"]
        # Strava pot rotar el refresh_token: actualitza’l en memòria
        refresh = td.get("refresh_token", refresh)

        # Torna a provar /athlete amb el token nou
        r = check_me(access)
        if r.status_code != 200:
            raise SystemExit(f"El token refrescat tampoc funciona: {r.status_code} - {r.text}")

    # 1) Baixa activitats
    try:
        acts = get_activities(access)
    except PermissionError:
        # Si entrem aquí, el token refrescat també ha fallat al /activities.
        raise SystemExit(
            "401 a /athlete/activities. Revisa que l'app tingui 'activity:read' o 'activity:read_all' i que l'usuari hagi reautoritzat."
        )

    if not acts:
        print("No s'han trobat activitats.")
        return

    df = flatten(acts)
    df.to_csv("activities.csv", index=False)
    print(f"Desades {len(df)} activitats a activities.csv")

if __name__ == "__main__":
    main()
