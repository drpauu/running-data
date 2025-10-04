# -*- coding: utf-8 -*-
# filename: strava_export_fast.py

import os, time, statistics as stats, json
import requests, pandas as pd
from dotenv import load_dotenv

# ----------------- CONFIG -----------------
load_dotenv()
CLIENT_ID     = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
ACCESS_TOKEN  = os.getenv("STRAVA_ACCESS_TOKEN")
REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN")

API_BASE   = "https://www.strava.com/api/v3"
TOKEN_URL  = f"{API_BASE}/oauth/token"
ACTIVITIES_URL = f"{API_BASE}/athlete/activities"

# Config: quantes activitats processar per execució
BATCH_SIZE = 90
OUTFILE    = "activities_enriched.csv"
# ------------------------------------------

def headers(token): return {"Authorization": f"Bearer {token}"}

def refresh_token(refresh):
    r = requests.post(TOKEN_URL, data={
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh
    }, timeout=30)
    r.raise_for_status()
    return r.json()

def check_token(token):
    r = requests.get(f"{API_BASE}/athlete", headers=headers(token), timeout=15)
    return r.status_code == 200

def get_activities(token, page, per_page=200):
    r = requests.get(ACTIVITIES_URL, headers=headers(token),
                     params={"page": page,"per_page":per_page}, timeout=60)
    if r.status_code==429: raise RuntimeError("RATE_LIMIT")
    r.raise_for_status()
    return r.json()

def get_detail(token, act_id):
    r = requests.get(f"{API_BASE}/activities/{act_id}",
                     headers=headers(token),
                     params={"include_all_efforts":"false"}, timeout=60)
    if r.status_code==429: raise RuntimeError("RATE_LIMIT")
    if r.status_code==404: return {}
    r.raise_for_status()
    return r.json()

def get_streams(token, act_id):
    keys="time,distance,altitude,heartrate,cadence,watts,velocity_smooth"
    r = requests.get(f"{API_BASE}/activities/{act_id}/streams",
                     headers=headers(token),
                     params={"keys":keys,"key_by_type":"true"}, timeout=60)
    if r.status_code==429: raise RuntimeError("RATE_LIMIT")
    if r.status_code==404: return {}
    r.raise_for_status()
    return r.json()

def enrich(base, det, stm):
    row=dict(base)
    for k in ["device_name","average_cadence","average_watts",
              "weighted_average_watts","kilojoules","suffer_score","calories"]:
        row[k]=det.get(k)
    try:
        hr=stm.get("heartrate",{}).get("data")
        cad=stm.get("cadence",{}).get("data")
        pwr=stm.get("watts",{}).get("data")
        spd=stm.get("velocity_smooth",{}).get("data")
        row.update({
            "stream_avg_hr": float(stats.mean(hr)) if hr else None,
            "stream_max_hr": max(hr) if hr else None,
            "stream_avg_cad": float(stats.mean(cad)) if cad else None,
            "stream_avg_watts": float(stats.mean(pwr)) if pwr else None,
            "stream_max_watts": max(pwr) if pwr else None,
            "stream_avg_speed": float(stats.mean(spd)) if spd else None,
            "stream_max_speed": max(spd) if spd else None,
        })
    except: pass
    return row

def main():
    access, refresh = ACCESS_TOKEN, REFRESH_TOKEN
    if not check_token(access):
        td=refresh_token(refresh)
        access=td["access_token"]; refresh=td["refresh_token"]

    # carregar ja processats
    done_ids=set()
    if os.path.exists(OUTFILE):
        done_ids=set(pd.read_csv(OUTFILE,usecols=["id"])["id"].astype(str))

    # descarregar resum activitats (només ids + bàsics)
    acts=[]
    page=1
    while True:
        chunk=get_activities(access,page)
        if not chunk: break
        acts.extend(chunk); page+=1
        if len(chunk)<200: break
    print(f"[INFO] Trobades {len(acts)} activitats")

    # seleccionar no processades
    pending=[a for a in acts if str(a["id"]) not in done_ids]
    print(f"[INFO] Pendents {len(pending)}")

    rows=[]
    for idx,a in enumerate(pending[:BATCH_SIZE],1):
        aid=a["id"]
        base={"id":aid,"name":a.get("name"),"sport_type":a.get("sport_type") or a.get("type"),
              "start_date":a.get("start_date"),"distance_m":a.get("distance"),
              "moving_time":a.get("moving_time"),"elapsed_time":a.get("elapsed_time"),
              "elev_gain":a.get("total_elevation_gain")}
        try:
            det=get_detail(access,aid)
            stm=get_streams(access,aid)
            row=enrich(base,det,stm)
            rows.append(row)
        except RuntimeError as e:
            if "RATE_LIMIT" in str(e):
                print("[INFO] Rate limit, aturant…")
                break
        except Exception as e:
            print(f"[WARN] Act {aid} error: {e}")
        if idx%10==0: print(f"[INFO] {idx}/{BATCH_SIZE} processades")

    if rows:
        df=pd.DataFrame(rows)
        if os.path.exists(OUTFILE):
            df.to_csv(OUTFILE,mode="a",header=False,index=False)
        else:
            df.to_csv(OUTFILE,index=False)
        print(f"[INFO] Desa {len(df)} noves activitats → {OUTFILE}")
    else:
        print("[INFO] Cap activitat nova processada")

if __name__=="__main__":
    main()
