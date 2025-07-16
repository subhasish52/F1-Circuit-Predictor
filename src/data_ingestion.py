# -*- coding: utf-8 -*-
import fastf1
import os
import pandas as pd
from datetime import datetime
from time import sleep

# Setup FastF1 cache
os.makedirs('data/raw/cache', exist_ok=True)
fastf1.Cache.enable_cache('data/raw/cache')

# Define years and session types
YEARS = list(range(2018, datetime.now().year + 1))  # From 2021 to current year
SESSION_TYPES = ['R', 'Q']  # Race and Qualifying


def safe_filename(name):
    """Make filename safe by replacing unsafe characters"""
    return name.replace(" ", "_").replace("/", "-")


def download_all():
    all_laps, all_results, all_weather = [], [], []
    drivers = []

    for year in YEARS:
        print(f"\n➡️ Season: {year}")
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"⚠️ Could not fetch schedule for {year}: {e}")
            continue

        for _, event in schedule.iterrows():
            gp = event['EventName']
            for sess in SESSION_TYPES:
                folder = f"data/raw/{year}_{safe_filename(gp)}_{sess}"
                laps_file = f"{folder}/laps.csv"

                # Skip if data already exists
                if os.path.exists(laps_file):
                    print(f"🟡 Skipping {year}–{gp}–{sess}, data already exists.")
                    continue

                try:
                    print(f"⏱ Downloading {year}–{gp}–{sess}")
                    session = fastf1.get_session(year, gp, sess)
                    session.load()

                    laps = session.laps
                    results = session.results
                    weather = session.weather_data

                    os.makedirs(folder, exist_ok=True)
                    laps.to_csv(laps_file, index=False)
                    results.to_csv(f"{folder}/results.csv", index=False)
                    weather.to_csv(f"{folder}/weather.csv", index=False)

                    # Add metadata columns
                    laps['Year'], laps['GP'], laps['Session'] = year, gp, sess
                    results['Year'], results['GP'], results['Session'] = year, gp, sess

                    # Append data
                    all_laps.append(laps)
                    all_results.append(results)
                    all_weather.append(weather)
                    drivers += results['Abbreviation'].astype(str).tolist()
                    sleep(1)

                except Exception as ex:
                    print(f"❌ Skipped {year}-{gp}-{sess}: {ex}")

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)

    if all_laps:
        df_laps = pd.concat(all_laps, ignore_index=True).drop_duplicates()
        df_laps.to_csv("data/processed/all_laps.csv", index=False)

    if all_results:
        df_results = pd.concat(all_results, ignore_index=True).drop_duplicates()
        df_results.to_csv("data/processed/all_results.csv", index=False)

    if all_weather:
        df_weather = pd.concat(all_weather, ignore_index=True).drop_duplicates()
        df_weather.to_csv("data/processed/all_weather.csv", index=False)

    # Active drivers (appeared in at least 10 sessions)
    drv_counts = pd.Series(drivers).value_counts()
    active = drv_counts[drv_counts >= 10].index.tolist()
    pd.DataFrame({'Driver': active}).to_csv("data/processed/active_drivers.csv", index=False)

    print("\n✅ Download complete. All processed data saved.")


if __name__ == "__main__":
    download_all()