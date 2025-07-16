import os
import pandas as pd
from glob import glob
from tqdm import tqdm

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def is_valid_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        return not df.empty
    except Exception as e:
        print(f"⚠️ Skipping invalid file: {filepath} ({e})")
        return False

def rebuild_processed_data():
    all_laps, all_results, all_weather = [], [], []

    # ✅ Use a more robust method to scan all folders inside raw/
    folders = [f.path for f in os.scandir(RAW_DIR) if f.is_dir()]
    print(f"🔍 Found {len(folders)} session folders")

    for folder in tqdm(folders, desc="Processing sessions"):
        try:
            year_gp_sess = os.path.basename(folder)
            parts = year_gp_sess.split("_")
            year = parts[0]
            sess = parts[-1]
            gp = "_".join(parts[1:-1])  # handles names like Las_Vegas_Grand_Prix

            lap_path = os.path.join(folder, "laps.csv")
            result_path = os.path.join(folder, "results.csv")
            weather_path = os.path.join(folder, "weather.csv")

            if is_valid_csv(lap_path):
                df = pd.read_csv(lap_path)
                df['Year'], df['GP'], df['Session'] = year, gp, sess
                all_laps.append(df)

            if is_valid_csv(result_path):
                df = pd.read_csv(result_path)
                df['Year'], df['GP'], df['Session'] = year, gp, sess
                all_results.append(df)

            if is_valid_csv(weather_path):
                df = pd.read_csv(weather_path)
                df['Year'], df['GP'], df['Session'] = year, gp, sess
                all_weather.append(df)

        except Exception as ex:
            print(f"❌ Skipping {folder}: {ex}")

    print("\n💾 Writing merged processed files...")
    if all_laps:
        pd.concat(all_laps, ignore_index=True).to_csv(f"{PROCESSED_DIR}/all_laps.csv", index=False)
    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(f"{PROCESSED_DIR}/all_results.csv", index=False)
    if all_weather:
        pd.concat(all_weather, ignore_index=True).to_csv(f"{PROCESSED_DIR}/all_weather.csv", index=False)

    if all_results:
        drivers = pd.concat(all_results)['Abbreviation'].astype(str).value_counts()
        active = drivers[drivers >= 10].index.tolist()
        pd.DataFrame({'Driver': active}).to_csv(f"{PROCESSED_DIR}/active_drivers.csv", index=False)

    print("\n✅ Finished rebuilding processed data.")

if __name__ == "__main__":
    rebuild_processed_data()
