import os
import pandas as pd

DATA_PATH = "data/raw"

def load_data():
    files = [f for f in os.listdir(DATA_PATH) if f.startswith("metadata")]

    all_data = []

    for file in files:
        path = os.path.join(DATA_PATH, file)
        df = pd.read_csv(path)
        df["category"] = file.split("_")[1]  # récupère la catégorie
        all_data.append(df)

    data = pd.concat(all_data, ignore_index=True)
    return data

if __name__ == "__main__":
    data = load_data()
    print(data.head())
    print("Nombre total d'exemples :", len(data))