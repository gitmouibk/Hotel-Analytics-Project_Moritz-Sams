# ============================================================
# Load and Clean the dataset
# ============================================================
def load_and_clean_data():
    """
    Load and clean the Las Vegas TripAdvisor dataset inclduing the necessary libraries.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    df = pd.read_csv("data/LasVegasTripAdvisorReviews-Dataset.csv", sep=";")

    df_clean = df.copy()

    # 1) Drop fully empty rows (blank lines in the CSV)
    df_clean = df_clean.dropna(how="all")

    # 2) Drop fully empty columns
    df_clean = df_clean.dropna(axis=1, how="all")

    # 3) Strip whitespace from object columns
    obj_cols = df_clean.select_dtypes(include="object").columns
    df_clean[obj_cols] = df_clean[obj_cols].apply(lambda s: s.str.strip())

    # 4) Convert empty strings to NaN
    df_clean = df_clean.replace("", np.nan)

    # 5) Standardize YES/NO columns to True/False
    yn_cols = ["Pool", "Gym", "Tennis court", "Spa", "Casino", "Free internet", "status"]
    for c in yn_cols:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].str.upper().map({"YES": True, "NO": False})

    # 6) Convert numeric columns EXCEPT "Hotel stars"
    num_cols = [
        "Nr. reviews",
        "Nr. hotel reviews",
        "Helpful votes",
        "Score",
        "Nr. rooms",
        "Member years"
    ]
    for c in num_cols:
        if c in df_clean.columns:
            df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
    
    # 7) Remove invalid negative values in "Member years"
    if "Member years" in df_clean.columns:
        df_clean = df_clean[df_clean["Member years"] >= 0]


    # 8) Clean and convert "Hotel stars" explicitly
    if "Hotel stars" in df_clean.columns:
        s = df_clean["Hotel stars"].astype(str).str.strip()

        # Treat known empty-like strings as missing
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})

        # Convert European decimal comma to dot (e.g., '4,5' -> '4.5')
        s = s.str.replace(",", ".", regex=False)

        df_clean["Hotel stars"] = pd.to_numeric(s, errors="coerce")

    # 9) Drop column "status" (too few entries)
    if "status" in df_clean.columns:
        df_clean = df_clean.drop(columns=["status"])

    return df_clean
