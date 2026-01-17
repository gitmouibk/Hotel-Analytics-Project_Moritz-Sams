import numpy as np
import matplotlib.pyplot as plt


# Step 12
def plot_member_years_vs_helpful_votes(df):
    data = df[["Member years", "Helpful votes"]].dropna()

    if len(data) > 1:
        corr = data["Member years"].corr(data["Helpful votes"])
    else:
        corr = float("nan")

    plt.figure()
    plt.scatter(data["Member years"], data["Helpful votes"], alpha=0.4)
    plt.title("Member Years vs Helpful Votes")
    plt.xlabel("Member years")
    plt.ylabel("Helpful votes")
    plt.show()

    return corr


# Step 13
def casino_score_comparison(df):
    data = df[["Casino", "Score"]].dropna()

    # Convert Casino to True/False (YES -> True, everything else -> False)
    casino_is_yes = (
        data["Casino"].astype(str).str.strip().str.upper().isin(["YES", "Y", "TRUE", "1"])
    )
    data = data.copy()
    data["casino"] = np.where(casino_is_yes, "YES", "NO")

    result = (
        data.groupby("casino", as_index=False)
            .agg(avg_score=("Score", "mean"), reviews=("Score", "count"))
            .sort_values("casino")
            .reset_index(drop=True)
    )
    return result


# Step 14
def traveler_type_scores_for_one_hotel(df, hotel_name=None):
    if hotel_name is None:
        hotel_name = df["Hotel name"].value_counts().idxmax()

    sub = df[df["Hotel name"] == hotel_name].copy()

    result = (
        sub.groupby("Traveler type", as_index=False)
           .agg(avg_score=("Score", "mean"), reviews=("Score", "count"))
           .sort_values(["avg_score", "reviews"], ascending=[False, False])
           .reset_index(drop=True)
    )

    return hotel_name, result


# Step 15
def plot_numeric_corr_heatmap(df):
    num = df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")
    corr = num.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Heatmap (Numeric Variables)")
    plt.show()

    return corr
