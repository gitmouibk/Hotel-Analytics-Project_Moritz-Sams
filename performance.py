import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1
def top5_hotels(df):
    result = (
        df.groupby("Hotel name", as_index=False)
          .agg(avg_score=("Score", "mean"), reviews=("Score", "count"))
          .sort_values(["avg_score", "reviews"], ascending=[False, False])
          .head(5)
          .reset_index(drop=True)
    )
    return result

# Step 2
def top10_hotels_europe(df):
    eu = df[df["User continent"] == "Europe"]
    result = (
        eu.groupby("Hotel name", as_index=False)
          .agg(avg_score=("Score", "mean"), reviews=("Score", "count"))
          .sort_values(["avg_score", "reviews"], ascending=[False, False])
          .head(10)
          .reset_index(drop=True)
    )
    return result

# Step 3
def bottom5_hotels_all_amenities(df):
    sub = df[
        (df["Tennis court"] == True) &
        (df["Gym"] == True) &
        (df["Spa"] == True) &
        (df["Casino"] == True)
    ]

    result = (
        sub.groupby("Hotel name", as_index=False)
           .agg(avg_score=("Score", "mean"), reviews=("Score", "count"))
           .sort_values(["avg_score", "reviews"], ascending=[True, False])
           .head(5)
           .reset_index(drop=True)
    )
    return result

# Step 4
def top10_hotels_review_volume_with_countries(df):
    top10 = (
        df.groupby("Hotel name")["Score"]
          .size()
          .sort_values(ascending=False)
          .head(10)
    )

    rows = []
    for hotel, n in top10.items():
        top_countries = (
            df[df["Hotel name"] == hotel]["User country"]
              .value_counts()
              .head(3)
        )

        top_countries_str = ", ".join(
            [country + " (" + str(count) + ")" for country, count in top_countries.items()]
        )

        rows.append({
            "Hotel name": hotel,
            "reviews": int(n),
            "top_countries": top_countries_str
        })

    return pd.DataFrame(rows)

# Step 5
def continent_summary(df):
    counts = df["User continent"].value_counts()

    stats = (
        df.groupby("User continent", as_index=False)
          .agg(
              avg_score=("Score", "mean"),
              avg_helpful=("Helpful votes", "mean"),
              reviews=("Score", "count")
          )
          .reset_index(drop=True)
    )

    top3_score = stats.sort_values("avg_score", ascending=False).head(3).reset_index(drop=True)
    top3_helpful = stats.sort_values("avg_helpful", ascending=False).head(3).reset_index(drop=True)

    return counts, stats, top3_score, top3_helpful

# Step 6
def no_free_internet_summary(df):
    no_net = df[df["Free internet"] == False]

    top3_countries = no_net["User country"].value_counts().head(3)

    top3_hotels = (
        no_net.groupby("Hotel name", as_index=False)
              .agg(avg_helpful=("Helpful votes", "mean"), reviews=("Helpful votes", "count"))
              .sort_values(["avg_helpful", "reviews"], ascending=[False, False])
              .head(3)
              .reset_index(drop=True)
    )

    return top3_countries, top3_hotels

# Step 7
def top5_hotels_by_rooms_meeting_conditions(df):
    hotel = (
        df.groupby("Hotel name", as_index=False)
          .agg(
              rooms=("Nr. rooms", "max"),
              stars=("Hotel stars", "max"),
              avg_score=("Score", "mean"),
              reviews=("Score", "count"),
              free_net=("Free internet", "any"),
              gym=("Gym", "any"),
              pool=("Pool", "any")
          )
    )

    ok = hotel[
        (hotel["stars"] >= 4) &
        (hotel["avg_score"] >= 4) &
        (hotel["free_net"] == True) &
        (hotel["gym"] == True) &
        (hotel["pool"] == True)
    ]

    result = (
        ok.sort_values("rooms", ascending=False)
          .head(5)
          .reset_index(drop=True)
    )

    # Keep or remove amenity columns here:
    return result[["Hotel name", "rooms", "stars", "avg_score", "reviews", "free_net", "gym", "pool"]]

# Step 8
def most_frequent_fields(df):
    return {
        "top_months": df["Review month"].value_counts().head(3),
        "top_weekdays": df["Review weekday"].value_counts().head(3),
        "top_traveler_type": df["Traveler type"].value_counts().head(1),
        "top_period": df["Period of stay"].value_counts().head(1),
    }