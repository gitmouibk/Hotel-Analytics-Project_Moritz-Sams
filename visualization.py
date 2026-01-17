import matplotlib.pyplot as plt

def plot_score_histogram(df):
    plt.figure()
    df["Score"].dropna().hist(bins=10)
    plt.title("Distribution of Review Scores")
    plt.xlabel("Score")
    plt.ylabel("Number of Reviews")
    plt.show(block=False)
    plt.pause(0.1)


def plot_score_by_traveler_type(df):
    data = df[["Traveler type", "Score"]].dropna()

    order = (
        data.groupby("Traveler type")["Score"]
            .median()
            .sort_values(ascending=False)
            .index
    )

    scores = [data[data["Traveler type"] == t]["Score"] for t in order]

    plt.figure(figsize=(10, 5))
    plt.boxplot(scores, labels=order)
    plt.title("Review Score by Traveler Type")
    plt.xlabel("Traveler Type")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.show()


def plot_rooms_by_stars(df):
    data = df[["Hotel stars", "Nr. rooms"]].dropna()
    stars = sorted(data["Hotel stars"].unique())

    rooms = [data[data["Hotel stars"] == s]["Nr. rooms"] for s in stars]

    plt.figure()
    plt.boxplot(rooms, labels=stars)
    plt.title("Number of Rooms by Hotel Star Rating")
    plt.xlabel("Hotel Stars")
    plt.ylabel("Number of Rooms")
    plt.show()
