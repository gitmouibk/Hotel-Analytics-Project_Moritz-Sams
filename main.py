# ============================================================
# Hotel Analytics – Main Menu Program
# ============================================================
import os as os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cleaning import load_and_clean_data
from performance import (
    continent_summary,
    top5_hotels,
    top10_hotels_europe,
    bottom5_hotels_all_amenities,
)

from visualization import plot_score_histogram
from relationship import casino_score_comparison, plot_numeric_corr_heatmap
from prediction import (
    prepare_model_data,
    split_data,
    fit_linear_regression,
    evaluate_model,
    coefficients_table,
    plot_actual_vs_predicted,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def pause():
    input("\nPress Enter to return to the menu...")


def ensure_plots():
    os.makedirs("plots", exist_ok=True)


# ------------------------------------------------------------
# Section 1 – Program Info
# ------------------------------------------------------------
def section_1_info():
    print("\n=== Program Info ===")
    print(
        "This program analyzes hotel review data from TripAdvisor for Las Vegas hotels.\n"
        "It provides hotel rankings, visual insights, relationship analysis, and\n"
        "a linear regression model to predict review Score.\n"
        "The dataset used is LasVegasTripAdvisorReviews-Dataset.csv.\n"
        "Included analyses: hotel performance, visual storytelling, relationships, prediction."
    )
    pause()


# ------------------------------------------------------------
# Section 2 – Analytics
# ------------------------------------------------------------
def section_2_analytics(df):
    print("\n=== Analytics ===")

    # ------------------------------------------------------------
    # 1) Customer / Reviewer Overview
    # ------------------------------------------------------------
    print("\n--- Customer / Reviewer Overview ---")

    cont_counts, cont_stats, top3_score, top3_helpful = continent_summary(df)

    print("\nReview volume by continent:")
    print(cont_counts.to_string())

    print("\nAverage Score by continent (with review counts):")
    print(cont_stats[["User continent", "avg_score", "reviews"]].to_string(index=False))

    print("\nTop traveler types (review counts):")
    print(df["Traveler type"].value_counts().head(5).to_string())

    # ------------------------------------------------------------
    # 2) Hotel Performance Insights
    # ------------------------------------------------------------
    print("\n--- Hotel Performance Insights ---")

    print("\nTop 5 hotels by average Score:")
    print(top5_hotels(df).to_string(index=False))

    print("\nTop 10 hotels (Europe reviews only):")
    print(top10_hotels_europe(df).to_string(index=False))

    print("\nBottom hotels that offer major amenities (Gym/Spa/Tennis/Casino):")
    print(bottom5_hotels_all_amenities(df).to_string(index=False))

    # ------------------------------------------------------------
    # 3) Visual Data Storytelling
    # ------------------------------------------------------------
    print("\n--- Visual Data Storytelling ---")
    print("\nScore Distribution (Histogram):")
    plot_score_histogram(df)

    # ------------------------------------------------------------
    # 4) Relationship Exploration
    # ------------------------------------------------------------
    print("\n--- Relationship Exploration ---")

    print("\nCasino vs average Score:")
    print(casino_score_comparison(df).to_string(index=False))

    print("\nCorrelation Heatmap (Numeric Variables):")
    plot_numeric_corr_heatmap(df)

    # ------------------------------------------------------------
    # Section Summary
    # ------------------------------------------------------------
    print(
        "\nSection Summary:\n"
        "This section explored relationships in the data by first comparing average Scores for hotels\n"
        "with and without casinos. It then displayed a correlation heatmap of numeric variables,\n"
        "showing strong links between review activity variables and weak correlations with Score,\n"
        "which explains why predicting Scores is difficult."
    )

    pause()


# ------------------------------------------------------------
# Section 3 – Prediction
# ------------------------------------------------------------
def section_3_prediction(df):
    ensure_plots()
    print("\n=== Hotel Score Prediction ===")

    X, y, numeric_cols, _ = prepare_model_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model, scaler = fit_linear_regression(X_train, y_train, numeric_cols)
    r2, mse, y_pred = evaluate_model(model, scaler, X_test, y_test, numeric_cols)

    print("\nModel Performance (Test Set)")
    perf = pd.DataFrame([{
        "Train rows": len(X_train),
        "Test rows": len(X_test),
        "R²": r2,
        "MSE": mse
    }])
    print(perf.to_string(index=False))

    print("\nVariable Importance (Coefficients)")
    coef = coefficients_table(model, list(X.columns))
    print(coef.to_string(index=False))

    plot_actual_vs_predicted(y_test, y_pred)

    print(
        "\nSection Summary:\n"
        "A linear regression model was trained to predict review Score.\n"
        "R² and MSE show how well the model explains the data.\n"
        "Coefficients indicate which features increase or decrease the predicted Score."
    )
    pause()


# ------------------------------------------------------------
# Section 4 – Help
# ------------------------------------------------------------
def section_4_help():
    print("\n=== Help ===")
    print(
        "Enter a number from 1 to 5 and press Enter to choose a menu option.\n"
        "1 shows program information.\n"
        "2 runs hotel analytics and visual insights.\n"
        "3 runs the Score prediction model.\n"
        "Plots appear on screen and some are saved in the plots/ folder.\n"
        "Choose 5 to exit the program."
    )
    pause()


# ------------------------------------------------------------
# Main Menu (with required input rules)
# ------------------------------------------------------------
def main():
    df = load_and_clean_data()

    while True:
        print("\n" + "=" * 50)
        print("HOTEL ANALYTICS - MAIN MENU")
        print("=" * 50)
        print("1) Program Info")
        print("2) Analytics")
        print("3) Hotel Score Prediction")
        print("4) Help")
        print("5) Exit")

        user_input = input("Select an option (1–5): ")

        # Rule: whitespace → ignore
        if user_input.strip() == "":
            continue

        # Rule: must be integer
        try:
            choice = int(user_input)
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 5.")
            continue

        # Rule: negative number
        if choice < 0:
            print("Error: input must be between 1 and 5.")
            continue

        # Rule: >= 6
        if choice >= 6:
            print("Invalid number. Please enter a number between 1 and 5.")
            continue

        # Valid choices only
        match choice:
            case 1:
                section_1_info()
            case 2:
                section_2_analytics(df)
            case 3:
                section_3_prediction(df)
            case 4:
                section_4_help()
            case 5:
                print("\nProgram exited successfully. Goodbye!")
                break


if __name__ == "__main__":
    main()
