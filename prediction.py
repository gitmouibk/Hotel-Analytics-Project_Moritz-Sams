import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def prepare_model_data(df):
    """
    Creates X (features) and y (target) for the regression model.
    - Target: Score
    - Features: Hotel stars, Nr. rooms, Member years, Helpful votes, amenities
    - Standardize numeric variables only (the 4 numeric ones listed below)
    - Convert amenity flags to 0/1
    """

    # Columns required by the assignment
    numeric_cols = ["Hotel stars", "Nr. rooms", "Member years", "Helpful votes"]
    amenity_cols = ["Pool", "Gym", "Spa", "Casino", "Free internet", "Tennis court"]
    target_col = "Score"

    # Make a copy with only needed columns
    cols_needed = [target_col] + numeric_cols + amenity_cols
    data = df[cols_needed].copy()

    # Ensure numeric cols are numeric
    for c in numeric_cols + [target_col]:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # Amenities: convert True/False or YES/NO to 0/1
    for c in amenity_cols:
        if data[c].dtype == bool:
            data[c] = data[c].astype(int)
        else:
            data[c] = (
                data[c].astype(str).str.strip().str.upper().isin(["YES", "Y", "TRUE", "1"])
            ).astype(int)

    # Drop rows with missing values in any model column
    data = data.dropna()

    # Build X and y
    X = data[numeric_cols + amenity_cols]
    y = data[target_col]

    return X, y, numeric_cols, amenity_cols


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Step 16: Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def fit_linear_regression(X_train, y_train, numeric_cols):
    """
    Step 17: Fit a Linear Regression model.
    Standardize numeric variables only.
    Returns the trained model and the fitted scaler.
    """
    scaler = StandardScaler()

    # Copy so we don't change original dataframes
    X_train_scaled = X_train.copy()

    # Standardize only numeric columns
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_scaled[numeric_cols])

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler


def evaluate_model(model, scaler, X_test, y_test, numeric_cols):
    """
    Step 18: Compute R2 and MSE on test set.
    Returns (r2, mse, y_pred).
    """
    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_cols] = scaler.transform(X_test_scaled[numeric_cols])

    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return r2, mse, y_pred


def coefficients_table(model, feature_names):
    """
    Step 19: Return a table of coefficients.
    """
    coefs = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_
    }).sort_values("coefficient", ascending=False).reset_index(drop=True)

    return coefs


def plot_actual_vs_predicted(y_test, y_pred):
    """
    Step 21 (optional): Scatter plot of actual vs predicted score.
    """
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title("Actual vs Predicted Score")
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.show()
