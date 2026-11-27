
"""Utility functions for data preprocessing."""

import numpy as np
import pandas as pd


def detect_outliers(df, columns):
    """Detect potential outliers using the IQR method."""

    results = []

    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = (
            (df[column] < lower)
            | (df[column] > upper)
        )

        results.append({
            "Feature": column,
            "Lower Bound": lower,
            "Upper Bound": upper,
            "Outliers": int(outliers.sum())
        })

    return pd.DataFrame(results)


def binary_encode(X):
    """Convert BRFSS binary codes: 1 to 1, 2 to 0."""

    return 2 - np.asarray(X, dtype=float) + 1 - 1


def validate_data(X, name="Dataset"):
    """Check that transformed data contains finite values."""

    if not np.isfinite(np.asarray(X, dtype=float)).all():
        raise ValueError(
            f"{name} contains missing or infinite values."
        )

    print(f"{name}: validation successful.")


def display_class_distribution(y):
    """Display target counts and percentages."""

    counts = y.value_counts().sort_index()

    percentages = (
        y.value_counts(normalize=True)
        .sort_index() * 100
    )

    return pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2)
    })
