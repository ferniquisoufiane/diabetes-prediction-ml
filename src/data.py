from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "BRFSS_2025.xlsx"
)


def load_data(path=DATA_PATH):
    """Load the raw BRFSS dataset."""

    df = pd.read_excel(path)
    return df


def split_features_target(df):
    """Separate raw features and target."""

    X = df.drop(columns=["DIABETE4"])
    y = df["DIABETE4"]

    return X, y


if __name__ == "__main__":
    df = load_data()
    X, y = split_features_target(df)

    print("Dataset shape:", df.shape)
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print("Dataset loaded successfully!")