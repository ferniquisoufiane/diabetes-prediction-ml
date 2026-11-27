
"""Preprocessing pipeline for diabetes prediction."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from .utils import (
    detect_outliers,
    binary_encode,
    validate_data,
    display_class_distribution
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "BRFSS_2025.xlsx"
)

NUMERICAL_FEATURES = [
    "_AGE80",
    "_BMI5",
    "PHYSHLTH",
    "MENTHLTH",
    "SLEPTIM1",
    "FRUTDA2_",
    "VEGEDA2_"
]

LOG_FEATURES = [
    "FRUTDA2_",
    "VEGEDA2_"
]

ORDINAL_FEATURES = [
    "EDUCA",
    "INCOME3",
    "GENHLTH"
]

CATEGORICAL_FEATURES = [
    "SEXVAR",
    "EMPLOY1",
    "_SMOKER3"
]

BINARY_FEATURES = [
    "CVDCRHD4",
    "CVDSTRK3",
    "CHCKDNY2",
    "DIFFWALK",
    "EXERANY2",
    "DRNKANY6",
    "_HLTHPL2"
]

FEATURES = (
    NUMERICAL_FEATURES
    + ORDINAL_FEATURES
    + CATEGORICAL_FEATURES
    + BINARY_FEATURES
)


# ==========================================
# 2. DATA CLEANING
# ==========================================

def clean_data(df):
    """Clean raw BRFSS survey values."""

    df = df.copy()

    missing_codes = {
        "EDUCA": [9],
        "INCOME3": [77, 99],
        "EMPLOY1": [9],
        "GENHLTH": [7, 9],
        "PHYSHLTH": [77, 99],
        "MENTHLTH": [77, 99],
        "CVDCRHD4": [7, 9],
        "CVDSTRK3": [7, 9],
        "CHCKDNY2": [7, 9],
        "DIFFWALK": [7, 9],
        "_SMOKER3": [9],
        "EXERANY2": [7, 9],
        "DRNKANY6": [7, 9],
        "SLEPTIM1": [77, 99],
        "_HLTHPL2": [9]
    }

    for column, codes in missing_codes.items():
        df[column] = df[column].replace(
            codes, np.nan
        )

    df["PHYSHLTH"] = df["PHYSHLTH"].replace(88, 0)
    df["MENTHLTH"] = df["MENTHLTH"].replace(88, 0)

    df["_BMI5"] = df["_BMI5"] / 100

    for column in LOG_FEATURES:
        df[column] = (df[column] / 100).round(2)

    return df


# ==========================================
# 3. TARGET TRANSFORMATION
# ==========================================

def transform_target(df):
    """Convert diabetes responses into binary labels."""

    df = df[
        df["DIABETE4"].isin([1, 2, 3, 4])
    ].copy()

    target_mapping = {
        1: 1,
        2: 0,
        3: 0,
        4: 0
    }

    df["DIABETE4"] = (
        df["DIABETE4"]
        .map(target_mapping)
        .astype(int)
    )

    return df.rename(
        columns={"DIABETE4": "Diabetes"}
    )


# ==========================================
# 4. PREPARE FEATURES AND TARGET
# ==========================================

def prepare_data(df):
    """Clean raw data and return features and target."""

    df = clean_data(df)
    df = transform_target(df)

    X = df[FEATURES].copy()
    y = df["Diabetes"].copy()

    return X, y


# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

def split_data(X, y):
    """Create reproducible stratified data splits."""

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


# ==========================================
# 6. BUILD PREPROCESSOR
# ==========================================

def build_preprocessor():
    """Create an unfitted preprocessing pipeline."""

    regular_numerical = [
        col for col in NUMERICAL_FEATURES
        if col not in LOG_FEATURES
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median"
        )),
        ("scaler", StandardScaler())
    ])

    log_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median"
        )),
        ("log", FunctionTransformer(
            np.log1p,
            feature_names_out="one-to-one"
        )),
        ("scaler", StandardScaler())
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", FunctionTransformer(
            binary_encode,
            feature_names_out="one-to-one"
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                regular_numerical
            ),
            (
                "logarithmic",
                log_pipeline,
                LOG_FEATURES
            ),
            (
                "ordinal",
                ordinal_pipeline,
                ORDINAL_FEATURES
            ),
            (
                "binary",
                binary_pipeline,
                BINARY_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor


# ==========================================
# 7. MODEL PIPELINE
# ==========================================

def build_model_pipeline(model):
    """Combine preprocessing, balancing and a model."""

    return ImbPipeline([
        ("preprocessing", build_preprocessor()),
        ("oversampling", RandomOverSampler(
            random_state=RANDOM_STATE
        )),
        ("model", model)
    ])


# ==========================================
# 8. TEST PREPROCESSING
# ==========================================

def main():
    """Test the preprocessing workflow."""

    df = pd.read_excel(DATA_PATH)

    print("Raw dataset shape:", df.shape)

    X, y = prepare_data(df)

    print("Cleaned features shape:", X.shape)

    X_train, X_test, y_train, y_test = (
        split_data(X, y)
    )

    print("\n===== TRAINING OUTLIERS =====")

    print(
        detect_outliers(
            X_train,
            NUMERICAL_FEATURES
        ).to_string(index=False)
    )

    print("\n===== BEFORE PREPROCESSING =====")

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    # Fit preprocessing only on training data
    preprocessor = build_preprocessor()

    X_train_processed = (
        preprocessor.fit_transform(X_train)
    )

    X_test_processed = (
        preprocessor.transform(X_test)
    )

    # Recover feature names
    feature_names = (
        preprocessor.get_feature_names_out()
    )

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )

    # Validate processed datasets
    validate_data(
        X_train_processed,
        "Training dataset"
    )

    validate_data(
        X_test_processed,
        "Testing dataset"
    )

    assert X_train_processed.columns.equals(
        X_test_processed.columns
    )

    assert X_train_processed.index.equals(
        y_train.index
    )

    assert X_test_processed.index.equals(
        y_test.index
    )

    # Balance training data for demonstration
    oversampler = RandomOverSampler(
        random_state=RANDOM_STATE
    )

    X_train_balanced, y_train_balanced = (
        oversampler.fit_resample(
            X_train_processed,
            y_train
        )
    )

    validate_data(
        X_train_balanced,
        "Balanced training dataset"
    )

    print("\n===== FINAL PREPROCESSING RESULTS =====")

    print("Training shape:",
          X_train_processed.shape)

    print("Balanced training shape:",
          X_train_balanced.shape)

    print("Testing shape:",
          X_test_processed.shape)

    print("\nTraining distribution before balancing:")
    print(display_class_distribution(y_train))

    print("\nTraining distribution after balancing:")
    print(display_class_distribution(
        y_train_balanced
    ))

    print("\nTesting distribution:")
    print(display_class_distribution(y_test))

    print("\nFinal feature names:")
    print(feature_names.tolist())

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()
