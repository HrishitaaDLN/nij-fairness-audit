from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "nij_challenge2021_full_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

TARGET_CANDIDATES = [
    "Recidivism_Arrest_Year1",
    "recidivism_arrest_year1",
]

RACE_CANDIDATES = ["Race", "race", "RACE"]
GENDER_CANDIDATES = ["Gender", "gender", "Sex", "sex"]
AGE_CANDIDATES = ["Age_at_Release", "age_at_release", "Age", "age"]
ID_CANDIDATES = ["ID", "id", "Id"]

ALL_TARGET_CANDIDATES = [
    "Recidivism_Arrest_Year1",
    "Recidivism_Arrest_Year2",
    "Recidivism_Arrest_Year3",
    "Recidivism_Within_3years",
    "recidivism_arrest_year1",
    "recidivism_arrest_year2",
    "recidivism_arrest_year3",
    "recidivism_within_3years",
]

# Explicit Year-1 drop list: use release-time features only
YEAR1_POST_RELEASE_DROP_CANDIDATES = [
    "Violations_ElectronicMonitoring",
    "Violations_Instruction",
    "Violations_InstructionsNotFollowed",
    "Violations_FailToReport",
    "Violations_MoveWithoutPermission",
    "Delinquency_Reports",
    "Program_Attendances",
    "Program_UnexcusedAbsences",
    "Residence_Changes",
    "Avg_Days_per_DrugTest",
    "DrugTests_THC_Positive",
    "DrugTests_Cocaine_Positive",
    "DrugTests_Meth_Positive",
    "DrugTests_Other_Positive",
    "Percent_Days_Employed",
    "Jobs_Per_Year",
    "Employment_Exempt",
]

AGE_BUCKET_ORDER = [
    "18-22",
    "23-27",
    "28-32",
    "33-37",
    "38-42",
    "43-47",
    "48 or older",
]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Put your CSV in data/nij_challenge2021_full_dataset.csv"
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")
    return df


def find_first_existing(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find {label}. Tried: {candidates}")


def find_optional_existing(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_binary_target(series: pd.Series) -> pd.Series:
    unique_values = set(series.dropna().unique())
    if unique_values.issubset({0, 1}):
        return series.astype(int)

    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "y": 1,
        "n": 0,
        "recidivated": 1,
        "not recidivated": 0,
    }

    lowered = series.astype(str).str.strip().str.lower()
    converted = lowered.map(mapping)

    if converted.isna().any():
        bad_values = sorted(series[converted.isna()].astype(str).unique().tolist())
        raise ValueError(
            "Target column could not be converted cleanly to binary 0/1.\n"
            f"Unexpected values: {bad_values}"
        )
    return converted.astype(int)


def get_column_names(df: pd.DataFrame) -> Dict[str, str | None]:
    return {
        "target": find_first_existing(df, TARGET_CANDIDATES, "Year-1 target column"),
        "race": find_first_existing(df, RACE_CANDIDATES, "race column"),
        "gender": find_first_existing(df, GENDER_CANDIDATES, "gender column"),
        "age": find_first_existing(df, AGE_CANDIDATES, "age column"),
        "id": find_optional_existing(df, ID_CANDIDATES),
    }


def resolve_explicit_drop_columns(df: pd.DataFrame, cols: Dict[str, str | None]) -> List[str]:
    drop_cols: List[str] = []

    for col in ALL_TARGET_CANDIDATES:
        if col in df.columns:
            drop_cols.append(col)

    for col in YEAR1_POST_RELEASE_DROP_CANDIDATES:
        if col in df.columns:
            drop_cols.append(col)

    if cols["id"] is not None and cols["id"] in df.columns:
        drop_cols.append(cols["id"])

    return sorted(set(drop_cols))


def save_basic_eda(df: pd.DataFrame, cols: Dict[str, str | None]) -> None:
    target = cols["target"]
    race = cols["race"]
    gender = cols["gender"]
    age = cols["age"]

    print("\n===== BASIC INFO =====")
    print("Shape:", df.shape)

    print("\n===== MISSING VALUES (TOP 20) =====")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\n===== CLASS BALANCE =====")
    print(df[target].value_counts(dropna=False))
    print("\nTarget mean:")
    print(df[target].mean())

    print("\n===== SUBGROUP COUNTS =====")
    print(df.groupby([race, gender]).size())

    print("\n===== YEAR-1 RECIDIVISM RATE BY RACE =====")
    print(df.groupby(race)[target].mean().sort_values(ascending=False))

    print("\n===== YEAR-1 RECIDIVISM RATE BY GENDER =====")
    print(df.groupby(gender)[target].mean().sort_values(ascending=False))

    print("\n===== YEAR-1 RECIDIVISM RATE BY RACE x GENDER =====")
    print(df.groupby([race, gender])[target].mean().sort_values(ascending=False))

    # Save EDA tables
    df.groupby(race)[target].mean().reset_index(name="year1_rate").to_csv(
        OUTPUT_DIR / "eda_recidivism_by_race.csv",
        index=False,
    )
    df.groupby(gender)[target].mean().reset_index(name="year1_rate").to_csv(
        OUTPUT_DIR / "eda_recidivism_by_gender.csv",
        index=False,
    )
    df.groupby([race, gender])[target].mean().reset_index(name="year1_rate").to_csv(
        OUTPUT_DIR / "eda_recidivism_by_race_gender.csv",
        index=False,
    )
    df.groupby([race, gender]).size().reset_index(name="count").to_csv(
        OUTPUT_DIR / "eda_group_counts.csv",
        index=False,
    )

    # Race x Gender recidivism rate plot
    subgroup_rates = df.groupby([race, gender])[target].mean().reset_index()
    subgroup_rates["group"] = (
        subgroup_rates[race].astype(str) + " | " + subgroup_rates[gender].astype(str)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(subgroup_rates["group"], subgroup_rates[target])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Year-1 recidivism rate")
    plt.title("Year-1 Recidivism Rate by Race x Gender")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_recidivism_by_race_gender.png", dpi=200)
    plt.close()

    # Age bucket bar chart
    age_counts = df[age].value_counts(dropna=False)
    ordered_index = [bucket for bucket in AGE_BUCKET_ORDER if bucket in age_counts.index]
    remaining = [idx for idx in age_counts.index if idx not in ordered_index]
    age_counts = age_counts.reindex(ordered_index + remaining)

    plt.figure(figsize=(10, 5))
    plt.bar(age_counts.index.astype(str), age_counts.values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.xlabel(age)
    plt.title("Age Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_age_distribution.png", dpi=200)
    plt.close()

    age_counts.reset_index().rename(
        columns={"index": age, age: "count"}
    ).to_csv(OUTPUT_DIR / "eda_age_counts.csv", index=False)


def build_feature_matrix(
    df: pd.DataFrame,
    cols: Dict[str, str | None],
    include_sensitive: bool,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    target_col = cols["target"]
    race_col = cols["race"]
    gender_col = cols["gender"]

    y = normalize_binary_target(df[target_col].copy()).reset_index(drop=True)

    drop_cols = resolve_explicit_drop_columns(df, cols)

    if not include_sensitive:
        for col in [race_col, gender_col]:
            if col in df.columns:
                drop_cols.append(col)

    drop_cols = sorted(set(drop_cols))
    X = df.drop(columns=drop_cols).copy().reset_index(drop=True)

    return X, y, drop_cols


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def get_models() -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def compute_metrics(
    y_true: pd.Series,
    prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auroc": roc_auc_score(y_true, prob),
        "auprc": average_precision_score(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
    }


def subgroup_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "n": len(y_true),
        "base_rate": float(np.mean(y_true)),
        "positive_pred_rate": float(np.mean(y_pred)),
        "TPR": tpr,
        "FPR": fpr,
        "FNR": fnr,
        "TNR": tnr,
    }


def make_subgroup_audit(
    meta_df: pd.DataFrame,
    y_true: pd.Series,
    prob: np.ndarray,
    race_col: str,
    gender_col: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    pred = (prob >= threshold).astype(int)

    audit_df = meta_df.copy()
    audit_df["y_true"] = y_true.values
    audit_df["y_prob"] = prob
    audit_df["y_pred"] = pred

    rows = []
    for (race, gender), g in audit_df.groupby([race_col, gender_col]):
        m = subgroup_metrics(g["y_true"], g["y_pred"].values)
        m["Race"] = race
        m["Gender"] = gender
        rows.append(m)

    subgroup_df = pd.DataFrame(rows)[
        ["Race", "Gender", "n", "base_rate", "positive_pred_rate", "TPR", "FPR", "FNR", "TNR"]
    ].sort_values(["Race", "Gender"])

    return subgroup_df


def run_cross_validated_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    meta_df: pd.DataFrame,
    race_col: str,
    gender_col: str,
    setting_name: str,
    n_splits: int = 5,
    threshold: float = 0.5,
) -> None:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = get_models()

    summary_rows = []
    fold_rows = []
    subgroup_audits: Dict[str, pd.DataFrame] = {}

    for model_name, base_model in models.items():
        oof_prob = np.zeros(len(y), dtype=float)

        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_valid = y.iloc[valid_idx]

            preprocessor = make_preprocessor(X_train)
            model = clone(base_model)

            pipe = Pipeline(
                steps=[
                    ("prep", preprocessor),
                    ("model", model),
                ]
            )

            pipe.fit(X_train, y_train)
            valid_prob = pipe.predict_proba(X_valid)[:, 1]
            oof_prob[valid_idx] = valid_prob

            fold_metric = compute_metrics(y_valid, valid_prob, threshold=threshold)
            fold_metric["fold"] = fold_idx
            fold_metric["model"] = model_name
            fold_rows.append(fold_metric)

        overall = compute_metrics(y, oof_prob, threshold=threshold)
        overall["model"] = model_name

        fold_df_model = pd.DataFrame([r for r in fold_rows if r["model"] == model_name])

        overall["cv_mean_auroc"] = fold_df_model["auroc"].mean()
        overall["cv_std_auroc"] = fold_df_model["auroc"].std(ddof=1)
        overall["cv_mean_auprc"] = fold_df_model["auprc"].mean()
        overall["cv_std_auprc"] = fold_df_model["auprc"].std(ddof=1)
        overall["cv_mean_brier"] = fold_df_model["brier"].mean()
        overall["cv_std_brier"] = fold_df_model["brier"].std(ddof=1)

        summary_rows.append(overall)

        subgroup_audits[model_name] = make_subgroup_audit(
            meta_df=meta_df,
            y_true=y,
            prob=oof_prob,
            race_col=race_col,
            gender_col=gender_col,
            threshold=threshold,
        )

    summary_df = pd.DataFrame(summary_rows)

    # Multi-metric model ranking: AUROC high, AUPRC high, Brier low
    summary_df["rank_auroc"] = summary_df["auroc"].rank(ascending=False, method="min")
    summary_df["rank_auprc"] = summary_df["auprc"].rank(ascending=False, method="min")
    summary_df["rank_brier"] = summary_df["brier"].rank(ascending=True, method="min")
    summary_df["selection_rank"] = (
        summary_df["rank_auroc"] + summary_df["rank_auprc"] + summary_df["rank_brier"]
    ) / 3.0

    summary_df = summary_df.sort_values(
        ["selection_rank", "auroc", "auprc", "brier"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    fold_df = pd.DataFrame(fold_rows).sort_values(["model", "fold"]).reset_index(drop=True)

    summary_path = OUTPUT_DIR / f"step1_{setting_name}_model_summary.csv"
    fold_path = OUTPUT_DIR / f"step1_{setting_name}_fold_metrics.csv"

    summary_df.to_csv(summary_path, index=False)
    fold_df.to_csv(fold_path, index=False)

    best_model_name = summary_df.iloc[0]["model"]
    best_audit_df = subgroup_audits[best_model_name]
    best_audit_df.to_csv(
        OUTPUT_DIR / f"step1_{setting_name}_best_subgroup_audit.csv",
        index=False,
    )

    print(f"\n===== {setting_name.upper()} : MODEL SUMMARY =====")
    print(summary_df[[
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "brier",
        "selection_rank",
    ]])

    print(f"\nBest model for {setting_name}: {best_model_name}")
    print(f"\n===== {setting_name.upper()} : BEST SUBGROUP AUDIT (threshold=0.5) =====")
    print(best_audit_df)


def main() -> None:
    ensure_output_dir()

    print("Loading data...")
    df = load_data(DATA_PATH)

    cols = get_column_names(df)
    target_col = cols["target"]
    race_col = cols["race"]
    gender_col = cols["gender"]

    print("\nDetected columns:")
    for key, value in cols.items():
        print(f"{key}: {value}")

    df[target_col] = normalize_binary_target(df[target_col])

    save_basic_eda(df, cols)

    explicit_drop_cols = resolve_explicit_drop_columns(df, cols)
    print("\nExplicit Year-1 drop columns:")
    print(explicit_drop_cols)

    meta_df = df[[race_col, gender_col, cols["age"]]].copy().reset_index(drop=True)

    # Baseline A: with Race and Gender
    X_with_sensitive, y, dropped_with_sensitive = build_feature_matrix(
        df=df,
        cols=cols,
        include_sensitive=True,
    )
    print("\nFeature matrix WITH sensitive attributes:", X_with_sensitive.shape)
    print("Dropped columns:", dropped_with_sensitive)

    run_cross_validated_experiment(
        X=X_with_sensitive,
        y=y,
        meta_df=meta_df,
        race_col=race_col,
        gender_col=gender_col,
        setting_name="with_sensitive",
        n_splits=5,
        threshold=0.5,
    )

    # Baseline B: without Race and Gender
    X_without_sensitive, y, dropped_without_sensitive = build_feature_matrix(
        df=df,
        cols=cols,
        include_sensitive=False,
    )
    print("\nFeature matrix WITHOUT sensitive attributes:", X_without_sensitive.shape)
    print("Dropped columns:", dropped_without_sensitive)

    run_cross_validated_experiment(
        X=X_without_sensitive,
        y=y,
        meta_df=meta_df,
        race_col=race_col,
        gender_col=gender_col,
        setting_name="without_sensitive",
        n_splits=5,
        threshold=0.5,
    )

    print("\nSaved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()