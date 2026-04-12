"""
In Step 1, we prepared the NIJ dataset, selected Recidivism_Arrest_Year1 as the initial 
prediction target, removed the ID field, alternative recidivism outcome variables, and a 
set of likely post-release supervision/activity variables, conducted exploratory data analysis 
across demographic groups, trained baseline Logistic Regression and Random Forest models, and
generated subgroup-level performance summaries across race × gender intersections. The removed 
variables included Recidivism_Arrest_Year1, Recidivism_Arrest_Year2, Recidivism_Arrest_Year3, 
Recidivism_Within_3years, Violations_ElectronicMonitoring, Violations_Instruction, 
Violations_FailToReport, Violations_MoveWithoutPermission, Delinquency_Reports, 
Program_Attendances, Program_UnexcusedAbsences, Residence_Changes, Avg_Days_per_DrugTest, 
DrugTests_THC_Positive, DrugTests_Cocaine_Positive, DrugTests_Meth_Positive, DrugTests_Other_Positive, 
Jobs_Per_Year, Employment_Exempt, and ID."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "nij_challenge2021_full_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            f"Put your CSV in the data/ folder and name it 'nij_challenge2021_full_dataset.csv'"
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")
    return df


def find_first_existing(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find {label}. Tried: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def normalize_binary_target(series: pd.Series) -> pd.Series:
    """
    Convert common binary encodings to 0/1 safely.
    """
    if set(series.dropna().unique()).issubset({0, 1}):
        return series.astype(int)

    mapping = {
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "y": 1, "n": 0,
        "recidivated": 1, "not recidivated": 0,
    }

    lowered = series.astype(str).str.strip().str.lower()
    converted = lowered.map(mapping)

    if converted.isna().any():
        bad_values = sorted(series[converted.isna()].astype(str).unique().tolist())
        raise ValueError(
            f"Target column could not be converted cleanly to binary 0/1.\n"
            f"Unexpected values: {bad_values}"
        )
    return converted.astype(int)


def get_column_names(df: pd.DataFrame) -> Dict[str, str]:
    """
    Tries common column-name variants.
    Adjust these if your CSV uses different names.
    """
    target_col = find_first_existing(
        df,
        [
            "Recidivism_Arrest_Year1",
            "recidivism_arrest_year1",
            "RecidivismArrestYear1",
            "Return_to_Prison_Year1",
        ],
        "Year-1 target column",
    )

    race_col = find_first_existing(
        df,
        ["Race", "race", "RACE"],
        "race column",
    )

    gender_col = find_first_existing(
        df,
        ["Gender", "gender", "Sex", "sex"],
        "gender column",
    )

    age_col = find_first_existing(
        df,
        ["Age_at_Release", "age_at_release", "Age", "age"],
        "age column",
    )

    id_col = None
    for c in ["ID", "id", "Id"]:
        if c in df.columns:
            id_col = c
            break

    return {
        "target": target_col,
        "race": race_col,
        "gender": gender_col,
        "age": age_col,
        "id": id_col,
    }


def get_target_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    candidates = [
        "Recidivism_Within_3years",
        "Recidivism_Arrest_Year1",
        "Recidivism_Arrest_Year2",
        "Recidivism_Arrest_Year3",
        "recidivism_within_3years",
        "recidivism_arrest_year1",
        "recidivism_arrest_year2",
        "recidivism_arrest_year3",
    ]
    found = [c for c in candidates if c in df.columns]
    if target_col not in found:
        found.append(target_col)
    return sorted(set(found))


def guess_post_release_columns(df: pd.DataFrame) -> List[str]:
    """
    Remove columns that likely reflect supervision-period behavior.
    This is only a heuristic for Step 1.
    """
    keywords = [
        "Violation",
        "Violations",
        "DrugTest",
        "DrugTests",
        "Residence_Change",
        "Residence_Changes",
        "Employment",
        "Jobs",
        "Program",
        "Delinquency",
        "Days_per_DrugTest",
        "Report",
        "Reports",
        "Absence",
        "Absences",
        "THC",
        "Cocaine",
        "Meth",
    ]

    matched = []
    for col in df.columns:
        if any(k.lower() in col.lower() for k in keywords):
            matched.append(col)
    return matched


def basic_eda(df: pd.DataFrame, cols: Dict[str, str]) -> None:
    target = cols["target"]
    race = cols["race"]
    gender = cols["gender"]
    age = cols["age"]

    print("\n===== BASIC INFO =====")
    print("Shape:", df.shape)
    print("\nMissing values (top 20):")
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
        OUTPUT_DIR / "eda_recidivism_by_race.csv", index=False
    )
    df.groupby(gender)[target].mean().reset_index(name="year1_rate").to_csv(
        OUTPUT_DIR / "eda_recidivism_by_gender.csv", index=False
    )
    df.groupby([race, gender])[target].mean().reset_index(name="year1_rate").to_csv(
        OUTPUT_DIR / "eda_recidivism_by_race_gender.csv", index=False
    )

    # Plot subgroup rates
    subgroup_rates = (
        df.groupby([race, gender])[target]
        .mean()
        .reset_index()
    )
    subgroup_rates["group"] = subgroup_rates[race].astype(str) + " | " + subgroup_rates[gender].astype(str)

    plt.figure(figsize=(10, 5))
    plt.bar(subgroup_rates["group"], subgroup_rates[target])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Year-1 recidivism rate")
    plt.title("Year-1 Recidivism Rate by Race x Gender")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_recidivism_by_race_gender.png", dpi=200)
    plt.close()

    # Age histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df[age].dropna(), bins=30)
    plt.xlabel(age)
    plt.ylabel("Count")
    plt.title("Age Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_age_distribution.png", dpi=200)
    plt.close()


def build_feature_matrix(df: pd.DataFrame, cols: Dict[str, str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    target_col = cols["target"]
    id_col = cols["id"]

    y = normalize_binary_target(df[target_col].copy())

    target_cols = get_target_columns(df, target_col)
    post_release_cols = guess_post_release_columns(df)

    drop_cols = target_cols + post_release_cols
    if id_col is not None:
        drop_cols.append(id_col)

    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols).copy()

    return X, y, drop_cols


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
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


def evaluate_model(
    name: str,
    model,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Pipeline, np.ndarray, np.ndarray, Dict[str, float]]:
    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    results = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auroc": roc_auc_score(y_test, prob),
        "auprc": average_precision_score(y_test, prob),
        "brier": brier_score_loss(y_test, prob),
    }

    return pipe, prob, pred, results


def subgroup_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppr = np.mean(y_pred)

    return {
        "n": len(y_true),
        "base_rate": float(np.mean(y_true)),
        "positive_pred_rate": float(ppr),
        "TPR": tpr,
        "FPR": fpr,
        "FNR": fnr,
        "TNR": tnr,
    }


def save_subgroup_audit(
    audit_df: pd.DataFrame,
    race_col: str,
    gender_col: str,
) -> pd.DataFrame:
    rows = []
    for (race, gender), g in audit_df.groupby([race_col, gender_col]):
        m = subgroup_metrics(g["y_true"], g["y_pred"].values)
        m["Race"] = race
        m["Gender"] = gender
        rows.append(m)

    subgroup_df = pd.DataFrame(rows)[
        ["Race", "Gender", "n", "base_rate", "positive_pred_rate", "TPR", "FPR", "FNR", "TNR"]
    ].sort_values(["Race", "Gender"])

    subgroup_df.to_csv(OUTPUT_DIR / "step1_subgroup_audit.csv", index=False)
    return subgroup_df


def main() -> None:
    ensure_output_dir()

    print("Loading data...")
    df = load_data(DATA_PATH)

    cols = get_column_names(df)

    print("\nDetected columns:")
    for k, v in cols.items():
        print(f"{k}: {v}")

    # Normalize target early for clean EDA
    df[cols["target"]] = normalize_binary_target(df[cols["target"]])

    basic_eda(df, cols)

    X, y, dropped_cols = build_feature_matrix(df, cols)

    print("\nDropped columns for Step 1:")
    print(dropped_cols)

    print("\nFinal feature matrix shape:", X.shape)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    preprocessor = make_preprocessor(X)

    models = {
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

    trained = {}
    overall_results = []

    print("\nTraining models...")
    for name, model in models.items():
        pipe, prob, pred, results = evaluate_model(
            name=name,
            model=model,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        trained[name] = {"pipe": pipe, "prob": prob, "pred": pred}
        overall_results.append(results)
        print(f"\n{name}")
        for metric_name, value in results.items():
            if metric_name != "model":
                print(f"  {metric_name}: {value:.4f}")

    overall_df = pd.DataFrame(overall_results).sort_values("auroc", ascending=False)
    overall_df.to_csv(OUTPUT_DIR / "step1_overall_baseline_results.csv", index=False)

    best_model_name = overall_df.iloc[0]["model"]
    print(f"\nBest baseline model: {best_model_name}")

    audit_df = df.loc[idx_test, [cols["race"], cols["gender"], cols["age"]]].copy()
    audit_df["y_true"] = y_test.values
    audit_df["y_pred"] = trained[best_model_name]["pred"]
    audit_df["y_prob"] = trained[best_model_name]["prob"]

    subgroup_df = save_subgroup_audit(
        audit_df=audit_df,
        race_col=cols["race"],
        gender_col=cols["gender"],
    )

    print("\n===== OVERALL RESULTS =====")
    print(overall_df)

    print("\n===== SUBGROUP AUDIT =====")
    print(subgroup_df)

    print("\nSaved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()