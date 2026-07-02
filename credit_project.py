from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Credit_Score"
DROP_COLUMNS = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"]
NUMERIC_FEATURES = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age_Months",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Type_of_Loan_Count",
    "Type_of_Loan_Has_Not_Specified",
]
CATEGORICAL_FEATURES = [
    "Month",
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]


class CreditFeatureCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if isinstance(X, pd.DataFrame):
            self.input_columns_ = list(X.columns)
        else:
            self.input_columns_ = None
        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            if self.input_columns_ is None:
                raise ValueError()
            X = pd.DataFrame(X, columns=self.input_columns_)
        df = X.copy()

        for column in DROP_COLUMNS:
            if column in df.columns:
                df = df.drop(columns=column)

        df = self._standardize_missing_values(df)
        df = self._clean_numeric_columns(df)
        df["Credit_History_Age_Months"] = df.get("Credit_History_Age", pd.Series(index=df.index)).apply(
            self._parse_credit_history_age
        )
        df["Type_of_Loan_Count"] = df.get("Type_of_Loan", pd.Series(index=df.index)).apply(self._count_loans)
        df["Type_of_Loan_Has_Not_Specified"] = df.get("Type_of_Loan", pd.Series(index=df.index)).apply(
            self._has_not_specified_loan
        )

        if "Month" in df.columns:
            df["Month"] = df["Month"].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})
        for column in CATEGORICAL_FEATURES:
            if column in df.columns:
                df[column] = df[column].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})

        if "Credit_History_Age" in df.columns:
            df = df.drop(columns=["Credit_History_Age"])
        if "Type_of_Loan" in df.columns:
            df = df.drop(columns=["Type_of_Loan"])

        return df

    @staticmethod
    def _standardize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        missing_tokens = {"", "_", "__", "___", "____", "_____", "_______", "nan", "None", "null"}
        cleaned = df.copy()
        for column in cleaned.columns:
            cleaned[column] = cleaned[column].replace(list(missing_tokens), np.nan)
        return cleaned

    @staticmethod
    def _clean_numeric_value(value: Any) -> float:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.number)):
            return float(value)
        text = str(value).strip()
        if text in {"", "nan", "None", "null", "_"}:
            return np.nan
        text = text.replace(",", "")
        text = re.sub(r"[^0-9.\-]", "", text)
        if text in {"", "-", ".", "-."}:
            return np.nan
        try:
            return float(text)
        except ValueError:
            return np.nan

    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        numeric_like_columns = [
            "Age",
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Interest_Rate",
            "Num_of_Loan",
            "Delay_from_due_date",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance",
        ]
        for column in numeric_like_columns:
            if column in cleaned.columns:
                cleaned[column] = cleaned[column].apply(self._clean_numeric_value)
        if "Age" in cleaned.columns:
            cleaned["Age"] = cleaned["Age"].clip(lower=0, upper=100)
        return cleaned

    @staticmethod
    def _parse_credit_history_age(value: Any) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip().lower()
        match = re.search(r"(?:(\d+)\s*years?)?\s*(?:and\s*)?(?:(\d+)\s*months?)?", text)
        if not match:
            return np.nan
        years = int(match.group(1)) if match.group(1) else 0
        months = int(match.group(2)) if match.group(2) else 0
        return float(years * 12 + months)

    @staticmethod
    def _count_loans(value: Any) -> float:
        if pd.isna(value):
            return 0.0
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return 0.0
        parts = [part.strip() for part in re.split(r",| and ", text) if part.strip()]
        filtered = [part for part in parts if part.lower() != "not specified"]
        return float(len(filtered))

    @staticmethod
    def _has_not_specified_loan(value: Any) -> float:
        if pd.isna(value):
            return 0.0
        text = str(value).lower()
        return 1.0 if "not specified" in text else 0.0


@dataclass
class ExperimentResult:
    model_name: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    report: Dict[str, Any]


class CreditScoreModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.cleaner = CreditFeatureCleaner()
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "onehot",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.model_candidates = {
            "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state),
            "random_forest": RandomForestClassifier(
                n_estimators=120,
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=160,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
        }

    def load_data(self, csv_path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
            df = df.drop(columns=df.columns[0])
        return df

    def split_data(self, df: pd.DataFrame):
        X = df.drop(columns=[TARGET_COLUMN]).copy()
        y = df[TARGET_COLUMN].copy()
        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y,
        )

    def build_pipeline(self, estimator) -> Pipeline:
        return Pipeline(
            steps=[
                ("cleaner", self.cleaner),
                ("preprocessor", self.preprocessor),
                ("model", estimator),
            ]
        )

    @staticmethod
    def evaluate_model(y_true, y_pred) -> Dict[str, float]:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    def run_experiments(
        self,
        df: pd.DataFrame,
        artifact_dir: str | Path,
        experiment_name: str = "credit_score_local_pipeline",
    ) -> Tuple[ExperimentResult, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)
        tracking_db = artifact_path / "mlflow.db"
        mlflow_enabled = os.environ.get("FORCE_NO_MLFLOW", "").lower() not in {"1", "true", "yes"}
        try:
            if mlflow_enabled:
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
                mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
                mlflow.set_experiment(experiment_name)
        except Exception as exc:
            mlflow_enabled = False
            (artifact_path / "mlflow_status.json").write_text(
                json.dumps({"enabled": False, "reason": str(exc)}, indent=2),
                encoding="utf-8",
            )

        X_train, X_test, y_train, y_test = self.split_data(df)

        results: List[Dict[str, Any]] = []
        best_result: Optional[ExperimentResult] = None
        best_score = -np.inf
        best_confusion = None

        for model_name, estimator in self.model_candidates.items():
            pipeline = self.build_pipeline(estimator)
            if mlflow_enabled:
                with mlflow.start_run(run_name=model_name):
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    metrics = self.evaluate_model(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("random_state", self.random_state)
                    for key, value in metrics.items():
                        mlflow.log_metric(key, float(value))
                    mlflow.log_text(classification_report(y_test, y_pred, zero_division=0), "classification_report.txt")
                    mlflow.sklearn.log_model(pipeline, artifact_path="model")
            else:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                metrics = self.evaluate_model(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

                results.append({"model_name": model_name, **metrics})
                weighted_f1 = metrics["f1_weighted"]
                if weighted_f1 > best_score:
                    best_score = weighted_f1
                    best_result = ExperimentResult(model_name=model_name, pipeline=pipeline, metrics=metrics, report=report)
                    best_confusion = confusion_matrix(y_test, y_pred, labels=["Good", "Poor", "Standard"])

        results_df = pd.DataFrame(results).sort_values(by="f1_weighted", ascending=False).reset_index(drop=True)
        results_df.to_csv(artifact_path / "model_comparison.csv", index=False)

        assert best_result is not None
        joblib.dump(best_result.pipeline, artifact_path / "best_model.pkl")
        self._save_metadata(artifact_path, best_result.model_name, best_result.metrics)
        if best_confusion is not None:
            self._save_confusion_matrix(best_confusion, artifact_path / "confusion_matrix.png")

        sample_cases = self._build_sample_cases(df)
        sample_cases.to_csv(artifact_path / "sample_cases.csv", index=False)

        return best_result, X_train, X_test, y_train, y_test

    @staticmethod
    def _save_metadata(artifact_dir: Path, best_model_name: str, metrics: Dict[str, float]) -> None:
        payload = {
            "best_model_name": best_model_name,
            "metrics": metrics,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        }
        (artifact_dir / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _save_confusion_matrix(conf_matrix: np.ndarray, output_path: Path) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        labels = ["Good", "Poor", "Standard"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _build_sample_cases(df: pd.DataFrame) -> pd.DataFrame:
        samples = []
        for label in ["Good", "Poor", "Standard"]:
            subset = df[df[TARGET_COLUMN] == label]
            if subset.empty:
                continue
            sample = subset.iloc[[0]].copy()
            sample.insert(0, "Sample_Label", label)
            samples.append(sample)
        if not samples:
            return pd.DataFrame()
        return pd.concat(samples, ignore_index=True)


class CreditScoreInference:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.pipeline = joblib.load(self.model_path)

    def predict_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        predictions = self.pipeline.predict(df)
        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba(df)
        else:
            probabilities = np.empty((len(df), 0))
        return predictions, probabilities

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        frame = pd.DataFrame([input_data])
        prediction, probabilities = self.predict_dataframe(frame)
        result: Dict[str, Any] = {"prediction": prediction[0]}
        if probabilities.size:
            classes = list(self.pipeline.classes_)
            result["probabilities"] = {cls: float(prob) for cls, prob in zip(classes, probabilities[0])}
        return result


def load_metadata(model_dir: str | Path) -> Dict[str, Any]:
    metadata_path = Path(model_dir) / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def prepare_input_dataframe(data: Dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame([data])
