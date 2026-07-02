from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from credit_project import CreditScoreInference, load_metadata

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "best_model.pkl"

class CreditScorePredictor:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.engine = CreditScoreInference(self.model_path)
        self.metadata = load_metadata(self.model_path.parent)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.engine.predict_single(input_data)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        predictions, probabilities = self.engine.predict_dataframe(df)
        result = df.copy()
        result["prediction"] = predictions
        if probabilities.size:
            class_names = list(self.engine.pipeline.classes_)
            for index, class_name in enumerate(class_names):
                result[f"prob_{class_name}"] = probabilities[:, index]
        return result

if __name__ == "__main__":
    predictor = CreditScorePredictor()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    print(json.dumps(predictor.metadata, indent=2))
