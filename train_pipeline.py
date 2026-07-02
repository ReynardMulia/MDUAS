from __future__ import annotations

from pathlib import Path
import os

from credit_project import CreditScoreModelTrainer

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data_A.csv"
ARTIFACT_DIR = ROOT_DIR / "artifacts"

os.environ.setdefault("FORCE_NO_MLFLOW", "1")

def main() -> None:
    trainer = CreditScoreModelTrainer(random_state=42)
    df = trainer.load_data(DATA_PATH)
    best_result, _, _, _, _ = trainer.run_experiments(df=df, artifact_dir=ARTIFACT_DIR)

    print("Best model:", best_result.model_name)
    for metric_name, metric_value in best_result.metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print(f"Artifacts saved to: {ARTIFACT_DIR}")

if __name__ == "__main__":
    main()
