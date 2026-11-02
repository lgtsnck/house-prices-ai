"""Метрики и численная диагностика обучения регрессионных моделей."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Сопоставимые MAE, RMSE и R² в исходной шкале SalePrice (USD)."""
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def save_mlp_training_diagnostics(mlp_pipeline, output_dir: str | Path) -> dict:
    """Сохраняет фактическую историю обучения MLP для анализа Early Stopping."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reg = mlp_pipeline.named_steps["model"].regressor_
    loss = np.asarray(reg.loss_curve_, dtype=float)
    validation = np.asarray(reg.validation_scores_, dtype=float)
    if len(loss) != len(validation) or len(loss) == 0:
        raise ValueError("Не найдена полная история обучения MLP с Early Stopping")
    best_index = int(np.nanargmax(validation))
    frame = pd.DataFrame({
        "Epoch": np.arange(1, len(loss) + 1),
        "Training_Loss": loss,
        "Validation_R2": validation,
    })
    frame.to_csv(output_dir / "mlp_learning_history.csv", index=False)
    data = {
        "epochs_run": int(len(loss)),
        "best_epoch": best_index + 1,
        "first_loss": float(loss[0]),
        "best_epoch_loss": float(loss[best_index]),
        "last_loss": float(loss[-1]),
        "first_validation_r2": float(validation[0]),
        "best_validation_r2": float(validation[best_index]),
        "last_validation_r2": float(validation[-1]),
        "early_stopping": bool(reg.early_stopping),
        "best_weights_restored": bool(reg.early_stopping),
        "note": "После Early Stopping восстановлены лучшие веса; loss_ относится к последней эпохе.",
    }
    (output_dir / "mlp_training_summary.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return data
