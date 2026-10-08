from __future__ import annotations

import argparse
from importlib import import_module

import numpy as np
import pandas as pd


common = import_module("00_common")
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def fit_ridge(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    penalty = alpha * np.eye(x.shape[1], dtype=np.float64)
    lhs = x_centered.T @ x_centered + penalty
    rhs = x_centered.T @ y_centered
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return x_mean, y_mean, weights


def predict_ridge(
    x: np.ndarray, fitted: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    x_mean, y_mean, weights = fitted
    return (x - x_mean) @ weights + y_mean


def fit_predict_per_variable(
    x: np.ndarray,
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    input_idx: np.ndarray,
    target_idx: np.ndarray,
    alpha: float,
) -> np.ndarray:
    prediction = np.empty(
        (len(pred_idx), len(target_idx), x.shape[2]), dtype=np.float32
    )
    for variable_index in range(x.shape[2]):
        fitted = fit_ridge(
            x[train_idx][:, input_idx, variable_index],
            x[train_idx][:, target_idx, variable_index],
            alpha,
        )
        prediction[:, :, variable_index] = predict_ridge(
            x[pred_idx][:, input_idx, variable_index], fitted
        )
    return prediction


def select_alpha(
    x: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    input_idx: np.ndarray,
    target_idx: np.ndarray,
) -> tuple[float, float]:
    best_alpha = ALPHAS[0]
    best_rmse = np.inf
    for alpha in ALPHAS:
        prediction = fit_predict_per_variable(
            x, train_idx, val_idx, input_idx, target_idx, alpha
        )
        rmse = common.regression_metrics(
            x[val_idx][:, target_idx, :], prediction
        )["RMSE"]
        if rmse < best_rmse:
            best_alpha = alpha
            best_rmse = rmse
    return float(best_alpha), float(best_rmse)


def run_ridge(
    raw: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    input_idx: np.ndarray,
    target_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    x, train_mean, train_std = common.standardize_from_train(raw, train_idx)
    alpha, val_rmse = select_alpha(
        x, train_idx, val_idx, input_idx, target_idx
    )
    prediction = fit_predict_per_variable(
        x, train_idx, test_idx, input_idx, target_idx, alpha
    )
    true = x[test_idx][:, target_idx, :]
    return true, prediction, {
        "alpha": alpha,
        "val_RMSE": val_rmse,
        "train_mean": train_mean,
        "train_std": train_std,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the supervised per-variable Ridge baseline.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = common.load_config(args.config)
    matrix = common.load_matrix(config)
    raw = matrix["x_raw"].astype(np.float32)
    subjects = matrix["subjects"].astype(str)
    te_values = matrix["te_values"].astype(int)
    split_labels = common.load_split_labels(config, subjects)
    metadata = pd.read_csv(common.resolve_path(config, "metadata_file"))
    output_dir = common.resolve_path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_idx = np.where(split_labels == "train")[0]
    val_idx = np.where(split_labels == "val")[0]
    test_idx = np.where(split_labels == "test")[0]
    input_idx = common.te_indices(te_values, config["fixed_input_te"])
    target_idx = common.te_indices(te_values, config["fixed_target_te"])

    true, prediction, details = run_ridge(
        raw, train_idx, val_idx, test_idx, input_idx, target_idx
    )
    result = common.regression_metrics(true, prediction)
    summary = pd.DataFrame(
        [
            {
                "model": "per_variable_ridge",
                "grouping": "per_variable",
                "task": "fixed_target",
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "alpha": details["alpha"],
                "val_RMSE": details["val_RMSE"],
                **result,
            }
        ]
    )
    summary.to_csv(
        output_dir / "02_ridge_summary.csv", index=False, encoding="utf-8-sig"
    )
    common.summarize_by_roi(true, prediction, metadata).to_csv(
        output_dir / "02_ridge_by_roi.csv", index=False, encoding="utf-8-sig"
    )
    report = [
        "# 02 监督式 Ridge 固定目标 TE 结果",
        "",
        f"- 输入 TE：{config['fixed_input_te']}",
        f"- 目标 TE：{config['fixed_target_te']}",
        f"- 训练/验证/测试受试者：{len(train_idx)}/{len(val_idx)}/{len(test_idx)}",
        f"- 验证集选择 alpha：{details['alpha']:g}",
        f"- 测试 MAE：{result['MAE']:.6f}",
        f"- 测试 RMSE：{result['RMSE']:.6f}",
        f"- 测试 R2：{result['R2']:.6f}",
        "",
        "该模型对每个 Model+Metric+ROI 变量分别学习训练集 TE 映射，是后续所有神经模型的主要比较基线。",
    ]
    (output_dir / "02_ridge_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
