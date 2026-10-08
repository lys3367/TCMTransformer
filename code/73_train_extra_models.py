from __future__ import annotations

import argparse
import copy
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


common = importlib.import_module("00_common")


EXTRA_MODELS = ["segrnn", "sparsetsf", "mtsmixer", "mole"]


@dataclass
class TrainSettings:
    max_epochs: int = 100
    patience: int = 12
    batch_size: int = 16
    learning_rate: float = 7e-4
    weight_decay: float = 1e-4
    seed: int = 20260623
    device: str = "cuda"


class ForecastWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.model(x, None, None, None)
        except TypeError:
            return self.model(x)


class MarkForecastWrapper(nn.Module):
    def __init__(self, model: nn.Module, seq_len: int, n_time_features: int = 4):
        super().__init__()
        self.model = model
        base = torch.zeros(seq_len, n_time_features, dtype=torch.float32)
        base[:, 0] = torch.arange(seq_len, dtype=torch.float32)
        if n_time_features > 1 and seq_len > 1:
            base[:, 1] = torch.linspace(0.0, 1.0, seq_len)
        self.register_buffer("x_mark", base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        marks = self.x_mark.unsqueeze(0).expand(x.shape[0], -1, -1)
        return self.model(x, marks)


def clear_extra_imports(repo_paths: list[Path]) -> None:
    prefixes = ("models", "model", "layers")
    for name in list(sys.modules):
        if name in prefixes or name.startswith(tuple(prefix + "." for prefix in prefixes)):
            del sys.modules[name]
    repo_strings = {str(path) for path in repo_paths}
    sys.path[:] = [path for path in sys.path if path not in repo_strings]


def import_first(candidates: list[tuple[str, str]]):
    errors = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{module_name}.{class_name}: {exc}")
    raise ImportError("Could not import an official model class. Tried:\n" + "\n".join(errors))


def import_model_from_repo_files(repo: Path, preferred_patterns: list[str]):
    import importlib.util

    candidates: list[Path] = []
    for pattern in preferred_patterns:
        candidates.extend(sorted(repo.rglob(pattern)))
    candidates.extend(sorted(repo.rglob("*.py")))
    seen = set()
    errors = []
    for path in candidates:
        if path in seen or path.name.startswith("__"):
            continue
        seen.add(path)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        if "class Model" not in text:
            continue
        module_name = f"_mte_extra_{path.stem}_{abs(hash(str(path))) & 0xfffffff}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                errors.append(f"{path}: could not create import spec")
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return getattr(module, "Model")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")
    raise ImportError(
        "Could not discover a class named Model inside the repository. Tried files:\n"
        + "\n".join(errors[:30])
    )


def build_extra_model(
    model_name: str,
    seq_len: int,
    pred_len: int,
    n_variables: int,
    repos_dir: Path,
) -> nn.Module:
    all_repos = [
        repos_dir / "SegRNN",
        repos_dir / "SparseTSF",
        repos_dir / "MTS-Mixers",
        repos_dir / "MoLE",
    ]
    clear_extra_imports(all_repos)

    class Config:
        pass

    config = Config()
    config.seq_len = seq_len
    config.pred_len = pred_len
    config.enc_in = n_variables
    config.c_out = n_variables
    config.d_model = 16
    config.d_ff = 32
    config.e_layers = 1
    config.n_heads = 2
    config.dropout = 0.10
    config.head_dropout = 0.05
    config.individual = False
    config.revin = True
    config.embed = "fixed"
    config.freq = "h"

    if model_name == "segrnn":
        repo = repos_dir / "SegRNN"
        if not repo.exists():
            raise FileNotFoundError("Missing external_repos/SegRNN.")
        sys.path.insert(0, str(repo))
        config.rnn_type = "gru"
        config.dec_way = "pmf"
        config.seg_len = 1
        config.channel_id = 1
        Model = import_first(
            [
                ("models.SegRNN", "Model"),
                ("model.SegRNN", "Model"),
                ("SegRNN", "Model"),
            ]
        )
        return ForecastWrapper(Model(config))

    if model_name == "sparsetsf":
        repo = repos_dir / "SparseTSF"
        if not repo.exists():
            raise FileNotFoundError("Missing external_repos/SparseTSF.")
        sys.path.insert(0, str(repo))
        config.model_type = "linear"
        config.period_len = 1
        Model = import_first(
            [
                ("models.SparseTSF", "Model"),
                ("models.SparseTSF", "SparseTSF"),
                ("model.SparseTSF", "Model"),
                ("SparseTSF", "Model"),
            ]
        )
        return ForecastWrapper(Model(config))

    if model_name == "mtsmixer":
        repo = repos_dir / "MTS-Mixers"
        if not repo.exists():
            raise FileNotFoundError("Missing external_repos/MTS-Mixers.")
        sys.path.insert(0, str(repo))
        config.fac_T = True
        config.fac_C = True
        config.sampling = 1
        config.norm = "batch"
        config.rev = True
        Model = import_first(
            [
                ("models.MTSMixer", "Model"),
                ("models.MTS_Mixer", "Model"),
                ("models.MTSMixers", "Model"),
                ("model.MTSMixer", "Model"),
            ]
        )
        return ForecastWrapper(Model(config))

    if model_name == "mole":
        repo = repos_dir / "MoLE"
        if not repo.exists():
            raise FileNotFoundError("Missing external_repos/MoLE.")
        sys.path.insert(0, str(repo))
        config.num_experts = 4
        config.top_k = 2
        config.t_dim = pred_len
        try:
            Model = import_first(
                [
                    ("models.MoLE", "Model"),
                    ("models.MOLE", "Model"),
                    ("models.mole", "Model"),
                    ("model.MoLE", "Model"),
                    ("MoLE", "Model"),
                    ("mole", "Model"),
                ]
            )
        except ImportError:
            Model = import_model_from_repo_files(
                repo,
                ["*MoLE*.py", "*MOLE*.py", "*mole*.py", "models/*.py", "model/*.py"],
            )
        return MarkForecastWrapper(Model(config), seq_len, n_time_features=4)

    raise ValueError(f"Unknown extra model: {model_name}")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_one(
    model_name: str,
    raw: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    input_idx: np.ndarray,
    target_idx: np.ndarray,
    repos_dir: Path,
    settings: TrainSettings,
) -> tuple[np.ndarray, np.ndarray, dict]:
    common.set_seed(settings.seed)
    x, _, _ = common.standardize_from_train(raw, train_idx)
    train_x = x[train_idx][:, input_idx, :]
    train_y = x[train_idx][:, target_idx, :]
    val_x = x[val_idx][:, input_idx, :]
    val_y = x[val_idx][:, target_idx, :]
    test_x = x[test_idx][:, input_idx, :]
    true = x[test_idx][:, target_idx, :]

    device = torch.device(settings.device)
    model = build_extra_model(
        model_name,
        len(input_idx),
        len(target_idx),
        raw.shape[2],
        repos_dir,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    loss_fn = nn.MSELoss()
    loader = make_loader(
        train_x.astype(np.float32),
        train_y.astype(np.float32),
        min(settings.batch_size, len(train_idx)),
        True,
    )

    best_state = None
    best_val = np.inf
    best_epoch = 0
    wait = 0
    start = time.time()
    epochs_run = 0
    for epoch in range(1, settings.max_epochs + 1):
        epochs_run = epoch
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(val_x.astype(np.float32)).to(device))
            val_loss = float(
                loss_fn(val_pred, torch.from_numpy(val_y.astype(np.float32)).to(device)).item()
            )
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= settings.patience:
                break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced.")
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_x.astype(np.float32))),
            batch_size=min(settings.batch_size, len(test_idx)),
            shuffle=False,
        )
        for (batch_x,) in test_loader:
            preds.append(model(batch_x.to(device)).cpu().numpy())
    prediction = np.concatenate(preds, axis=0).astype(np.float32)
    details = {
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "val_RMSE": float(np.sqrt(best_val)),
        "training_seconds": time.time() - start,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    return true, prediction, details


def make_folds(n_subjects: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        np.asarray(indices, dtype=int)
        for indices in np.array_split(rng.permutation(n_subjects), n_folds)
    ]


def keep_subject_indices(subjects: np.ndarray, exclude_prefixes: list[str]) -> np.ndarray:
    keep = np.ones(len(subjects), dtype=bool)
    for prefix in exclude_prefixes:
        prefix = prefix.strip()
        if prefix:
            keep &= ~np.char.startswith(subjects.astype(str), prefix)
    return np.where(keep)[0]


def fixed_indices(config: dict, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_labels = common.load_split_labels(config, subjects)
    return (
        np.where(split_labels == "train")[0],
        np.where(split_labels == "val")[0],
        np.where(split_labels == "test")[0],
    )


def run_fixed(args, config, raw, subjects, te_values, repos_dir, output_dir) -> None:
    train_idx, val_idx, test_idx = fixed_indices(config, subjects)
    input_idx = common.te_indices(te_values, config["fixed_input_te"])
    target_idx = common.te_indices(te_values, config["fixed_target_te"])
    settings = TrainSettings(
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed or int(config["random_seed"]),
        device=args.device,
    )
    true, prediction, details = train_one(
        args.model, raw, train_idx, val_idx, test_idx, input_idx, target_idx, repos_dir, settings
    )
    result = common.regression_metrics(true, prediction)
    stem = f"73_extra_fixed_{args.model}_full"
    pd.DataFrame(
        [
            {
                "model": args.model,
                "grouping": "full",
                "task": "fixed_target",
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                **details,
                **result,
            }
        ]
    ).to_csv(output_dir / f"{stem}_summary.csv", index=False, encoding="utf-8-sig")
    print(f"{args.model}/fixed: MAE={result['MAE']:.6f}, RMSE={result['RMSE']:.6f}, R2={result['R2']:.6f}")


def run_resampling(args, config, raw, te_values, repos_dir, output_dir) -> None:
    base_seed = int(config["random_seed"])
    n_subjects = len(raw)
    folds = n_subjects if args.loso else args.folds
    repeats = 1 if args.loso else args.repeats
    input_idx = common.te_indices(te_values, config["fixed_input_te"])
    target_idx = common.te_indices(te_values, config["fixed_target_te"])
    rows = []
    for repeat in range(repeats):
        fold_sets = make_folds(n_subjects, folds, base_seed + repeat)
        for fold_number, test_idx in enumerate(fold_sets, start=1):
            outer_train = np.concatenate(
                [fold for index, fold in enumerate(fold_sets) if index != fold_number - 1]
            )
            rng = np.random.default_rng(base_seed + repeat * 100 + fold_number)
            shuffled = rng.permutation(outer_train)
            n_val = max(5, int(round(0.2 * len(shuffled))))
            val_idx = shuffled[:n_val]
            train_idx = shuffled[n_val:]
            settings = TrainSettings(
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                seed=base_seed + repeat * 100 + fold_number,
                device=args.device,
            )
            true, prediction, details = train_one(
                args.model,
                raw,
                train_idx,
                val_idx,
                test_idx,
                input_idx,
                target_idx,
                repos_dir,
                settings,
            )
            result = common.regression_metrics(true, prediction)
            rows.append(
                {
                    "strategy": "loso" if args.loso else "repeated_cv",
                    "repeat": repeat + 1,
                    "fold": fold_number,
                    "model": args.model,
                    "grouping": "full",
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "n_test": len(test_idx),
                    **details,
                    **result,
                }
            )
            print(
                f"{args.model} repeat={repeat+1} fold={fold_number} "
                f"RMSE={result['RMSE']:.6f}",
                flush=True,
            )
    table = pd.DataFrame(rows)
    prefix = "73_extra_loso" if args.loso else "73_extra_cv"
    stem = f"{prefix}_{args.model}_full"
    table.to_csv(output_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    means = table[["MAE", "RMSE", "R2"]].mean()
    stds = table[["MAE", "RMSE", "R2"]].std(ddof=1)
    report = [
        f"# 10 extra model {'LOSO' if args.loso else 'CV'}: {args.model}/full",
        "",
        f"- folds: {folds}",
        f"- repeats: {repeats}",
        f"- MAE: {means['MAE']:.6f} ± {stds['MAE']:.6f}",
        f"- RMSE: {means['RMSE']:.6f} ± {stds['RMSE']:.6f}",
        f"- R2: {means['R2']:.6f} ± {stds['R2']:.6f}",
    ]
    (output_dir / f"{stem}_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train extra official models separately from the main DLinear/PatchTST/TimesNet/iTransformer script."
    )
    parser.add_argument("--model", required=True, choices=EXTRA_MODELS)
    parser.add_argument("--mode", default="fixed", choices=["fixed", "cv", "loso"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--exclude-prefixes",
        nargs="*",
        default=["43_", "48_"],
        help="Subject prefixes excluded before fixed split or LOSO. Use no values to disable.",
    )
    args = parser.parse_args()
    args.loso = args.mode == "loso"

    config = common.load_config(args.config)
    matrix = common.load_matrix(config)
    raw = matrix["x_raw"].astype(np.float32)
    subjects = matrix["subjects"].astype(str)
    te_values = matrix["te_values"].astype(int)
    keep_idx = keep_subject_indices(subjects, args.exclude_prefixes)
    raw = raw[keep_idx]
    subjects = subjects[keep_idx]
    repos_dir = common.resolve_path(config, "external_repos_dir")
    output_dir = common.resolve_path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "fixed":
        run_fixed(args, config, raw, subjects, te_values, repos_dir, output_dir)
    else:
        run_resampling(args, config, raw, te_values, repos_dir, output_dir)


if __name__ == "__main__":
    main()
