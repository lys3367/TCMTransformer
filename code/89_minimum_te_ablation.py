from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib

ROOT = Path(__file__).resolve().parents[1]
common = importlib.import_module("00_common"); exporter = importlib.import_module("87_export_loso_target_predictions")

COMBOS = {1: [115], 2: [75, 115], 3: [75, 95, 115], 4: [75, 95, 105, 115], 5: [75, 85, 95, 105, 115]}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default=None); p.add_argument("--device", default="cuda"); p.add_argument("--batch-size", type=int, default=1); p.add_argument("--max-epochs", type=int, default=100); p.add_argument("--patience", type=int, default=12); args = p.parse_args()
    cfg = common.load_config(args.config); matrix = common.load_matrix(cfg); raw_all = matrix["x_raw"].astype(np.float32); subjects_all = matrix["subjects"].astype(str)
    cohort = exporter.canonical_indices(ROOT, subjects_all, "outputs/80_extra_loso_mole_full.csv"); raw = raw_all[cohort]; subjects = subjects_all[cohort]; te = matrix["te_values"].astype(int); repos = common.resolve_path(cfg, "external_repos_dir")
    meta_path = common.resolve_path(cfg, "metadata_file"); meta = pd.read_csv(meta_path) if meta_path.exists() else exporter.metadata_from_matrix(matrix)
    seed = int(cfg["random_seed"]); out = ROOT / "outputs/minimum_te_ablation"; out.mkdir(parents=True, exist_ok=True); csv = out / "89_minimum_te_ablation_folds.csv"
    existing = pd.read_csv(csv) if csv.exists() else pd.DataFrame(columns=["n_input_te", "input_te", "fold", "subject", "MAE"]); done = {(int(r.n_input_te), int(r.fold)) for r in existing.itertuples()}
    for n, input_te in COMBOS.items():
        input_idx = common.te_indices(te, input_te); target_idx = common.te_indices(te, cfg["fixed_target_te"]); folds = exporter.make_loso_folds(49, seed)
        for fold_no, test_idx in enumerate(folds, 1):
            if (n, fold_no) in done: continue
            outer = np.concatenate([f for i, f in enumerate(folds) if i != fold_no - 1]); rng = np.random.default_rng(seed + fold_no); shuffled = rng.permutation(outer); val_n = max(5, int(round(.2 * len(shuffled)))); val_idx, train_idx = shuffled[:val_n], shuffled[val_n:]
            true, pred, _ = exporter.train_fold("itransformer", raw, meta, train_idx, val_idx, test_idx, input_idx, target_idx, repos, seed + fold_no, args.device, args.batch_size, args.max_epochs, args.patience)
            row = pd.DataFrame([{"n_input_te": n, "input_te": "+".join(map(str, input_te)), "fold": fold_no, "subject": subjects[test_idx[0]], "MAE": float(np.abs(pred - true).mean())}]); existing = pd.concat([existing, row], ignore_index=True); existing.to_csv(csv, index=False)
            print(f"n_input_te={n} fold={fold_no}/49 MAE={row.MAE.iloc[0]:.6f}", flush=True)
    summary = existing.groupby(["n_input_te", "input_te"]).MAE.agg(["mean", "std", "count"]).reset_index(); summary.to_csv(out / "89_minimum_te_ablation_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5)); x = summary.n_input_te.to_numpy(); y = summary["mean"].to_numpy(); sd = summary["std"].to_numpy(); ax.plot(x, y, "o-", color="#EF5F5F", lw=2.5); ax.fill_between(x, y - 1.96 * sd / np.sqrt(summary["count"]), y + 1.96 * sd / np.sqrt(summary["count"]), color="#EF5F5F", alpha=.18); ax.set(xlabel="Number of input TEs", ylabel="LOSO MAE", title="Minimum-TE acquisition ablation"); ax.set_xticks(x); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(out / "Figure6_Minimum_TE_Ablation.png", dpi=600, bbox_inches="tight"); fig.savefig(out / "Figure6_Minimum_TE_Ablation.svg", bbox_inches="tight"); plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__": main()
