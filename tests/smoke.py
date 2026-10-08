"""Run against an installed wheel from a fresh, unrelated working directory."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.pop("PYTHONPATH", None)


def run_smoke():
    import numpy as np
    import pandas as pd
    import torch
    from teprediction import common, train
    from teprediction.__main__ import COMMANDS

    for n, expected in [(49, (38, 10, 1)), (42, (33, 8, 1))]:
        folds = train.make_folds(n, 20260623)
        held_out = []
        for i in range(1, n+1):
            a, b, c = train.split_fold(folds, i, 20260623)
            assert (len(a), len(b), len(c)) == expected
            assert not (set(a) & set(b) or set(a) & set(c) or set(b) & set(c))
            assert set(a) | set(b) | set(c) == set(range(n))
            held_out.extend(c.tolist())
        assert sorted(held_out) == list(range(n))

    raw = np.arange(8*7*4, dtype=np.float32).reshape(8, 7, 4)
    _, mu, sd = common.standardize_from_train(raw, np.array([0, 1]))
    raw[2:] += 1e6
    _, mu2, sd2 = common.standardize_from_train(raw, np.array([0, 1]))
    np.testing.assert_array_equal(mu, mu2)
    np.testing.assert_array_equal(sd, sd2)

    def command(*args):
        result = subprocess.run([sys.executable, "-m", "teprediction", *args],
                                capture_output=True, text=True, encoding="utf-8",
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if result.returncode:
            raise RuntimeError(" ".join(args)+"\n"+result.stdout+"\n"+result.stderr)
        return result.stdout

    previous = Path.cwd()
    results = {}
    with tempfile.TemporaryDirectory(prefix="teprediction-smoke-") as temporary:
        os.chdir(temporary)
        try:
            for name, module in COMMANDS.items():
                importlib.import_module("teprediction."+module)
                command(name, "--help")
            command("synthetic", "--output", "example")
            command("prepare", "--config", "example/config.json")
            with np.load("example/processed/model_matrix_complete7te.npz", allow_pickle=True) as matrix:
                assert matrix["x_raw"].shape == (8, 7, 4)
                assert np.isfinite(matrix["x_raw"]).all()
            # A malformed table must fail, rather than silently double-count observations.
            frame = pd.read_csv("example/roi_metrics.csv")
            pd.concat([frame, frame.iloc[:1]]).to_csv("example/bad.csv", index=False)
            config = json.loads(Path("example/config.json").read_text())
            config["raw_csv"] = "example/bad.csv"
            Path("example/bad_config.json").write_text(json.dumps(config))
            try:
                command("prepare", "--config", "example/bad_config.json")
            except RuntimeError as exc:
                assert "duplicate" in str(exc).lower()
            else:
                raise AssertionError("Duplicate measurements were accepted")
            for model in ["dlinear", "itransformer", "timesnet"]:
                command("train", "--model", model, "--config", "example/config.json",
                        "--manifest", "example/cohort.csv", "--expected-subjects", "8",
                        "--output-dir", "outputs/smoke", "--device", "cpu",
                        "--max-epochs", "1", "--patience", "1", "--batch-size", "1")
                with np.load(f"outputs/smoke/{model}_predictions.npz", allow_pickle=True) as z:
                    assert z["pred_z"].shape == (8, 2, 4)
                    assert np.isfinite(z["pred_z"]).all()
                    assert len(set(z["subjects"].tolist())) == 8
                    results[model] = {"folds": 8, "prediction_shape": list(z["pred_z"].shape),
                                      "MAE": float(np.abs(z["true_z"]-z["pred_z"]).mean())}
            command("summarize", "--input-dir", "outputs/smoke", "--models", "dlinear", "itransformer", "timesnet",
                    "--expected-subjects", "8", "--expected-variables", "4")
            command("evaluate", "--input-dir", "outputs/smoke", "--output-dir", "outputs/evaluation",
                    "--models", "dlinear", "itransformer", "timesnet", "--groups", "example/groups.csv")
            assert Path("outputs/evaluation/group_pairwise_fdr.csv").is_file()
        finally:
            os.chdir(previous)
    return {"status": "passed", "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "command_help_checks": len(COMMANDS),
            "folds_and_train_only_scaling": "passed", "duplicate_input_rejection": "passed",
            "data": "entirely synthetic", "models": results,
            "limitations": "CPU one-epoch smoke, not paper-performance reproduction; MoLE source not distributed/tested here."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = run_smoke()
    text = json.dumps(result, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text+"\n", encoding="utf-8")
    print(text)
