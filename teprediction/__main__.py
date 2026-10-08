"""One discoverable entry point; each command exposes its own --help."""
import importlib
import sys

COMMANDS = {
    "prepare": "prepare", "train": "train", "summarize": "summarize",
    "synthetic": "synthetic", "evaluate": "evaluate",
    "linear-reference": "linear_reference", "leave-one-te": "leave_one_te",
    "leave-one-te-mole": "leave_one_te_mole", "input-te-count": "input_te_count",
    "te-pairs": "te_pairs", "export-predictions": "export_predictions",
    "interpolate": "interpolate", "external": "external",
    "prepare-incomplete": "prepare_incomplete", "pretrain-itransformer": "pretrain_itransformer",
    "pretrain-baselines": "pretrain_baselines", "pretrain-mole": "pretrain_mole",
    "matched-pretraining": "matched_pretraining", "native-units": "native_units",
    "metric-errors": "metric_errors", "wmti-sensitivity": "wmti_sensitivity",
    "summarize-wmti": "summarize_wmti", "kernel-sensitivity": "kernel_sensitivity",
    "summarize-kernel": "summarize_kernel",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python -m teprediction COMMAND [options]\nCommands:\n  " + "\n  ".join(COMMANDS))
        return
    command = sys.argv.pop(1)
    if command not in COMMANDS:
        raise SystemExit("Unknown command: " + command)
    importlib.import_module("teprediction." + COMMANDS[command]).main()


if __name__ == "__main__":
    main()
