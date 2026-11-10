"""Command line:  ml-pipe init | run | predict   (or python -m ml_pipe ...)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml_pipe.config import EXAMPLE_CONFIG, ConfigError, Experiment, read_table


def cmd_init(args: argparse.Namespace) -> int:
    """Write the example experiment config to `args.path`.

    Refuses to overwrite an existing file unless `args.force` is set. Returns the exit code."""
    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"{path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    path.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    print(f"wrote {path}: set [data] path / target / task, then: ml-pipe run {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the experiment config `args.config`, or with `args.check` only validate it and its data.

    Returns the exit code."""
    experiment = Experiment.from_file(args.config)
    if args.check:
        X, y = experiment.load_data()
        print(f"config OK: {len(X)} rows, {X.shape[1]} features, {len(experiment.models)} model(s)")
        return 0
    experiment.run(Path(args.output) if args.output else None)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Predict the rows of `args.data` with the saved model `args.model`.

    Writes a CSV to `args.output` (or stdout) with a prediction column, plus one
    probability column per class with `args.proba` and the input columns with
    `args.keep_columns`. Returns the exit code."""
    from ml_pipe.model import load
    model = load(args.model)
    data = read_table(Path(args.data))
    out = data.copy() if args.keep_columns else data.iloc[:, :0].copy()
    out["prediction"] = model.predict(data)
    if args.proba:
        proba = model.predict_proba(data)
        for column in proba.columns:
            out[f"proba_{column}"] = proba[column].to_numpy()
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"{len(out)} predictions -> {args.output}")
    else:
        out.to_csv(sys.stdout, index=False, lineterminator="\n")
    return 0


def parser() -> argparse.ArgumentParser:
    """The argument parser for the init, run and predict subcommands."""
    p = argparse.ArgumentParser(prog="ml-pipe", description="Train and compare many scikit-learn models.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="write an example experiment config")
    s.add_argument("path", nargs="?", default="experiment.toml")
    s.add_argument("--force", action="store_true", help="overwrite an existing file")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("run", help="run an experiment config (.toml or .json)")
    s.add_argument("config")
    s.add_argument("-o", "--output", help="output folder (default: [output] dir in the config)")
    s.add_argument("--check", action="store_true", help="validate the config and data, don't train")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("predict", help="predict with a saved model")
    s.add_argument("model", help=".joblib file written by run (only use files you trust)")
    s.add_argument("data", help="table with the model's feature columns (.csv, .parquet, .xlsx, .json)")
    s.add_argument("-o", "--output", help="CSV to write (default: print)")
    s.add_argument("--proba", action="store_true", help="add a probability column per class")
    s.add_argument("--keep-columns", action="store_true", help="copy the input columns into the output")
    s.set_defaults(func=cmd_predict)
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse `argv` (default: sys.argv) and run the chosen subcommand.

    Config, missing-column and missing-file errors are printed and give exit code 2."""
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, KeyError, FileNotFoundError) as e:
        print(f"error: {e.args[0] if isinstance(e, KeyError) else e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
