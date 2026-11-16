"""Command line:  ml-pipe init | run | predict | log   (or python -m ml_pipe ...)."""
from __future__ import annotations

import argparse
import json
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
    experiment.run(Path(args.output) if args.output else None, verbose=args.verbose)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Predict the rows of `args.data` with the saved model `args.model`.

    Writes a CSV to `args.output` (or stdout) with a prediction column, plus one
    probability column per class with `args.proba` and the input columns with
    `args.keep_columns`. Returns the exit code."""
    from ml_pipe.model import load
    from ml_pipe.tracking import ConsoleOutput, ExecutionTracker
    # stage lines and the summary go to stderr: stdout may be the predictions CSV
    console = ConsoleOutput(sys.stderr, verbose=args.verbose)
    with ExecutionTracker("Predict", module=__name__, operation="predict new rows with a saved model",
                          console=console, metadata={"model_file": Path(args.model).name,
                                                     "data_file": Path(args.data).name}) as tracker:
        with tracker.track("Load model", module="ml_pipe.model", operation="load the saved .joblib model"):
            model = load(args.model)
        with tracker.track("Read data", module=__name__, operation="read the table to predict") as step:
            data = read_table(Path(args.data))
            step.annotate(rows=len(data), columns=int(data.shape[1]))
        with tracker.track("Predict", module="ml_pipe.model", operation="predict every row"):
            out = data.copy() if args.keep_columns else data.iloc[:, :0].copy()
            out["prediction"] = model.predict(data)
            if args.proba:
                proba = model.predict_proba(data)
                for column in proba.columns:
                    out[f"proba_{column}"] = proba[column].to_numpy()
        with tracker.track("Write predictions", module=__name__, operation="write the predictions CSV"):
            if args.output:
                out.to_csv(args.output, index=False)
                print(f"{len(out)} predictions -> {args.output}")
            else:
                out.to_csv(sys.stdout, index=False, lineterminator="\n")
    return 0


def _level(text: str) -> int | tuple[int | None, int | None]:
    """'2' -> 2, '1:3' -> (1, 3), '2:' -> (2, None)."""
    if ":" not in text:
        return int(text)
    low, _, high = text.partition(":")
    return (int(low) if low else None, int(high) if high else None)


def cmd_log(args: argparse.Namespace) -> int:
    """Inspect an execution log: summary + stage tree, one stage expanded, or a filtered list of stages.

    Returns the exit code: with no filter, 1 when the execution did not pass (for scripts)."""
    from ml_pipe.tracking import ExecutionLog
    from ml_pipe.tracking.outputs import glyphs_for, write_safely as print  # never fails on a cp1252 console
    glyphs = glyphs_for(sys.stdout)
    log = ExecutionLog.load(args.path)
    if args.details:
        print(log._details(_stage(log, args.details)))
        return 0
    if args.expand:
        print(log._tree(_stage(log, args.expand), max_depth=args.depth, show_ids=args.ids, glyphs=glyphs))
        return 0
    filters = {"status": args.status, "execution_id": args.execution_id, "stage_name": args.stage,
               "module": args.module, "level": args.level, "since": args.since, "until": args.until,
               "error_type": args.error_type}
    if any(v is not None for v in filters.values()) or args.json:
        try:
            records = log.filter(**filters)
        except ValueError as e:
            sys.stderr.write(f"error: {e}\n")
            return 2
        if args.json:
            print(json.dumps([r.to_dict() for r in records], indent=2, default=str))
            return 0
        for r in records:
            seconds = "" if r.duration is None else f"{r.duration:9.3f}s"
            line = f"{r.status.value:<9} {seconds:>10}  {log.path(r, True)}"
            if args.ids:
                line += f"  [{r.stage_id}]"
            if r.status.value != "PASSED" and r.reason and (r.status.value != "FAILED" or r.is_root_cause):
                line += f"\n{'':<21}reason: {r.reason}"
            if r.is_root_cause and r.location:
                line += f"\n{'':<21}where : {r.location} (module {r.module or '-'})"
            print(line)
        print(f"{len(records)} of {len(log.records)} stages")
        return 0
    print(log._summary_text())
    print("")
    print(log._tree(max_depth=args.depth, show_ids=args.ids, glyphs=glyphs))
    return 0 if log.outcome.startswith("PASSED") else 1


def _stage(log, key: str):
    """The stage `key` (id or name) in `log`; KeyError with a hint if there is none."""
    record = log.find(key)
    if record is None:
        raise KeyError(f"no stage {key!r} in the log (use a stage id from --ids, or part of a stage name)")
    return record


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
    s.add_argument("-v", "--verbose", action="store_true",
                   help="developer view: print tracebacks and stage metadata (always in the run folder's log)")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("predict", help="predict with a saved model")
    s.add_argument("model", help=".joblib file written by run (only use files you trust)")
    s.add_argument("data", help="table with the model's feature columns (.csv, .parquet, .xlsx, .json)")
    s.add_argument("-o", "--output", help="CSV to write (default: print)")
    s.add_argument("--proba", action="store_true", help="add a probability column per class")
    s.add_argument("--keep-columns", action="store_true", help="copy the input columns into the output")
    s.add_argument("-v", "--verbose", action="store_true", help="print tracebacks and stage metadata")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("log", help="inspect an execution log: what passed, failed or was skipped, and why",
                       description="With no options: the execution summary and the stage tree. "
                                   "Any filter lists the matching stages instead.")
    s.add_argument("path", help="a run folder, or an execution_<id>.json / .jsonl file")
    s.add_argument("--status", action="append", help="PASSED / FAILED / SKIPPED / CANCELLED / RUNNING "
                                                     "(also pass, fail, skip); repeat for several")
    s.add_argument("--stage", help="stage name contains this text")
    s.add_argument("--module", help="module contains this text, e.g. ml_pipe.base")
    s.add_argument("--level", type=_level, help="hierarchy level: 2, or a range such as 1:3 or 2:")
    s.add_argument("--since", help="started at or after this ISO time, e.g. 2026-11-12T10:00")
    s.add_argument("--until", help="ended at or before this ISO time")
    s.add_argument("--error-type", help="exception class, e.g. ValueError")
    s.add_argument("--execution-id", help="execution id (or its start), for folders holding several runs")
    s.add_argument("--expand", metavar="STAGE", help="show this stage (id or name) and its sub-stages")
    s.add_argument("--details", metavar="STAGE", help="everything recorded about one stage, with its traceback")
    s.add_argument("--depth", type=int, help="levels of the tree to show")
    s.add_argument("--ids", action="store_true", help="show stage ids")
    s.add_argument("--json", action="store_true", help="print the matching records as JSON")
    s.set_defaults(func=cmd_log)
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
