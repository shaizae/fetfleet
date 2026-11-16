# Execution tracking

Every run of fitfleet records each stage it goes through: what ran, where, when it started and
ended, whether it **PASSED**, **FAILED**, was **SKIPPED** or **CANCELLED**, and for a failure,
why and where. You can see what happened without reading source code or raw stack traces.

The code is in `fitfleet/tracking/`. It has no dependencies beyond the standard library.

| Module | What it holds |
|---|---|
| `core.py` | `ExecutionTracker`, `StageRecord`, `Status`; the `stage` / `tracked` / `annotate` / `execution` helpers; plain-language error explanations; masking of sensitive values |
| `tasks.py` | `TrackedTask`: wraps process-pool tasks so worker results and failures carry the execution and stage ids |
| `outputs.py` | Console, developer log file, JSON event stream and summary-file outputs |
| `log.py` | `ExecutionLog`: filtering, tree view, summary, per-stage details, DataFrame export, PDF section; loads logs from disk |

---

## 1. What gets tracked

### `fitfleet run` / `Experiment.run()`

```
Experiment exp.json                         (level 0, the execution)
├── Prepare output folder
├── Load data
├── Configure run
│   └── Feature selection                   (when [feature_selection] uses a score function)
├── Train and validate
│   └── k_folds | train_test_split | leave_one_out
│       ├── Share data with workers
│       ├── Compute folds / Split rows
│       └── Search (grid search | random search | halving search | forward/backward feature selection)
│           └── Train batch: <round>        (one per pool batch: each halving round, each greedy step)
│               └── Task: <Model> #<id>     (the configuration that failed; every task with task_records=True)
├── Rank results and write CSV
├── Refit best models
├── Experiment PDF report                   (runs on the background report thread)
│   └── Plot: <title>                       (only plots that failed)
├── Save models
├── Finish PDF reports
└── PDF report: #<n> <Model>
```

All top-level stages are planned when the run starts. If one fails, the stages that never ran are
reported as **SKIPPED**, together with the stage that stopped them.

### Python API

`Classification` / `Regression`: `k_folds()`, `train_test_split()` and `leave_one_out()` are tracked
with the same sub-stages. `feature_selection()` and `brute_force_features()` are tracked too.
If no tracker is active, each run creates its own tracker with console output off. You read it
through the run's three log properties (see section 4):

```python
with Classification(log_dir="reports") as run:
    run.set(X, y, models)
    run.k_folds(5)
print(run.execution_log)      # the whole log: summary + stage tree
```

`FeatureResearch`: `single_feature_score`, `model_importance`, `permutation_importance`,
`random_probe`, `leakage_check`, `report`, `recommend` and `save_pdf_report` become stages when a
tracker is active. Without one they are plain method calls.

### `fitfleet predict`

The stages are Load model, Read data, Predict and Write predictions. Their console lines and the
summary go to **stderr**, so the predictions CSV on stdout stays clean.

---

## 2. The record

Each stage is one `StageRecord`. Its JSON form is what the log files contain.

| Field | Meaning |
|---|---|
| `execution_id` | Id of the whole execution (12 hex characters) |
| `stage_id` | `<execution_id>.<n>`: unique to this stage |
| `stage_name`, `operation` | What ran: a short name and a one-line description |
| `parent_stage_id`, `level` | Place in the hierarchy (the execution root is level 0) |
| `module` | Python module responsible for the stage |
| `status` | `PENDING` · `RUNNING` · `PASSED` · `FAILED` · `SKIPPED` · `CANCELLED` |
| `start_time`, `end_time` | ISO 8601 with milliseconds and UTC offset |
| `duration` | Seconds, from a monotonic clock |
| `error_type`, `error_message` | Exception class and message (credentials masked) |
| `traceback` | Full traceback, including the worker's traceback for pool tasks |
| `location` | `file.py:line in function`: where the exception was raised (in the worker, for pool tasks) |
| `reason` | Plain-language explanation for users: why it failed, was skipped or was cancelled |
| `origin` | For a failure, the stage where it started. A parent that only re-raised a child's exception points at the child |
| `metadata` | Extra facts: counts, shapes, settings, worker pids and times, the failing configuration |
| `pid`, `thread` | Process and thread that ran the stage |

**Root cause or propagated failure.** When "Correlation" fails inside "Feature Selection", both
stages are FAILED (the exception went through both). Only Correlation is a *root cause*: it has
the traceback and location. Feature Selection's reason reads
`Stopped because 'Correlation' failed: ...`. The summary lists root causes under FAILED STAGES and
names the stages they stopped.

**Status rules**

| What happened | Status |
|---|---|
| The block finished | PASSED |
| The block raised an exception | FAILED (the exception is re-raised unchanged) |
| `KeyboardInterrupt`, `SystemExit`, `GeneratorExit` | CANCELLED |
| A planned stage never started | SKIPPED, with the failure or cancellation that prevented it |
| `tracker.skip(name, reason)` or `stage.skip(reason)` | SKIPPED, with your reason |
| Still RUNNING when the execution ended (e.g. a report that timed out) | CANCELLED |

**Execution outcome.** The summary's `Result` line is one of the following:

* `PASSED`
* `FAILED`
* `CANCELLED`
* `PASSED WITH n CONTAINED FAILURE(S)`: some stages failed but the run still finished. An example
  is a failed PDF plot, which fitfleet isolates on purpose.
* `INCOMPLETE`: the log ends while stages are still running. The run was killed, or is still going.

---

## 3. Outputs

`fitfleet run` writes these files into the run folder (`runs/<date_time>/`):

| File | For | Written |
|---|---|---|
| console (stdout, through the experiment's `log`) | real-time monitoring: one line per stage start and end, then the summary | live |
| `execution_<id>.log` | developers: every transition with ids, pid, thread, module, timing, metadata and full tracebacks, then the summary and the tree | live, flushed per line |
| `execution_<id>.jsonl` | programs: one JSON event per line (`stage_started`, `stage_finished`, `execution_finished`) | live, flushed per line |
| `execution_<id>.json` | programs: the final records and the summary as one document | at the end |
| `execution_<id>_summary.txt` | everyone: the summary and the stage tree | at the end |
| experiment PDF and per-model PDFs, **Execution Summary** section | readers of the report: result, counts, failed stages with location and reason, skipped stages, the stage tree (no tracebacks) | when the PDF is rendered |

The experiment PDF is rendered in the background while the models are saved. Its Execution
Summary therefore covers the stages up to the point the report started; the section says so. The
complete summary is in `execution_<id>_summary.txt`.

The run folder is now created before the data is loaded. A run that fails while loading data
still leaves a folder with its execution logs.

---

## 4. Inspecting results: users

```console
$ fitfleet run experiment.toml                 # live stage lines + summary at the end
$ fitfleet run experiment.toml -v              # developer view: tracebacks and metadata on the console

$ fitfleet log runs/2026-11-12_10-00-00        # summary + stage tree (exit code 1 if it did not pass)
$ fitfleet log runs/... --status fail          # only failed stages, with reason and location
$ fitfleet log runs/... --status skip --status cancel
$ fitfleet log runs/... --stage "train batch"  # by name (substring, case-insensitive)
$ fitfleet log runs/... --module fitfleet.base  # by module
$ fitfleet log runs/... --level 1              # one level; or a range: --level 1:2, --level 2:
$ fitfleet log runs/... --since 2026-11-12T10:05 --until 2026-11-12T10:10
$ fitfleet log runs/... --error-type MemoryError
$ fitfleet log runs/ --execution-id 7b81       # a folder that holds several executions
$ fitfleet log runs/... --expand "Train and validate" --depth 2   # expand one stage
$ fitfleet log runs/... --details "Task: SVC #12"                 # everything about one stage
$ fitfleet log runs/... --status fail --json   # matching records as JSON
$ fitfleet log runs/... --ids                  # show stage ids (usable with --expand / --details)
```

A failed run's summary looks like this:

```text
EXECUTION SUMMARY
============================================================
Execution    : Experiment bad.json
Execution ID : e9bc5545d6c7
Result       : FAILED
Total stages : 9

PASSED:    1
FAILED:    1
SKIPPED:   7

Total execution time: 0.01 seconds

FAILED STAGES:
1. Load data
   Where   : Experiment bad.json > Load data
   Location: config.py:494 in read_table  (module fitfleet.config)
   Error   : ConfigError
   Reason  : The experiment configuration or its data is invalid: data file not found: ...\missing.csv
   Also stopped: Experiment bad.json

SKIPPED STAGES:
1. Configure run
   Reason: Not run because 'Load data' failed.
...
```

---

### The three log properties

A run (`Classification` / `Regression`) has exactly three ways to output its execution log. All
three contain the summary (result, counts, failed, cancelled and skipped stages with where and
why) and the stage tree. The folder is set once, in the constructor:

```python
run = Classification(log_dir="reports")       # default "logs"
...                                           # k_folds(), train_test_split(), ...
print(run.execution_log)                      # the log as text
run.save_pdf_log                              # reports/execution-log_<id>_<time>.pdf -> its path
run.save_json_log                             # reports/execution-log_<id>_<time>.json -> its path
```

They describe the last run; before any run they raise `RuntimeError`. The JSON holds
`{"summary": ..., "tree": [one line per stage], "stages": [every record]}`, and
`ExecutionLog.load()` reads it back. The files are named `execution-log_...` and not
`execution_...`, so `ExecutionLog.load(folder)` does not read them as a second execution.

These three are the only Python API for outputting a log. For a log on disk (a run folder of
`fitfleet run`, or the `output_dir` of your own `ExecutionTracker`), use `fitfleet log <folder>`.

**PDF reports add it automatically.** The log of the run they came from becomes the
**Execution Summary** section, and `save_pdf_report()` also writes `execution-log_<id>_<time>.pdf`
in the same folder:

* `run.experiment_report().save_pdf_report(folder)`
* `result.save_pdf_report(folder)` for the results `filter_by()` / `refit()` return
* `FeatureResearch.save_pdf_report(...)` when it runs inside a tracked execution (section only)

A saved (pickled) result does not keep its tracker.

## 5. Investigating failures: developers

```python
from fitfleet import ExecutionLog

log = ExecutionLog.load("runs/2026-11-12_10-00-00")    # folder, .json or .jsonl (even a crashed run)
for record in log.filter(status="FAILED", root_causes_only=True):
    print(record.location, record.reason)
log.filter(module="fitfleet.base", level=(2, None), since="2026-11-12T10:00")
log.to_frame()                        # pandas: one row per stage
```

If a pool task fails, its record (`Task: <Model> #<candidate id>`) holds the following:

* `pid`: the worker process
* `location`: the frame in the worker that raised
* `metadata.task`:
  * `execution_id` and `stage_id`: the batch the task belongs to
  * `model`, `params` and `n_features`: the exact configuration
  * `start` and `duration`
* the full traceback, including the worker's remote traceback

---

## 6. Tracking your own code

```python
from fitfleet import ExecutionTracker
from fitfleet.tracking import stage, tracked, annotate

with ExecutionTracker("Nightly retrain", output_dir="logs", verbose=False) as tracker:
    tracker.plan("Download", "Train", "Publish")        # anything not reached becomes SKIPPED
    with tracker.track("Download", source="s3"):
        ...
    with tracker.track("Train") as step:
        with Classification() as run:                    # nests: Train > k_folds > ...
            ...
            run.k_folds(5)
        step.annotate(best_accuracy=0.97)
    with tracker.track("Publish"):
        ...

@tracked("Clean data")             # a stage when a tracker is active, a plain call otherwise
def clean(df): ...

with stage("Inner step"):          # same for blocks inside library code
    annotate(rows=len(df))
```

Options for `ExecutionTracker`:

* `console`: `True` writes to stderr. It also accepts a stream, a callable such as `print` or
  `logger.info`, a `ConsoleOutput`, or `False`.
* `verbose`: show the developer view on the console.
* `console_level`: hide live lines for deeper levels. Failures are always shown.
* `task_records=True`: record one PASSED stage for every pool task. It is off by default because a
  run can train tens of thousands of configurations. Without it, each batch records the number of
  tasks, the worker pids and the total worker seconds.
* `outputs=[...]`: custom outputs (subclasses of `Output`).

Use `tracker.record(...)` for work that already finished elsewhere. Use
`tracker.track(..., parent=stage_id)` in threads that did not inherit the caller's context.

---

## 7. Design notes

* **The original exception is never swallowed.** A stage records the failure and returns `False`
  from `__exit__`, so the exception propagates unchanged (`pytest.raises(...).value is error`).
  Bookkeeping errors and output errors are caught separately and listed in
  `tracker.output_errors`. They are also reported once through the `fitfleet.execution` logger.
  A full disk or a closed console can never replace the application's exception.
* **Multiprocessing.** Workers never write logs. `TrackedTask` returns `(result, (pid, start,
  seconds))` and the parent unwraps it. On failure, the worker attaches an `fitfleet_task` dict to
  the exception. Exception attributes survive pickling, so the parent records which configuration
  failed, in which process and on which line. Only the parent process writes files, so records
  from different processes cannot mix. Each execution writes its own `execution_<id>.*` files, so
  concurrent executions cannot collide either.
* **Threads.** The current stage is a `ContextVar`, so each thread nests its own stages. A thread
  without an inherited context (such as the report thread) falls back to the active tracker of
  the process, but only when exactly one tracker is active. With several active trackers it gets
  none rather than guess. Every record change and output write happens under the tracker's lock,
  so log lines never interleave.
* **Overhead.**
  * `stage()` with no active tracker costs one `ContextVar` lookup: about 1 µs. The tests require
    under 20 µs.
  * A tracked stage costs about 20 µs in memory, or about 100 µs when it also writes the log and
    `.jsonl` files. Those files are flushed after every line so that a crash keeps them readable.
  * Planned stages are indexed, so the cost per stage stays the same in long executions with
    thousands of stages (tested).
  * Pool tasks are recorded per batch, not per task, unless `task_records=True`. A training run
    therefore adds only a handful of records.
* **Memory.**
  * A finished tracker is removed from the active list, and its `ContextVar` tokens are reset.
  * Records keep strings only: no exception objects, frames or data. A finished tracker is
    garbage-collected normally (tested).
  * A stage that a background thread closes after the execution ended is ignored. It is not
    duplicated.
* **Interrupted runs.** The `.jsonl` stream and the `.log` file are flushed after every record.
  `ExecutionLog.load` ignores a half-written last line and reports the run as `INCOMPLETE`.
* **Privacy.**
  * Metadata keys that look like credentials are masked as `***`. These include
    password/secret/token/api_key/credential/auth/cookie/session.
  * The same kinds of values are masked in error messages, as are `user:pass@` in URLs.
  * DataFrames, arrays and Series are reduced to `<Type shape=(...)>`, so dataset contents are
    never logged.
  * Records store file names, not full paths. An exception message that contains a path keeps it.
