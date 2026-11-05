this is a scikit-learn package for running many tests in multiprocessing with minimal garbage creation

## Usage notes

- `train_test_split(ratio)`: `ratio` is the fraction of rows used for **training** (0.8 = 80% train).
- Scalers and polynomial features (`pipeline.fetchers.standard_scaler()` etc.) are refitted on the
  training rows of every split, so test rows never leak into preprocessing.
- Classification uses stratified splits. All models in a run are scored on the same folds.
- Results carry metrics and predictions only. `filter_by()` (or `refit(results)`) fits the chosen
  models; the fitted model is a scikit-learn `Pipeline` that takes raw rows.
- The worker pool is reused across runs on the same object. Use `with Classification() as c:` or
  call `c.close()` to stop the workers.

## Tests

```
pytest                              # unit tests (load tests are skipped)
pytest --run-load -m load -s        # CPU load tests, a few minutes, prints a CPU timeline
python -m benchmarks.bench          # timing / memory benchmark
```


## Web service (Django)

`webapp/` is a Django project that runs ml_pipe as a background service with a browser GUI and a REST API.

```
pip install -r requirements.txt
cd webapp
python manage.py migrate
python manage.py runserver          # http://127.0.0.1:8000
```

In the GUI: **Upload data** (CSV or Excel) → open the dataset → **New experiment** (target, features, models,
hyper-parameter grid, scaling / polynomial features, feature selection, validation) → watch the progress bar →
compare every configuration in the sortable leaderboard → open a result for charts, the PDF report and the
fitted model.

**Job runner.** Experiments are queued in the database and run one at a time. With `runserver` a background
thread runs them (default). To run jobs in a separate process instead, e.g. behind gunicorn/waitress:

```
set MLWEB_INPROCESS_WORKER=0        # PowerShell: $env:MLWEB_INPROCESS_WORKER="0"
python manage.py runserver
python manage.py run_worker         # in a second terminal; --once processes the queue and exits
```

Run only one job runner per database. A job left running by a stopped runner is marked failed on the next start.

**REST API** (browsable at `/api/`):

| | |
|---|---|
| `POST /api/datasets/` | multipart `name`, `file` |
| `POST /api/experiments/` | JSON config, validated like the form, then queued |
| `GET /api/experiments/{id}/` | status, stage, `tasks_done` / `tasks_total` |
| `GET /api/experiments/{id}/results/` | metrics per configuration |
| `GET /api/results/{id}/` | one result with its out-of-sample predictions |
| `GET /api/results/{id}/report/`, `/model/` | PDF report, fitted model (joblib) |

**Downloaded models** are a dict: `bundle = joblib.load(path)`, then
`bundle["model"].predict(df[bundle["feature_columns"]])`. The model is a scikit-learn Pipeline that applies the
experiment's scaling, polynomial features and feature selection itself. For classification, predictions are class
indexes into `bundle["class_names"]`.

**Not for the open internet as is:** there are no user accounts (everyone sees everything), the database is SQLite,
and `DEBUG` is on. Set `DJANGO_SECRET_KEY`, `DJANGO_DEBUG=0` and `DJANGO_ALLOWED_HOSTS` before deploying anywhere shared.


## Docker (run on your own machine)

Needs Docker Desktop running.

```
docker compose up --build -d      # build the image and start in the background
                                  # open http://localhost:8000
docker compose logs -f            # watch the server and training progress (Ctrl+C stops watching only)
docker compose down               # stop (your data stays)
docker compose down -v            # stop and DELETE all datasets, experiments and models
```

Datasets, experiments, reports and models live on the `mlpipe-data` Docker volume, so they survive restarts
and rebuilds. After changing the code, run `docker compose up --build -d` again.

`shm_size: 2gb` in `docker-compose.yml` is required: ml_pipe passes the data to its worker processes through
shared memory, and Docker's default of 64 MB is too small for real datasets. Raise it for data larger than ~1 GB.
If it is too small the experiment fails with "Not enough shared memory".
