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
