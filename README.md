# trading-signal-pipeline

A minimal, production-ready MLOps batch pipeline that generates binary trading signals from OHLCV price data using a rolling mean strategy.

Built as part of the Primetrade.ai ML/MLOps Engineering Internship — Task 0 Technical Assessment.

---

## What it does

1. Loads OHLCV price data from a CSV file
2. Validates config and input data
3. Computes a rolling mean on the `close` price
4. Generates a binary signal: `1` if `close > rolling_mean`, else `0`
5. Writes structured metrics to JSON and detailed logs to a log file

---

## Project Structure

```
trading-signal-pipeline/
├── run.py              # Main pipeline script
├── config.yaml         # Job configuration (seed, window, version)
├── data.csv            # Input OHLCV dataset (10,000 rows)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
├── README.md           # This file
├── metrics.json        # Sample output from a successful run
└── run.log             # Sample log from a successful run
```

---

## Quickstart

### Prerequisites

- Python 3.9+
- pip

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

---

## Docker

### Build

```bash
docker build -t mlops-task .
```

### Run

```bash
docker run --rm mlops-task
```

- Exit code `0` → success
- Exit code non-zero → failure (error metrics still written to `metrics.json`)

---

## Configuration

Edit `config.yaml` to change job parameters:

```yaml
seed: 42       # Random seed for reproducibility
window: 5      # Rolling mean window size
version: "v1"  # Pipeline version label
```

| Key     | Type   | Description                     |
|---------|--------|---------------------------------|
| seed    | int    | Ensures deterministic runs      |
| window  | int    | Rolling mean lookback period    |
| version | string | Version tag in metrics output   |

---

## Output

### metrics.json (success)

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 59,
  "seed": 42,
  "status": "success"
}
```

### metrics.json (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found."
}
```

---

## Logging

Logs are written to both stdout and `run.log`. Every run captures:

- Job start timestamp
- Config values (seed, window, version)
- Rows loaded and column names
- Rolling mean warm-up rows excluded
- Signal computation row count
- Metrics summary
- Job end status

Example:

```
2026-06-08T14:41:50 | INFO     | MLOps Batch Job — START
2026-06-08T14:41:50 | INFO     | Config loaded  — seed=42, window=5, version=v1
2026-06-08T14:41:50 | INFO     | Dataset loaded — 10,000 rows
2026-06-08T14:41:50 | INFO     | Rolling mean computed — window=5, warm-up rows excluded (NaN): 4
2026-06-08T14:41:50 | INFO     | Signal generated — 9,996 valid rows used for signal computation
2026-06-08T14:41:50 | INFO     | Summary — rows_processed=10,000, signal_rate=0.4991, latency_ms=59
2026-06-08T14:41:50 | INFO     | MLOps Batch Job — SUCCESS
```

---

## Error Handling

The pipeline handles all edge cases gracefully:

| Scenario | Behaviour |
|---|---|
| Missing input file | Logs error, writes error `metrics.json`, exits non-zero |
| Empty CSV | Logs error, writes error `metrics.json`, exits non-zero |
| Missing `close` column | Logs error, writes error `metrics.json`, exits non-zero |
| Invalid config keys | Logs error, writes error `metrics.json`, exits non-zero |
| Non-numeric close values | Drops invalid rows with a warning, continues |

---

## Design Decisions

- **NaN handling:** First `window - 1` rows produce `NaN` rolling mean and are excluded from signal computation. `signal_rate` is calculated only over valid rows.
- **Determinism:** `numpy.random.seed(seed)` is set immediately after config load — results are fully reproducible across runs.
- **No hardcoded paths:** All file paths are passed via CLI arguments.
- **Metrics always written:** `metrics.json` is written in both success and error cases.

---

## Tech Stack

- Python 3.9
- pandas 2.1.4
- numpy 1.26.4
- PyYAML 6.0.1
- Docker (python:3.9-slim)
