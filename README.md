# MLOps Batch Job — Task 0

A minimal MLOps-style batch pipeline that computes a rolling-mean trading signal on OHLCV close prices.

## Project Structure

```
.
├── run.py           # Main pipeline script
├── config.yaml      # Job configuration
├── data.csv         # Input OHLCV dataset (10,000 rows)
├── requirements.txt # Python dependencies
├── Dockerfile       # Container definition
├── README.md        # This file
├── metrics.json     # Sample output from successful run
└── run.log          # Sample log from successful run
```

## Local Run

```bash
pip install -r requirements.txt

python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

## Docker

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

## Example metrics.json

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.499,
  "latency_ms": 127,
  "seed": 42,
  "status": "success"
}
```
