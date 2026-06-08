import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="MLOps batch signal job")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-file", required=True, dest="log_file")
    return parser.parse_args()

def setup_logging(log_file):
    logger = logging.getLogger("mlops_job")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

def load_config(config_path, logger):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must be a valid YAML mapping.")
    missing = {"seed", "window", "version"} - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    if not isinstance(cfg["seed"], int):
        raise ValueError(f"'seed' must be an integer")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"'window' must be a positive integer")
    if not isinstance(cfg["version"], str):
        raise ValueError(f"'version' must be a string")
    logger.info(f"Config loaded  — seed={cfg['seed']}, window={cfg['window']}, version={cfg['version']}")
    return cfg

def load_dataset(input_path, logger):
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1:
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip().strip('"') for line in f.readlines()]
            df = pd.read_csv(io.StringIO("\n".join(lines)))
    except Exception as exc:
        raise ValueError(f"Invalid CSV format: {exc}") from exc
    if df.empty:
        raise ValueError("Input CSV is empty.")
    if "close" not in df.columns:
        raise ValueError(f"Required column 'close' not found. Available columns: {list(df.columns)}")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    n_invalid = df["close"].isna().sum()
    if n_invalid > 0:
        logger.warning(f"{n_invalid} rows with non-numeric 'close' values will be dropped.")
        df = df.dropna(subset=["close"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows remain after coercing 'close' to numeric.")
    logger.info(f"Dataset loaded — {len(df):,} rows, columns: {list(df.columns)}")
    return df

def compute_rolling_mean(df, window, logger):
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=window).mean()
    n_nan = df["rolling_mean"].isna().sum()
    logger.info(f"Rolling mean computed — window={window}, warm-up rows excluded (NaN): {n_nan}")
    return df

def compute_signal(df, logger):
    df = df.copy()
    valid_mask = df["rolling_mean"].notna()
    df.loc[valid_mask, "signal"] = (df.loc[valid_mask, "close"] > df.loc[valid_mask, "rolling_mean"]).astype(int)
    n_signals = valid_mask.sum()
    logger.info(f"Signal generated — {n_signals:,} valid rows used for signal computation")
    return df

def compute_metrics(df, cfg, latency_ms):
    valid_signals = df["signal"].dropna()
    return {
        "version": cfg["version"],
        "rows_processed": len(df),
        "metric": "signal_rate",
        "value": round(float(valid_signals.mean()), 4),
        "latency_ms": int(latency_ms),
        "seed": cfg["seed"],
        "status": "success",
    }

def write_metrics(metrics, output_path):
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

def write_error_metrics(output_path, version, message):
    try:
        with open(output_path, "w") as f:
            json.dump({"version": version, "status": "error", "error_message": message}, f, indent=2)
    except Exception:
        pass

def main():
    args = parse_args()
    logger = setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("MLOps Batch Job — START")
    logger.info(f"  input  : {args.input}")
    logger.info(f"  config : {args.config}")
    logger.info(f"  output : {args.output}")
    logger.info(f"  log    : {args.log_file}")
    logger.info("=" * 60)
    t_start = time.time()
    version = "unknown"
    try:
        cfg = load_config(args.config, logger)
        version = cfg["version"]
        np.random.seed(cfg["seed"])
        logger.info(f"Random seed set — numpy.random.seed({cfg['seed']})")
        df = load_dataset(args.input, logger)
        df = compute_rolling_mean(df, cfg["window"], logger)
        df = compute_signal(df, logger)
        latency_ms = (time.time() - t_start) * 1000
        metrics = compute_metrics(df, cfg, latency_ms)
        write_metrics(metrics, args.output)
        logger.info(f"Metrics written — {args.output}")
        logger.info(f"Summary — rows_processed={metrics['rows_processed']:,}, signal_rate={metrics['value']}, latency_ms={metrics['latency_ms']}")
        logger.info("MLOps Batch Job — SUCCESS")
        logger.info("=" * 60)
        print(json.dumps(metrics, indent=2))
        sys.exit(0)
    except Exception as exc:
        logger.error(f"Job FAILED: {exc}", exc_info=True)
        write_error_metrics(args.output, version, str(exc))
        print(json.dumps({"version": version, "status": "error", "error_message": str(exc)}, indent=2))
        logger.info("MLOps Batch Job — FAILED")
        logger.info("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()