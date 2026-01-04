import io
import sys
import time
import logging
from datetime import datetime

import boto3
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

S3_BUCKET = "vandatrack-option-data"
OUTPUT_BUCKET = "vandalabs-data"
OUTPUT_PREFIX = "turnover/master/"
AWS_REGION = "us-east-1"

FILES = {
    "retail_call": [
        "C_ATM_small_turnover.csv",
        "C_ITM_small_turnover.csv",
        "C_OTM_small_turnover.csv",
    ],
    "retail_put": [
        "P_ATM_small_turnover.csv",
        "P_ITM_small_turnover.csv",
        "P_OTM_small_turnover.csv",
    ],
    "inst_call": [
        "C_ATM_large_turnover.csv",
        "C_ITM_large_turnover.csv",
        "C_OTM_large_turnover.csv",
    ],
    "inst_put": [
        "P_ATM_large_turnover.csv",
        "P_ITM_large_turnover.csv",
        "P_OTM_large_turnover.csv",
    ],
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

log = logging.getLogger("turnover_master")


# ============================================================
# AWS
# ============================================================

s3 = boto3.client("s3", region_name=AWS_REGION)


# ============================================================
# HELPERS
# ============================================================

def load_from_s3(key: str) -> pd.DataFrame:
    log.info(f"Downloading s3://{S3_BUCKET}/{key}")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def melt_wide(df: pd.DataFrame) -> pd.DataFrame:
    date_col = df.columns[0]
    return df.melt(
        id_vars=[date_col],
        var_name="ticker",
        value_name="turnover",
    )


def normalise_ticker(t: str) -> str:
    t = str(t).upper()
    for d in [" ", "-", "/", ".", "_"]:
        t = t.split(d)[0]
    return "".join(ch for ch in t if ch.isalnum())


import gc

def build_group(name: str, file_list: list[str]) -> pd.DataFrame:
    log.info(f"=== Building {name} ===")
    t0 = time.time()

    acc = None

    for key in file_list:
        log.info(f"Processing {key}")

        # 1) Load + melt ONE file only
        df = load_from_s3(key)
        log.info(f"{key}: raw={df.shape}")

        df = melt_wide(df)
        log.info(f"{key}: melted={df.shape}")

        df["ticker_norm"] = df["ticker"].map(normalise_ticker)
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
        df.dropna(subset=["turnover"], inplace=True)
        df["date"] = pd.to_datetime(df.iloc[:, 0])

        # 2) Aggregate immediately
        grouped = (
            df.groupby(["date", "ticker", "ticker_norm"], as_index=False)
              .agg(turnover=("turnover", "sum"))
        )
        log.info(f"{key}: grouped={grouped.shape}")

        # 3) Merge into accumulator (small)
        if acc is None:
            acc = grouped
        else:
            acc = (
                pd.concat([acc, grouped], ignore_index=True)
                  .groupby(
                      ["date", "ticker", "ticker_norm"],
                      as_index=False
                  )
                  .agg(turnover=("turnover", "sum"))
            )

        # 4) Free memory aggressively
        del df, grouped
        gc.collect()

    log.info(
        f"{name}: final rows={len(acc):,} "
        f"(took {time.time() - t0:.1f}s)"
    )

    return acc


def save_parquet(df: pd.DataFrame, name: str) -> None:
    key = f"{OUTPUT_PREFIX}{name}.parquet"

    out = io.BytesIO()
    df.to_parquet(out, index=False)
    out.seek(0)

    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=key,
        Body=out.getvalue(),
    )

    size_mb = out.tell() / 1024 / 1024
    log.info(
        f"Uploaded s3://{OUTPUT_BUCKET}/{key} "
        f"({size_mb:.1f} MB)"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log.info("Turnover master cron started")

    start = time.time()

    for name, files in FILES.items():
        group_start = time.time()

        df = build_group(name, files)
        save_parquet(df, name)

        del df
        log.info(
            f"{name}: completed in {time.time() - group_start:.1f}s"
        )

    log.info(
        f"All groups completed in {time.time() - start:.1f}s"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Fatal error in turnover master cron")
        raise
