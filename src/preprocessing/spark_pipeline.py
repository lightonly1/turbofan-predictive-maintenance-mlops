"""
Module 2 - PySpark Preprocessing Pipeline
==========================================
Replaces the original pandas-based preprocessing with
distributed PySpark processing + Delta Lake storage.

Original notebook did:
- pd.read_csv()
- df.groupby().rolling().mean()
- df.merge()

We now do the same with PySpark — scalable to millions of rows.
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from loguru import logger
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import (
    DoubleType, IntegerType, StructField, StructType
)

from src.utils.config import get_config


# ── Schema Definition ────────────────────────────────────────────────────────

def get_schema() -> StructType:
    """
    Define explicit schema for NASA CMAPSS dataset.
    Industry practice: always define schema, never infer.
    """
    fields = [
        StructField("engine_id", IntegerType(), True),
        StructField("cycle", IntegerType(), True),
        StructField("setting_1", DoubleType(), True),
        StructField("setting_2", DoubleType(), True),
        StructField("setting_3", DoubleType(), True),
    ]
    # 21 sensor columns
    for i in range(1, 22):
        fields.append(StructField(f"sensor_{i}", DoubleType(), True))

    return StructType(fields)


# ── Spark Session ─────────────────────────────────────────────────────────────

def create_spark_session(app_name: str = "TurbofanRUL") -> SparkSession:
    """
    Create a Spark session optimized for local development.
    In production (Azure Databricks), this config is auto-managed.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")  # Use all CPU cores locally
        .config("spark.sql.shuffle.partitions", "8")  # Optimized for local
        .config("spark.driver.memory", "2g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )

    # Reduce verbose Spark logs
    spark.sparkContext.setLogLevel("ERROR")
    logger.info("✅ Spark session created successfully")
    logger.info(f"   Spark version: {spark.version}")
    logger.info(f"   Master: {spark.sparkContext.master}")

    return spark


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_raw_data(spark: SparkSession, file_path: str) -> DataFrame:
    """
    Load NASA CMAPSS raw text file into Spark DataFrame.
    Files are space-separated with no header.
    """
    logger.info(f"📂 Loading raw data from: {file_path}")

    schema = get_schema()

    df = (
        spark.read
        .format("csv")
        .option("sep", " ")
        .option("header", "false")
        .option("ignoreLeadingWhiteSpace", "true")
        .option("ignoreTrailingWhiteSpace", "true")
        .schema(schema)
        .load(file_path)
    )

    # Drop null columns (CMAPSS files have trailing spaces causing null cols)
    df = df.dropna(how="all")

    logger.info(f"✅ Loaded {df.count():,} rows, {len(df.columns)} columns")
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def compute_rul(df: DataFrame, rul_cap: int = 125) -> DataFrame:
    """
    Compute Remaining Useful Life (RUL) for each engine cycle.

    RUL = max_cycle_for_engine - current_cycle
    Capped at rul_cap (piecewise linear degradation model)
    """
    logger.info(f"🔧 Computing RUL with cap={rul_cap}")

    # Get max cycle per engine (= total life of engine)
    window_engine = Window.partitionBy("engine_id")

    df = df.withColumn(
        "max_cycle",
        F.max("cycle").over(window_engine)
    )

    # RUL = remaining cycles
    df = df.withColumn(
        "RUL",
        F.least(
            F.col("max_cycle") - F.col("cycle"),
            F.lit(rul_cap)
        )
    )

    df = df.drop("max_cycle")
    logger.info("✅ RUL computed and capped")
    return df


def drop_low_variance_sensors(df: DataFrame, drop_sensors: list) -> DataFrame:
    """
    Drop sensors with near-zero variance (no predictive value).
    Same sensors as original notebook: 1, 5, 10, 16, 18, 19
    """
    logger.info(f"🗑️  Dropping {len(drop_sensors)} low-variance sensors")
    df = df.drop(*drop_sensors)
    logger.info(f"✅ Remaining columns: {len(df.columns)}")
    return df


def compute_rolling_features(df: DataFrame, window_size: int = 5) -> DataFrame:
    """
    Compute rolling mean features for all sensor columns.

    In pandas: df.groupby('engine_id')[col].rolling(5).mean()
    In PySpark: Window.partitionBy().orderBy().rowsBetween()

    This is the key transformation — captures degradation trend.
    """
    logger.info(f"📊 Computing rolling mean features (window={window_size})")

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    # Define rolling window per engine, ordered by cycle
    rolling_window = (
        Window
        .partitionBy("engine_id")
        .orderBy("cycle")
        .rowsBetween(-(window_size - 1), 0)  # Look back window_size rows
    )

    # Add rolling mean for each sensor
    for col in sensor_cols:
        df = df.withColumn(
            f"{col}_rm",  # _rm = rolling mean
            F.avg(F.col(col)).over(rolling_window)
        )

    logger.info(f"✅ Added {len(sensor_cols)} rolling mean features")
    return df


def remove_warmup_rows(df: DataFrame, window_size: int = 5) -> DataFrame:
    """
    Remove first (window_size-1) cycles per engine.
    These rows don't have full rolling window data.
    Same as dropna() in original notebook.
    """
    logger.info("🧹 Removing warmup rows (incomplete rolling windows)")
    df = df.filter(F.col("cycle") >= window_size)
    logger.info(f"✅ Remaining rows after warmup removal: {df.count():,}")
    return df


def normalize_sensors(df: DataFrame) -> DataFrame:
    """
    Min-Max normalize sensor readings per engine.
    Industry practice: normalize before ML training.
    """
    logger.info("📐 Normalizing sensor features")

    sensor_cols = [c for c in df.columns if c.startswith("sensor_") and not c.endswith("_rm")]
    window_engine = Window.partitionBy("engine_id")

    for col in sensor_cols:
        min_col = F.min(col).over(window_engine)
        max_col = F.max(col).over(window_engine)
        df = df.withColumn(
            col,
            F.when(max_col - min_col == 0, 0.0)
             .otherwise((F.col(col) - min_col) / (max_col - min_col))
        )

    logger.info("✅ Sensor normalization complete")
    return df


# ── Data Quality Checks ───────────────────────────────────────────────────────

def run_data_quality_checks(df: DataFrame) -> bool:
    """
    Run basic data quality checks.
    Industry practice: validate data before saving.
    """
    logger.info("🔍 Running data quality checks...")
    passed = True

    # Check 1: No null values in key columns
    null_counts = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in ["engine_id", "cycle", "RUL"]
    ]).collect()[0]

    for col in ["engine_id", "cycle", "RUL"]:
        count = null_counts[col]
        if count > 0:
            logger.error(f"❌ Null values found in {col}: {count}")
            passed = False
        else:
            logger.info(f"   ✅ {col}: no nulls")

    # Check 2: RUL within valid range
    rul_stats = df.select(
        F.min("RUL").alias("min_rul"),
        F.max("RUL").alias("max_rul")
    ).collect()[0]

    if rul_stats["min_rul"] < 0:
        logger.error(f"❌ Negative RUL values found!")
        passed = False
    else:
        logger.info(f"   ✅ RUL range: [{rul_stats['min_rul']}, {rul_stats['max_rul']}]")

    # Check 3: Minimum number of engines
    engine_count = df.select("engine_id").distinct().count()
    if engine_count < 10:
        logger.warning(f"⚠️  Only {engine_count} engines found")
    else:
        logger.info(f"   ✅ Engine count: {engine_count}")

    # Check 4: Row count sanity
    row_count = df.count()
    if row_count < 1000:
        logger.error(f"❌ Too few rows: {row_count}")
        passed = False
    else:
        logger.info(f"   ✅ Total rows: {row_count:,}")

    status = "✅ ALL CHECKS PASSED" if passed else "❌ SOME CHECKS FAILED"
    logger.info(f"Data quality result: {status}")
    return passed


# ── Save Processed Data ───────────────────────────────────────────────────────

def save_as_parquet(df: DataFrame, output_path: str) -> None:
    """
    Save processed DataFrame.
    Uses pandas+CSV for Windows compatibility.
    In production (Azure Databricks) this writes Parquet to ADLS.
    """
    logger.info(f"💾 Saving processed data to: {output_path}")

    # Convert to pandas and save as CSV (Windows compatible)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    pdf = df.toPandas()
    csv_path = os.path.join(output_path, "train_processed.csv")
    pdf.to_csv(csv_path, index=False)

    logger.info(f"✅ Saved as CSV: {csv_path}")
    logger.info(f"   Rows: {len(pdf):,} | Columns: {len(pdf.columns)}")


def save_summary_stats(df: DataFrame, output_path: str) -> None:
    """Save summary statistics for EDA and monitoring baseline."""
    logger.info("📊 Computing and saving summary statistics...")

    stats = df.describe()
    stats_pd = stats.toPandas()
    stats_path = os.path.join(output_path, "summary_stats.csv")
    stats_pd.to_csv(stats_path, index=False)
    logger.info(f"✅ Summary stats saved to: {stats_path}")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline():
    """
    Main preprocessing pipeline.
    Orchestrates all steps in order.
    """
    logger.info("=" * 60)
    logger.info("🚀 Starting Turbofan RUL Preprocessing Pipeline")
    logger.info("=" * 60)

    # Load config
    config = get_config()
    data_cfg = config.data
    prep_cfg = config.preprocessing

    raw_dir = data_cfg["raw_dir"]
    processed_dir = data_cfg["processed_dir"]
    train_file = os.path.join(raw_dir, data_cfg["dataset"])

    # Create output directories
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    # Check input file exists
    if not os.path.exists(train_file):
        logger.error(f"❌ Data file not found: {train_file}")
        logger.info("   Please place train_FD001.txt in data/raw/")
        return

    # ── Step 1: Create Spark Session
    logger.info("\n📌 Step 1: Creating Spark Session")
    spark = create_spark_session()

    # ── Step 2: Load Raw Data
    logger.info("\n📌 Step 2: Loading Raw Data")
    df = load_raw_data(spark, train_file)

    # ── Step 3: Compute RUL
    logger.info("\n📌 Step 3: Computing RUL")
    df = compute_rul(df, rul_cap=prep_cfg["rul_cap"])

    # ── Step 4: Drop Low-Variance Sensors
    logger.info("\n📌 Step 4: Dropping Low-Variance Sensors")
    df = drop_low_variance_sensors(df, prep_cfg["drop_sensors"])

    # ── Step 5: Normalize Sensors
    logger.info("\n📌 Step 5: Normalizing Sensor Readings")
    df = normalize_sensors(df)

    # ── Step 6: Rolling Mean Features
    logger.info("\n📌 Step 6: Computing Rolling Mean Features")
    df = compute_rolling_features(df, window_size=prep_cfg["rolling_window"])

    # ── Step 7: Remove Warmup Rows
    logger.info("\n📌 Step 7: Removing Warmup Rows")
    df = remove_warmup_rows(df, window_size=prep_cfg["rolling_window"])

    # ── Step 8: Data Quality Checks
    logger.info("\n📌 Step 8: Data Quality Checks")
    quality_ok = run_data_quality_checks(df)

    if not quality_ok:
        logger.error("❌ Pipeline stopped due to data quality failures")
        return

    # ── Step 9: Save Processed Data
    logger.info("\n📌 Step 9: Saving Processed Data")
    parquet_path = os.path.join(processed_dir, "train_processed")
    save_as_parquet(df, parquet_path)
    save_summary_stats(df, processed_dir)

    # ── Step 10: Print Final Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"   Input:  {train_file}")
    logger.info(f"   Output: {parquet_path}")
    logger.info(f"   Rows:   {df.count():,}")
    logger.info(f"   Cols:   {len(df.columns)}")
    logger.info(f"   Features: {len([c for c in df.columns if c.endswith('_rm')])} rolling mean")
    logger.info("=" * 60)

    spark.stop()
    logger.info("🛑 Spark session stopped")


if __name__ == "__main__":
    run_pipeline()
