"""
Unit Tests - PySpark Preprocessing Pipeline
Run: pytest tests/unit/test_preprocessing.py -v
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing."""
    from src.preprocessing.spark_pipeline import create_spark_session
    spark = create_spark_session("TurbofanTest")
    yield spark
    spark.stop()


def test_spark_session_created(spark):
    """Spark session should be created successfully."""
    assert spark is not None
    assert spark.version is not None


def test_schema_has_correct_columns():
    """Schema should have engine_id, cycle, 3 settings, 21 sensors."""
    from src.preprocessing.spark_pipeline import get_schema
    schema = get_schema()
    field_names = [f.name for f in schema.fields]

    assert "engine_id" in field_names
    assert "cycle" in field_names
    assert "setting_1" in field_names
    assert "sensor_1" in field_names
    assert "sensor_21" in field_names
    assert len(field_names) == 26  # 2 + 3 + 21


def test_rul_computation(spark):
    """RUL should be correctly computed and capped."""
    from pyspark.sql import functions as F
    from src.preprocessing.spark_pipeline import compute_rul

    # Create test data
    data = [
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 1), (2, 2), (2, 3),
    ]
    df = spark.createDataFrame(data, ["engine_id", "cycle"])

    result = compute_rul(df, rul_cap=125)

    # Engine 1: max_cycle=5, so cycle 1 → RUL=4, cycle 5 → RUL=0
    engine1 = result.filter(F.col("engine_id") == 1).orderBy("cycle")
    rows = engine1.collect()

    assert rows[0]["RUL"] == 4   # cycle 1: 5-1=4
    assert rows[4]["RUL"] == 0   # cycle 5: 5-5=0


def test_rul_cap_applied(spark):
    """RUL should never exceed the cap value."""
    from pyspark.sql import functions as F
    from src.preprocessing.spark_pipeline import compute_rul

    # Engine with 200 cycles — RUL at cycle 1 would be 199 without cap
    data = [(1, i) for i in range(1, 201)]
    df = spark.createDataFrame(data, ["engine_id", "cycle"])

    result = compute_rul(df, rul_cap=125)
    max_rul = result.select(F.max("RUL")).collect()[0][0]

    assert max_rul <= 125


def test_drop_sensors(spark):
    """Specified sensors should be dropped."""
    from src.preprocessing.spark_pipeline import drop_low_variance_sensors

    data = [(1, 1, 0.1, 0.2, 0.3)]
    df = spark.createDataFrame(data, ["engine_id", "cycle", "sensor_1", "sensor_2", "sensor_3"])

    result = drop_low_variance_sensors(df, ["sensor_1"])

    assert "sensor_1" not in result.columns
    assert "sensor_2" in result.columns
    assert "sensor_3" in result.columns


def test_rolling_features_added(spark):
    """Rolling mean columns should be added for each sensor."""
    from pyspark.sql import functions as F
    from src.preprocessing.spark_pipeline import compute_rolling_features

    data = [(1, i, float(i)) for i in range(1, 11)]
    df = spark.createDataFrame(data, ["engine_id", "cycle", "sensor_2"])

    result = compute_rolling_features(df, window_size=3)

    assert "sensor_2_rm" in result.columns


def test_data_quality_check_passes(spark):
    """Data quality check should pass for valid data."""
    from pyspark.sql import functions as F
    from src.preprocessing.spark_pipeline import run_data_quality_checks

    # Create valid dataset
    data = [(i, j, float(j), float(100 - j))
            for i in range(1, 11)
            for j in range(1, 101)]

    df = spark.createDataFrame(data, ["engine_id", "cycle", "sensor_2", "RUL"])
    result = run_data_quality_checks(df)
    assert result is True
