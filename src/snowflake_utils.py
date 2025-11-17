# ============================================================
# Snowflake Utilities for Logging Model Predictions
# ============================================================

import snowflake.connector
from datetime import datetime
from .snowflake_config import SNOWFLAKE_CONFIG

def get_connection():
    return snowflake.connector.connect(
        user=SNOWFLAKE_CONFIG["user"],
        password=SNOWFLAKE_CONFIG["password"],
        account=SNOWFLAKE_CONFIG["account"],
        warehouse=SNOWFLAKE_CONFIG["warehouse"],
        database=SNOWFLAKE_CONFIG["database"],
        schema=SNOWFLAKE_CONFIG["schema"],
    )

def log_prediction_to_snowflake(label, confidence):
    """
    Inserts model predictions into Snowflake for:
    - Analytics
    - Dashboarding
    - Model monitoring
    """

    try:
        conn = get_connection()
        cur = conn.cursor()

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS FER_PREDICTIONS (
            ID INTEGER AUTOINCREMENT,
            TIMESTAMP TIMESTAMP,
            LABEL STRING,
            CONFIDENCE FLOAT
        );
        """
        cur.execute(create_table_sql)

        insert_sql = """
        INSERT INTO FER_PREDICTIONS (TIMESTAMP, LABEL, CONFIDENCE)
        VALUES (%s, %s, %s)
        """

        cur.execute(insert_sql, (datetime.now(), label, confidence))
        conn.commit()

        print(f"[Snowflake] Logged prediction: {label} ({confidence})")

    except Exception as e:
        print(f"[Snowflake Error] {e}")

    finally:
        try:
            cur.close()
            conn.close()
        except:
            pass
