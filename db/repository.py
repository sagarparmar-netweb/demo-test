IMAGINE_RECORDS_TABLE = "imagine_records"


def fetch_pending(cursor, limit: int):
    cursor.execute(
        f"SELECT * FROM {IMAGINE_RECORDS_TABLE} WHERE status = 'CREATED' LIMIT %s",
        (limit,),
    )
    return cursor.fetchall()


def mark_error(cursor, record_id, error, ts):
    cursor.execute(
        f"""
        UPDATE {IMAGINE_RECORDS_TABLE}
        SET status='ERROR', log=%s, updated_at=%s
        WHERE id=%s
        """,
        (error[:1000], ts, record_id),
    )


def mark_processed(cursor, record_id, log_json, ts):
    cursor.execute(
        f"""
        UPDATE {IMAGINE_RECORDS_TABLE}
        SET status='PROCESSED', log=%s, updated_at=%s
        WHERE id=%s
        """,
        (log_json, ts, record_id),
    )
