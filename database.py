import io
import re
import sqlite3
import time
import uuid
from pathlib import Path
import pandas as pd

FOLDER = Path(__file__).parent / "datasets"
FOLDER.mkdir(exist_ok=True)

MAX_ROWS = 500
MAX_SECONDS = 5
BLOCKED = ["insert", "update", "delete", "drop", "alter", "create", "attach",
           "detach", "pragma", "vacuum", "reindex"]


class DatasetNotFound(Exception):
    pass


class BadSQL(Exception):
    pass


def quote(name):
    return '"' + str(name).replace('"', '""') + '"'


def find_file(dataset_id):
    # the id must look like the ones we create, so nobody can ask for ../something
    if not re.fullmatch(r"[0-9a-f]{8}", dataset_id):
        raise DatasetNotFound("Dataset not found")
    path = FOLDER / (dataset_id + ".db")
    if not path.exists():
        raise DatasetNotFound("Dataset not found")
    return path


def read_table(filename, content):
    name = filename.lower()
    if name.endswith(".csv"):
        # arabic csv files are often saved as cp1256 instead of utf-8
        for encoding in ("utf-8-sig", "cp1256", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(content), encoding=encoding)
            except UnicodeDecodeError:
                continue
    if name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(content))
    raise ValueError("Only .csv and .xlsx files are supported")


def save_dataset(filename, content):
    df = read_table(filename, content)
    df.columns = [str(c).strip() for c in df.columns]

    table = re.sub(r"\W+", "_", Path(filename).stem).strip("_") or "data"
    if table[0].isdigit():
        table = "t_" + table

    dataset_id = uuid.uuid4().hex[:8]
    conn = sqlite3.connect(FOLDER / (dataset_id + ".db"))
    df.to_sql(table, conn, index=False)
    conn.close()
    return dataset_id


def get_info(dataset_id):
    conn = sqlite3.connect(find_file(dataset_id))
    table = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchone()[0]
    count = conn.execute("SELECT COUNT(*) FROM " + quote(table)).fetchone()[0]

    columns = []
    for col in conn.execute("PRAGMA table_info(" + quote(table) + ")").fetchall():
        item = {"name": col[1], "type": col[2]}
        if col[2] == "TEXT":
            # short lists of values help the llm match words from the question
            sql = ("SELECT DISTINCT " + quote(col[1]) + " FROM " + quote(table)
                   + " WHERE " + quote(col[1]) + " IS NOT NULL LIMIT 16")
            values = [r[0] for r in conn.execute(sql)]
            if len(values) <= 15:
                item["values"] = values
        columns.append(item)

    sample = [list(r) for r in conn.execute("SELECT * FROM " + quote(table) + " LIMIT 3")]
    conn.close()
    return {"dataset_id": dataset_id, "table": table, "row_count": count,
            "columns": columns, "sample_rows": sample}


def schema_text(info):
    lines = ["Table " + quote(info["table"]) + " (" + str(info["row_count"]) + " rows)", "Columns:"]
    for c in info["columns"]:
        line = "- " + quote(c["name"]) + " (" + c["type"] + ")"
        if "values" in c:
            line += " values: " + ", ".join(str(v) for v in c["values"])
        lines.append(line)
    lines.append("Sample rows:")
    for row in info["sample_rows"]:
        lines.append(str(row))
    return "\n".join(lines)


def check_sql(sql):
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise BadSQL("The query is empty")
    if ";" in sql:
        raise BadSQL("Only one SQL statement is allowed")
    if not re.match(r"(?is)^(select|with)\b", sql):
        raise BadSQL("Only SELECT queries are allowed")
    for word in BLOCKED:
        if re.search(r"\b" + word + r"\b", sql.lower()):
            raise BadSQL("'" + word + "' is not allowed")
    return sql


def only_read(action, arg1, arg2, db_name, source):
    allowed = [sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
               getattr(sqlite3, "SQLITE_RECURSIVE", 33)]
    if action in allowed:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def run_sql(dataset_id, sql):
    sql = check_sql(sql)
    path = find_file(dataset_id)

    # read only connection + authorizer, so even a query that passed the check cannot change anything
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.set_authorizer(only_read)
    try:
        # without this, a wrong column name in double quotes is treated as a text value
        conn.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
    except AttributeError:
        pass   # python older than 3.12
    start = time.time()
    conn.set_progress_handler(lambda: 1 if time.time() - start > MAX_SECONDS else 0, 100000)

    try:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as e:
        raise BadSQL(str(e))
    finally:
        conn.close()

    truncated = len(rows) > MAX_ROWS
    rows = [list(r) for r in rows[:MAX_ROWS]]
    return {"sql": sql, "columns": columns, "rows": rows,
            "row_count": len(rows), "truncated": truncated}
