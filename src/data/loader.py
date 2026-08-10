"""
Unified data-ingestion layer.

Every input source (CSV, Excel, SQL database) is resolved into a single pandas
DataFrame here. The rest of the pipeline is completely decoupled from *where*
the data came from — it only ever sees a DataFrame. This is the "pluggable
input layer" described in the report.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ..utils.logger import get_logger

log = get_logger(__name__)


SUPPORTED_DB_DIALECTS = {
    "postgresql": "postgresql+psycopg2",
    "mysql": "mysql+pymysql",
    "sqlserver": "mssql+pyodbc",
    "sqlite": "sqlite",
}


def load_from_file(file, filename: str | None = None) -> pd.DataFrame:
    """Load a CSV or Excel file.

    `file` may be a path (str/Path) or a file-like object (e.g. a Streamlit
    upload). `filename` is used to detect the extension when `file` is a buffer.
    """
    name = filename or (str(file) if isinstance(file, (str, Path)) else "")
    ext = Path(name).suffix.lower()

    if ext in {".csv", ".txt", ""}:
        # try to be forgiving about delimiters / encodings
        df = _read_csv_robust(file)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(file)  # needs openpyxl for .xlsx
    else:
        raise ValueError(f"Unsupported file type: {ext!r}. Use .csv or .xlsx")

    df = _basic_clean(df)
    log.info("Loaded file '%s' -> shape %s", name, df.shape)
    return df


def _read_csv_robust(file) -> pd.DataFrame:
    """Read a CSV, retrying with a couple of common fallbacks."""
    # Streamlit uploads are bytes buffers; make them re-readable.
    if hasattr(file, "read") and not isinstance(file, (str, Path)):
        raw = file.read()
        if isinstance(raw, bytes):
            for enc in ("utf-8", "latin-1"):
                try:
                    return pd.read_csv(io.StringIO(raw.decode(enc)), sep=None, engine="python")
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
        return pd.read_csv(io.StringIO(raw if isinstance(raw, str) else raw.decode("latin-1")))
    # path-like
    try:
        return pd.read_csv(file, sep=None, engine="python")
    except pd.errors.ParserError:
        return pd.read_csv(file)


def load_from_database(
    dialect: str,
    query: str,
    *,
    host: str = "",
    port: str | int = "",
    database: str = "",
    user: str = "",
    password: str = "",
    sqlite_path: str = "",
) -> pd.DataFrame:
    """Connect to a relational database via SQLAlchemy and run one query.

    Only reads: the user supplies a SELECT query that returns one flat table.
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError as e:
        raise RuntimeError(
            "SQLAlchemy is not installed. Run: pip install sqlalchemy"
        ) from e

    dialect = dialect.lower()
    if dialect not in SUPPORTED_DB_DIALECTS:
        raise ValueError(
            f"Unsupported dialect {dialect!r}. "
            f"Supported: {list(SUPPORTED_DB_DIALECTS)}"
        )

    if dialect == "sqlite":
        conn_str = f"sqlite:///{sqlite_path}"
    else:
        driver = SUPPORTED_DB_DIALECTS[dialect]
        conn_str = f"{driver}://{user}:{password}@{host}:{port}/{database}"

    log.info("Connecting to %s database ...", dialect)
    engine = create_engine(conn_str)
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    df = _basic_clean(df)
    log.info("Loaded %s rows from database", len(df))
    return df


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal, safe cleaning applied to every source.

    Deliberately conservative — real cleaning decisions are made later by the
    agent. Here we only strip whitespace from column names and drop fully-empty
    rows/columns, which are always safe.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df = df.reset_index(drop=True)
    return df
