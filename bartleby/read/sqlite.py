from pathlib import Path
from typing import Optional

import apsw
import sqlite_vec

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def create_db(db_dir_path: Path, embedding_dimension: Optional[int] = None):
    """
    Create a new Bartleby database.

    Args:
        db_dir_path: Directory path for the database
        embedding_dimension: Dimension of embeddings (default: 768 for BAAI/bge-base-en-v1.5)
    """
    if embedding_dimension is None:
        embedding_dimension = 768

    sql_text = SCHEMA_PATH.read_text(encoding="utf-8")

    # Replace the embedding dimension placeholder
    sql_text = sql_text.replace("embedding float[768]", f"embedding float[{embedding_dimension}]")

    connection = apsw.Connection(str(db_dir_path / "bartleby.db"))

    # Enable extension loading
    connection.enable_load_extension(True)

    # Load sqlite-vec extension
    sqlite_vec.load(connection)

    cursor = connection.cursor()

    # Enable WAL mode for better concurrent write performance
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")

    # Store embedding dimension in metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    cursor.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('embedding_dimension', ?)",
        (str(embedding_dimension),)
    )

    # Execute all statements - APSW properly handles multiple statements
    # by yielding each one when we iterate through execute()
    for _ in cursor.execute(sql_text):
        pass  # Just consume the iterator

    try:
        cursor.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('optimize')")
    except apsw.Error:
        pass

    connection.close()


def get_connection(db_path: Path):
    connection = apsw.Connection(str(db_path))

    # Set busy timeout to wait up to 30 seconds for locks
    connection.set_busy_timeout(30000)

    # Enable extension loading
    connection.enable_load_extension(True)

    # Load sqlite-vec extension
    sqlite_vec.load(connection)

    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")

    return connection


def get_embedding_dimension(db_path: Path) -> Optional[int]:
    """
    Get the embedding dimension from database metadata.

    Args:
        db_path: Path to the database file

    Returns:
        Embedding dimension, or None if not set
    """
    try:
        connection = get_connection(db_path)
        cursor = connection.cursor()
        result = cursor.execute(
            "SELECT value FROM metadata WHERE key = 'embedding_dimension'"
        ).fetchone()
        connection.close()

        if result:
            return int(result[0])
        return None
    except Exception:
        return None


