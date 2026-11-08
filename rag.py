import sqlite3
from pathlib import Path

import sqlite_vec

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DB = Path("index.sqlite3")


def load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def open_index(path, *, read_only=False):
    database = path
    if read_only:
        if not path.is_file():
            raise FileNotFoundError(f"Index {path} does not exist. Run ingest.py first.")
        database = path.resolve().as_uri() + "?mode=ro"

    connection = sqlite3.connect(database, uri=read_only)
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
    except (sqlite3.Error, AttributeError) as error:
        connection.close()
        raise RuntimeError(f"Could not load the sqlite-vec extension: {error}") from error

    connection.row_factory = sqlite3.Row
    return connection
