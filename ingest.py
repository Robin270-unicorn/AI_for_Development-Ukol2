import argparse
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag import DEFAULT_DB, load_embedder, open_index

CHUNK_TOKENS = 200
OVERLAP_TOKENS = 40
EMBEDDING_BATCH = 64


def chunk_page(text, tokenizer):
    text = " ".join(text.split())
    offsets = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
    )["offset_mapping"]

    for start in range(0, len(offsets), CHUNK_TOKENS - OVERLAP_TOKENS):
        end = min(start + CHUNK_TOKENS, len(offsets))
        yield text[offsets[start][0]:offsets[end - 1][1]]
        if end == len(offsets):
            break


def read_chunks(pdf_paths, tokenizer):
    for path in pdf_paths:
        has_text = False
        try:
            with path.open("rb") as file:
                reader = PdfReader(file)
                if reader.is_encrypted:
                    raise ValueError("encrypted PDFs are not supported")
                for page_number, page in enumerate(reader.pages, start=1):
                    for chunk in chunk_page(page.extract_text() or "", tokenizer):
                        has_text = True
                        yield path.name, page_number, chunk
        except (OSError, PdfReadError, ValueError) as error:
            raise ValueError(f"{path.name}: {error}") from error
        if not has_text:
            print(f"Warning: {path.name} has no extractable text.", file=sys.stderr)


def build_index(pdf_dir, db_path):
    pdf_paths = sorted(
        path for path in pdf_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {pdf_dir}.")

    embedder = load_embedder()
    chunks = list(read_chunks(pdf_paths, embedder.tokenizer))
    if not chunks:
        raise ValueError("No text to index. Scanned PDFs require OCR first.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(open_index(db_path)) as connection, connection:
        # Readers see either the old index or the complete replacement.
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                page INTEGER NOT NULL CHECK (page > 0),
                text TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)
        connection.execute("DELETE FROM chunks")
        for start in range(0, len(chunks), EMBEDDING_BATCH):
            batch = chunks[start:start + EMBEDDING_BATCH]
            vectors = embedder.encode(
                [chunk[2] for chunk in batch],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            connection.executemany(
                "INSERT INTO chunks (filename, page, text, embedding) VALUES (?, ?, ?, ?)",
                ((*chunk, vector.tobytes()) for chunk, vector in zip(batch, vectors, strict=True)),
            )

    return len(chunks), len({filename for filename, _, _ in chunks})


def main():
    parser = argparse.ArgumentParser(description="Index the PDFs in a folder.")
    parser.add_argument("pdf_dir", type=Path, help="Folder with text-based PDFs")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite index path")
    args = parser.parse_args()

    try:
        chunks, files = build_index(args.pdf_dir, args.db)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        parser.exit(1, f"Indexing failed: {error}\n")
    print(f"Indexed {chunks} chunks from {files} PDFs in {args.db}.")


if __name__ == "__main__":
    main()
