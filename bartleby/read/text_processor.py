"""
Text file processor for Bartleby.

Handles processing of plain text files (.txt, .md, .rst, etc.)
into the Bartleby database format.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import shutil
import uuid

from bartleby.lib.consts import DEFAULT_PDF_PAGES_TO_SUMMARIZE
from bartleby.lib.embeddings import embed_chunk
from bartleby.read.nlp import chunk_page_body, clean_block_text

HASH_ALGORITHM = "sha256"
HASH_CHUNK_SIZE = 8_192


def save_chunk(
    embedding_model,
    cursor,
    body: str,
    index: int,
    page_id: str | None = None,
    summary_id: str | None = None,
):
    """Save a text chunk to the database with its embedding."""
    chunk_id = str(uuid.uuid4())
    cursor.execute(
        """
        INSERT INTO chunks(
            chunk_id,
            body,
            chunk_index,
            page_id,
            summary_id
        ) VALUES (? , ?, ?, ?, ?)
        """,
        (
            chunk_id,
            body,
            index,
            page_id,
            summary_id,
        )
    )

    chunk_embedding = embed_chunk(embedding_model, body)
    cursor.execute(
        """
        INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)
        """,
        (chunk_id, chunk_embedding.astype("float32").tobytes())
    )


def detect_encoding(file_path: Path) -> str:
    """
    Detect file encoding. Try UTF-8 first, then fall back to chardet if needed.
    """
    # Try UTF-8 first (most common)
    try:
        with file_path.open("r", encoding="utf-8") as f:
            f.read()
        return "utf-8"
    except UnicodeDecodeError:
        pass

    # Fall back to chardet for encoding detection
    try:
        import chardet
        with file_path.open("rb") as f:
            raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'
    except ImportError:
        # If chardet is not available, try common encodings
        for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
            try:
                with file_path.open("r", encoding=encoding) as f:
                    f.read()
                return encoding
            except UnicodeDecodeError:
                continue
        # Last resort: use utf-8 with error handling
        return "utf-8"


def process_text(
    connection,
    text_path: Path,
    db_path: Path,
    archive_path: Path,
    embedding_model,
    llm=None,
    llm_has_vision: bool = False,
    pdf_pages_to_summarize: int = DEFAULT_PDF_PAGES_TO_SUMMARIZE
):
    """
    Process a text file into the Bartleby database.

    Args:
        connection: Database connection
        text_path: Path to the text file
        db_path: Path to the database
        archive_path: Path to archive directory
        embedding_model: Model for generating embeddings
        llm: Optional LLM for summarization
        llm_has_vision: Whether LLM supports vision (not used for text)
        pdf_pages_to_summarize: Number of pages to summarize (not used for text)
    """
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found at: {text_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at: {db_path}")
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found at: {archive_path}")

    # Hash the file content for document_id
    hasher = hashlib.new(HASH_ALGORITHM)
    with text_path.open("rb") as in_file:
        for chunk in iter(lambda: in_file.read(HASH_CHUNK_SIZE), b""):
            hasher.update(chunk)

    document_id = hasher.hexdigest()

    # Archive the original file
    document_archive_path = archive_path / document_id / f"{document_id}{text_path.suffix}"
    document_archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(text_path, document_archive_path)

    # Detect encoding and read file
    encoding = detect_encoding(text_path)
    with text_path.open("r", encoding=encoding, errors="replace") as f:
        raw_body = f.read()

    # Clean the text (normalize, handle special characters, etc.)
    body = clean_block_text(raw_body)

    # Create database entries
    cursor = connection.cursor()

    # Insert document (text files are treated as single-page documents)
    cursor.execute(
        """
        INSERT INTO documents (
            document_id,
            origin_file_path,
            pages_count
        ) VALUES (?, ?, ?)
        """,
        (document_id, str(text_path), 1)
    )

    # Insert page
    page_id = str(uuid.uuid4())
    cursor.execute(
        """
        INSERT INTO pages(
            page_id,
            body,
            document_id,
            page_number
        ) VALUES (?, ?, ?, ?)
        """,
        (page_id, body, document_id, 1)
    )

    # Optional: Generate summary if LLM is provided
    if llm and pdf_pages_to_summarize > 0:
        from bartleby.read.llm import summarize_text
        summary_body = summarize_text(llm, body)
        if summary_body:
            summary_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO summaries (
                    summary_id,
                    body,
                    page_id
                ) VALUES (?, ?, ?)
                """,
                (summary_id, summary_body, page_id)
            )
            # Chunk and embed summary
            for chunk_index, chunk_body in enumerate(chunk_page_body(summary_body)):
                save_chunk(embedding_model, cursor, chunk_body, chunk_index, summary_id=summary_id)

    # Chunk and embed the main content
    for chunk_index, chunk_body in enumerate(chunk_page_body(body)):
        save_chunk(embedding_model, cursor, chunk_body, chunk_index, page_id=page_id)
