"""
JSON file processor for Bartleby.

Handles processing of JSON and JSONL (JSON Lines) files
into the Bartleby database format.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, List
import uuid

from loguru import logger

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


def get_nested_value(obj: dict, path: str) -> Any:
    """
    Get a nested value from a dictionary using dot notation.

    Args:
        obj: Dictionary to extract value from
        path: Dot-separated path (e.g., 'metadata.author')

    Returns:
        Value at the path, or None if not found
    """
    keys = path.split('.')
    current = obj
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def extract_attributes(obj: dict, attributes: List[str]) -> str:
    """
    Extract specified attributes from a JSON object and format as text.

    Args:
        obj: JSON object (dictionary)
        attributes: List of attribute paths to extract (supports dot notation)

    Returns:
        Formatted text with extracted attributes
    """
    parts = []
    for attr in attributes:
        value = get_nested_value(obj, attr)
        if value is not None:
            # Convert value to string
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, ensure_ascii=False)
            else:
                value_str = str(value)
            parts.append(f"{attr}: {value_str}")

    return "\n\n".join(parts)


def is_jsonl(file_path: Path) -> bool:
    """
    Check if a file is JSONL (JSON Lines) format.

    Returns True if:
    - File extension is .jsonl
    - OR first line parses as JSON object and second line also does
    """
    if file_path.suffix.lower() == '.jsonl':
        return True

    # Try to detect JSONL by checking first two lines
    try:
        with file_path.open('r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()

            if first_line and second_line:
                json.loads(first_line)
                json.loads(second_line)
                return True
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    return False


def process_json(
    connection,
    json_path: Path,
    db_path: Path,
    archive_path: Path,
    embedding_model,
    attributes: List[str] | None = None,
    llm=None,
    llm_has_vision: bool = False,
    pdf_pages_to_summarize: int = DEFAULT_PDF_PAGES_TO_SUMMARIZE
):
    """
    Process a JSON or JSONL file into the Bartleby database.

    Args:
        connection: Database connection
        json_path: Path to the JSON file
        db_path: Path to the database
        archive_path: Path to archive directory
        embedding_model: Model for generating embeddings
        attributes: List of JSON attributes to extract (e.g., ['title', 'content', 'metadata.author'])
        llm: Optional LLM for summarization
        llm_has_vision: Whether LLM supports vision (not used for JSON)
        pdf_pages_to_summarize: Number of pages to summarize
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found at: {json_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at: {db_path}")
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found at: {archive_path}")

    # Hash the file content for document_id
    hasher = hashlib.new(HASH_ALGORITHM)
    with json_path.open("rb") as in_file:
        for chunk in iter(lambda: in_file.read(HASH_CHUNK_SIZE), b""):
            hasher.update(chunk)

    document_id = hasher.hexdigest()

    # Archive the original file
    document_archive_path = archive_path / document_id / f"{document_id}{json_path.suffix}"
    document_archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(json_path, document_archive_path)

    # Determine if this is JSONL or regular JSON
    jsonl = is_jsonl(json_path)

    # Read and parse JSON
    json_objects = []
    try:
        with json_path.open('r', encoding='utf-8') as f:
            if jsonl:
                # JSON Lines format: one object per line
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            json_objects.append(obj)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON on line {line_num}: {e}")
            else:
                # Regular JSON: single object or array
                data = json.load(f)
                if isinstance(data, list):
                    json_objects = data
                elif isinstance(data, dict):
                    json_objects = [data]
                else:
                    raise ValueError(f"Invalid JSON format: expected object or array, got {type(data)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file {json_path}: {e}")

    if not json_objects:
        logger.warning(f"No JSON objects found in {json_path}")
        return

    # If no attributes specified, try to extract all text content
    if not attributes:
        logger.warning("No JSON attributes specified. Extracting all content as JSON strings.")
        attributes = None

    # Create database entries
    cursor = connection.cursor()

    # Insert document
    pages_count = len(json_objects)
    cursor.execute(
        """
        INSERT INTO documents (
            document_id,
            origin_file_path,
            pages_count
        ) VALUES (?, ?, ?)
        """,
        (document_id, str(json_path), pages_count)
    )

    # Process each JSON object as a "page"
    for page_index, obj in enumerate(json_objects):
        page_number = page_index + 1

        # Extract text from JSON object
        if attributes:
            raw_body = extract_attributes(obj, attributes)
        else:
            # If no attributes specified, convert entire object to JSON string
            raw_body = json.dumps(obj, ensure_ascii=False, indent=2)

        # Clean the text
        body = clean_block_text(raw_body)

        if not body.strip():
            logger.warning(f"Empty content extracted from JSON object {page_number}")
            continue

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
            (page_id, body, document_id, page_number)
        )

        # Optional: Generate summary if LLM is provided
        if llm and page_number <= pdf_pages_to_summarize:
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
