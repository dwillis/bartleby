from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys

from loguru import logger
from tqdm import tqdm

from bartleby.lib.console import send
from bartleby.lib.consts import DEFAULT_MAX_WORKERS
from bartleby.lib.embedding_providers import create_embedding_provider
from bartleby.lib.utils import load_config
from bartleby.read.processor import process_pdf
from bartleby.read.text_processor import process_text
from bartleby.read.json_processor import process_json
from bartleby.read.sqlite import get_connection, get_embedding_dimension


# File type mappings
FILE_TYPE_EXTENSIONS = {
    'pdf': ['.pdf'],
    'text': ['.txt', '.md', '.rst', '.text', '.markdown'],
    'json': ['.json', '.jsonl']
}


def detect_file_type(file_path: Path) -> str:
    """
    Detect file type based on extension.

    Args:
        file_path: Path to file

    Returns:
        File type: 'pdf', 'text', 'json', or 'unknown'
    """
    ext = file_path.suffix.lower()
    for file_type, extensions in FILE_TYPE_EXTENSIONS.items():
        if ext in extensions:
            return file_type
    return 'unknown'


def _process_document_worker(args):
    """
    Worker function for multiprocessing - creates its own connection, embedding model, and LLM.
    Must be at module level to be picklable.

    Args:
        args: Tuple of (file_path, file_type, db_path, archive_path, embedding_provider_name,
                       embedding_model_name, provider, model, llm_has_vision,
                       pdf_pages_to_summarize, json_attributes, config, verbose)
    """
    file_path, file_type, db_path, archive_path, embedding_provider_name, embedding_model_name, provider, model, llm_has_vision, pdf_pages_to_summarize, json_attributes, config, verbose = args

    # Configure logging in worker process
    logger.remove()  # Remove default handler
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    # Each process gets its own connection and embedding model for isolation
    connection = get_connection(db_path)

    # Create embedding provider in the worker process
    process_embedding_provider = create_embedding_provider(
        provider=embedding_provider_name,
        model=embedding_model_name,
        api_key=config.get("openai_api_key") if embedding_provider_name == "openai" else None,
        base_url=config.get("ollama_base_url") if embedding_provider_name == "ollama" else None
    )

    # Create LLM in worker process (can't pickle LLM objects due to thread locks)
    llm = None
    if model and provider:
        # Set API keys from config if present
        if provider:
            api_key_field = f"{provider}_api_key"
            config_api_key = config.get(api_key_field)
            env_var_name = f"{provider.upper()}_API_KEY"

            if config_api_key and not os.environ.get(env_var_name):
                os.environ[env_var_name] = config_api_key

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=model)
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            base_url = config.get("ollama_base_url", "http://localhost:11434")
            llm = ChatOllama(model=model, base_url=base_url)

    try:
        # Route to appropriate processor based on file type
        if file_type == 'pdf':
            process_pdf(
                connection,
                file_path,
                db_path,
                archive_path,
                process_embedding_provider,
                llm,
                llm_has_vision,
                pdf_pages_to_summarize
            )
        elif file_type == 'text':
            process_text(
                connection,
                file_path,
                db_path,
                archive_path,
                process_embedding_provider,
                llm,
                llm_has_vision,
                pdf_pages_to_summarize
            )
        elif file_type == 'json':
            # Parse JSON attributes if provided
            attributes = None
            if json_attributes:
                attributes = [attr.strip() for attr in json_attributes.split(',')]

            process_json(
                connection,
                file_path,
                db_path,
                archive_path,
                process_embedding_provider,
                attributes,
                llm,
                llm_has_vision,
                pdf_pages_to_summarize
            )
        else:
            logger.error(f"Unsupported file type: {file_type} for {file_path}")
    finally:
        connection.close()


def main(
    db_path,
    input_path,
    input_type: str = "auto",
    json_attributes: str = None,
    embedding_provider: str = None,
    embedding_model: str = None,
    max_workers: int = None,
    model: str = None,
    provider: str = None,
    verbose: bool = False
):
    # Configure logging level
    logger.remove()  # Remove default handler
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    # Load config from ~/.bartleby/config.yaml
    config = load_config()

    # Apply config defaults (CLI args take precedence)
    if max_workers is None:
        max_workers = config.get("max_workers", DEFAULT_MAX_WORKERS)
    if model is None:
        model = config.get("model")
    if provider is None:
        provider = config.get("provider")

    # Apply embedding config defaults
    if embedding_provider is None:
        embedding_provider = config.get("embedding_provider", "sentence-transformers")
    if embedding_model is None:
        embedding_model = config.get("embedding_model")

    # Get summarization config
    from bartleby.lib.consts import DEFAULT_PDF_PAGES_TO_SUMMARIZE
    pdf_pages_to_summarize = config.get("pdf_pages_to_summarize", DEFAULT_PDF_PAGES_TO_SUMMARIZE)

    # Set API keys from config if present and not already in environment
    if provider:
        api_key_field = f"{provider}_api_key"
        config_api_key = config.get(api_key_field)
        env_var_name = f"{provider.upper()}_API_KEY"

        if config_api_key and not os.environ.get(env_var_name):
            os.environ[env_var_name] = config_api_key
            logger.debug(f"Using {provider} API key from config")

    db_path = Path(db_path)
    input_path = Path(input_path)
    archive_path = db_path.parent / "archive"
    archive_path.mkdir(parents=True, exist_ok=True)

    # Create embedding provider and validate dimension
    embedding_provider_instance = create_embedding_provider(
        provider=embedding_provider,
        model=embedding_model,
        api_key=config.get("openai_api_key") if embedding_provider == "openai" else None,
        base_url=config.get("ollama_base_url") if embedding_provider == "ollama" else None
    )

    embedding_dimension = embedding_provider_instance.get_dimension()
    embedding_model_name = embedding_provider_instance.get_model_name()

    send(f"Embedding: {embedding_provider}/{embedding_model_name} (dim={embedding_dimension})", "BIG")

    # Validate embedding dimension matches database
    db_dimension = get_embedding_dimension(db_path)
    if db_dimension and db_dimension != embedding_dimension:
        send(
            f"ERROR: Embedding dimension mismatch! Database expects {db_dimension}, "
            f"but {embedding_model_name} produces {embedding_dimension}. "
            f"Create a new database or use a compatible embedding model.",
            "ERROR"
        )
        raise ValueError(
            f"Embedding dimension mismatch: database={db_dimension}, model={embedding_dimension}"
        )

    llm = None
    llm_has_vision = False
    if model and provider:
        send(f"Loading LLM: {provider}/{model}", "BIG")
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=model)
            llm_has_vision = "claude-3" in model or "claude-4" in model
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model)
            llm_has_vision = "gpt-4" in model and "vision" in model
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            base_url = config.get("ollama_base_url", "http://localhost:11434")
            llm = ChatOllama(model=model, base_url=base_url)
            llm_has_vision = False  # Ollama typically doesn't support vision
        else:
            send(f"Unknown provider: {provider}", "WARN")

    # Collect files to process based on input type
    files_to_process = []
    if input_path.is_file():
        # Single file
        detected_type = detect_file_type(input_path) if input_type == "auto" else input_type
        if detected_type == 'unknown':
            raise ValueError(f"Unknown file type for {input_path}. Please specify --input-type")
        files_to_process = [(input_path, detected_type)]
    elif input_path.is_dir():
        # Directory: collect files based on input type
        if input_type == "auto":
            # Collect all supported file types
            for file_type, extensions in FILE_TYPE_EXTENSIONS.items():
                for ext in extensions:
                    files_to_process.extend([(f, file_type) for f in input_path.rglob(f"*{ext}")])
        else:
            # Collect only files of specified type
            extensions = FILE_TYPE_EXTENSIONS.get(input_type, [])
            for ext in extensions:
                files_to_process.extend([(f, input_type) for f in input_path.rglob(f"*{ext}")])
    else:
        raise ValueError(f"Invalid input_path: {input_path}")

    if not files_to_process:
        send(f"No files found to process in {input_path}", "WARN")
        return

    # Show summary of file types
    type_counts = {}
    for _, file_type in files_to_process:
        type_counts[file_type] = type_counts.get(file_type, 0) + 1

    type_summary = ", ".join([f"{count} {ftype}" for ftype, count in type_counts.items()])
    send(f"Processing {len(files_to_process)} document(s) ({type_summary}) with {max_workers} workers", "BIG")

    # Use ProcessPoolExecutor instead of ThreadPoolExecutor for embedding provider thread-safety
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments for each worker (must be picklable - can't pass provider objects)
        worker_args = [
            (file_path, file_type, db_path, archive_path, embedding_provider, embedding_model,
             provider, model, llm_has_vision, pdf_pages_to_summarize, json_attributes, config, verbose)
            for file_path, file_type in files_to_process
        ]

        futures = {
            executor.submit(_process_document_worker, args): args[0]
            for args in worker_args
        }

        with tqdm(total=len(files_to_process), desc="Processing documents", unit="doc") as pbar:
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    future.result()
                    logger.debug(f"Successfully processed: {file_path}")
                except Exception as e:
                    send(f"Failed to process {file_path.name}: {e}", "ERROR")
                finally:
                    pbar.update(1)

    send("Processing complete!", "COMPLETE")