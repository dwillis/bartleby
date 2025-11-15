import argparse
import sys
from pathlib import Path

from loguru import logger

from bartleby.lib.consts import DEFAULT_MAX_WORKERS


def main():
    parser = argparse.ArgumentParser(
        prog="bartleby",
        description="Bartleby, the Scrivener - A PDF processor that might refuse."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ready command
    ready_parser = subparsers.add_parser("ready", help="Configure Bartleby settings")

    # Read command
    read_parser = subparsers.add_parser("read", help="Process documents (PDF, text, or JSON)")
    read_parser.add_argument(
        "--input",
        required=False,
        type=str,
        help="Path to a file or directory containing documents (PDF, text, or JSON)"
    )
    read_parser.add_argument(
        "--pdfs",
        required=False,
        type=str,
        help="(Deprecated: use --input) Path to a PDF file or directory containing PDFs"
    )
    read_parser.add_argument(
        "--db",
        required=True,
        type=str,
        help="Path to the database directory (will be created if it doesn't exist)"
    )
    read_parser.add_argument(
        "--input-type",
        type=str,
        choices=["auto", "pdf", "text", "json"],
        default="auto",
        help="Input file type: auto (detect from extension), pdf, text, or json (default: auto)"
    )
    read_parser.add_argument(
        "--json-attributes",
        type=str,
        default=None,
        help="Comma-separated list of JSON attributes to extract (e.g., 'title,content,metadata.author')"
    )
    read_parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=f"Maximum number of worker threads (default: from config or {DEFAULT_MAX_WORKERS})"
    )
    read_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (e.g., claude-3-5-sonnet-20241022, default: from config)"
    )
    read_parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "openai"],
        default=None,
        help="LLM provider (anthropic or openai, default: from config)"
    )
    read_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (show DEBUG messages)"
    )

    # Write command
    write_parser = subparsers.add_parser("write", help="Research agent for document investigation")
    write_parser.add_argument(
        "--db",
        required=True,
        type=str,
        help="Path to the database directory (created by 'bartleby read')"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "ready":
        from bartleby.lib.console import send
        from bartleby.ready.main import main as ready_main

        send(message_type="SPLASH")
        ready_main()

    elif args.command == "read":
        from bartleby.lib.console import send
        from bartleby.read.main import main as read_main

        send(message_type="SPLASH")

        # Handle backward compatibility: --pdfs is deprecated, use --input
        input_path = args.input or args.pdfs
        if not input_path:
            send("Error: Either --input or --pdfs is required", "ERROR")
            sys.exit(1)

        db_dir = Path(args.db)
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / "bartleby.db"

        # Create database if it doesn't exist
        if not db_path.exists():
            send(f"Creating database at {db_path}", "BIG")
            from bartleby.read.sqlite import create_db
            create_db(db_dir)

        read_main(
            db_path=db_path,
            input_path=input_path,
            input_type=args.input_type,
            json_attributes=args.json_attributes,
            max_workers=args.max_workers,
            model=args.model,
            provider=args.provider,
            verbose=args.verbose
        )

    elif args.command == "write":
        from bartleby.write.main import main as write_main

        db_dir = Path(args.db)
        db_path = db_dir / "bartleby.db"

        write_main(db_path=db_path)


if __name__ == "__main__":
    main()
