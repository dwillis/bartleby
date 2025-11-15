# Bartleby, the Scrivener

A powerful document processing tool that extracts text from PDFs, text files, and JSON files, generates embeddings, and optionally creates LLM-powered summaries.

---

## Background

I have found it useful to let an AI agent (e.g. Claude Code) run wild in a SQLite database containing the extracted text from a bunch of documents. I've explored giving that agent various tools to explore the SQLite database more effectively, including enabling full-text and semantic searching. This provides a toolkit _and_ agent to generate reports based on caches of PDF documents.

### The parser — `bartleby read`.

OCR-ing and parsing PDFs into a SQLite database, then paginating, summarizing, chunking, and embedding: These are valueable tasks regardless of your desire to sift through them with an AI agent. In fact, that workflow is something I use frequently, as it enables all sorts of deeper explorations of large corpora. So, I made this a standalone command.

Some gotchas with this: If you've set up an LLM to summarize pages for you, it can burn through tokens pretty fast on summarization, but you have a knob: how many pages of each PDF to summarize. You can tell it to only do the first n pages, where n can be zero.

Also, I'm using the excellent (but pre-v0) `sqlite-vec` plugin for SQLite, [here](https://github.com/asg017/sqlite-vec). There might be some instability there.

### The writer — `bartleby write`.

This is an experiment: Can an agent run RAG for you on a prepared corpus? Can I write one that does? The answer to this is complicated. This works reasonably well with paid models, such as `gpt-5-mini`, etc., but I am still getting it to reliably work with open-weights models served from Ollama (which I think is a crucial feature): `qwen3:8b` worked somewhat well; `gpt-oss:20b` failed completely. It's possible with open-weights models bigger than the sort that can run on a laptop you'd have better luck.

Be careful about token costs when using paid models. `gpt-5-nano` produces reports for pennies. `gpt-5-pro` or whatever might cost a good bit more. **The costs the tool shows are estimates!**

## Installation

### Prerequisites

Install system dependencies:

```bash
brew install tesseract
brew install uv
```

### Install Bartleby

From the project directory:

```bash
uv tool install .
```

This will install `bartleby` as a command-line tool in an isolated environment.

In dev, you might want to run:

```bash
uv tool install --editable .
```

---

## Usage

### Quick Start

1. **Configure once** (saves settings to `~/.bartleby/config.yaml`):

```bash
bartleby ready
```

This inits your `bartleby` instance, asking for everything it needs.

2. **Run anywhere**:

```bash
# Process PDFs (backward compatible)
bartleby read --pdfs path/to/pdfs --db path/to/db

# Process any supported document type
bartleby read --input path/to/documents --db path/to/db

# Process text files
bartleby read --input path/to/texts --input-type text --db path/to/db

# Process JSON files with specific attributes
bartleby read --input data.json --input-type json --json-attributes "title,content,metadata.author" --db path/to/db
```

### Options

**`bartleby ready`** - Interactive configuration wizard

**`bartleby read`** - Process documents (PDFs, text files, or JSON files)
- `--input`: Path to a file or directory containing documents (supports PDF, text, or JSON)
- `--pdfs` (deprecated): Use `--input` instead. Path to a single PDF file or directory containing PDFs
- `--db` (required): Path to database directory (created automatically if it doesn't exist)
- `--input-type`: File type - `auto` (default, detect from extension), `pdf`, `text`, or `json`
- `--json-attributes`: Comma-separated list of JSON attributes to extract (e.g., `title,content,metadata.author`)
- `--embedding-provider`: Embedding provider - `sentence-transformers` (local, default), `openai`, or `ollama`
- `--embedding-model`: Embedding model name (provider-specific, e.g., `BAAI/bge-base-en-v1.5`, `text-embedding-3-small`, `nomic-embed-text`)
- `--max-workers`: Maximum number of parallel workers (default: from config or 4)
- `--model`: LLM model name for summarization (optional)
- `--provider`: LLM provider for summarization (`anthropic` or `openai`)
- `--verbose`: Enable verbose logging

**`bartleby write`** - Write a report
- `--db` (required): Path to a database directory you've created with `bartleby read`.

---

## What `read` does

1. **Extracts text** from multiple document formats:
   - **PDFs**: Using PyMuPDF with OCR fallback (Tesseract) for image-based pages
   - **Text files**: Direct reading with automatic encoding detection (.txt, .md, .rst)
   - **JSON files**: Attribute extraction with support for nested fields (.json, .jsonl)
2. **Chunks text** intelligently using LangChain text splitters (400 chars with 50 char overlap)
3. **Generates embeddings** using sentence-transformers (BAAI/bge-base-en-v1.5, 768 dimensions)
4. **Creates summaries** (optional) for the first N pages/documents using LLMs
5. **Stores everything** in SQLite with full-text search (FTS5) and vector search (sqlite-vec)

### Supported File Types

- **PDF** (.pdf): Full PDF processing with OCR fallback
- **Text** (.txt, .md, .rst, .text, .markdown): Plain text and markdown files
- **JSON** (.json, .jsonl): Single JSON objects or JSON Lines format with attribute extraction

### JSON Attribute Extraction

When processing JSON files, you can specify which attributes to extract using the `--json-attributes` flag:

```bash
# Extract specific fields
bartleby read --input data.json --json-attributes "title,content" --db ./db

# Extract nested fields using dot notation
bartleby read --input data.json --json-attributes "title,content,metadata.author,metadata.date" --db ./db
```

For JSONL (JSON Lines) files, each line is treated as a separate "page" in the database.

### Embedding Providers

Bartleby supports multiple embedding providers for generating vector representations of text:

#### Sentence-Transformers (Local, Default)
Uses local transformer models from Hugging Face. No API key required.

```bash
# Use default model (BAAI/bge-base-en-v1.5, 768 dimensions)
bartleby read --input docs/ --db ./db

# Use a different local model
bartleby read --input docs/ --db ./db --embedding-provider sentence-transformers --embedding-model "all-MiniLM-L6-v2"
```

**Popular models**:
- `BAAI/bge-base-en-v1.5` (768 dim) - Default, good balance of quality and speed
- `all-MiniLM-L6-v2` (384 dim) - Faster, smaller
- `all-mpnet-base-v2` (768 dim) - Higher quality

#### OpenAI Embeddings
Uses OpenAI's embedding API. Requires OpenAI API key.

```bash
# Requires OPENAI_API_KEY environment variable or in config
bartleby read --input docs/ --db ./db --embedding-provider openai --embedding-model text-embedding-3-small
```

**Models**:
- `text-embedding-3-small` (1536 dim) - Cost-effective, good quality
- `text-embedding-3-large` (3072 dim) - Highest quality
- `text-embedding-ada-002` (1536 dim) - Legacy model

#### Ollama Embeddings
Uses locally-running Ollama models. Requires Ollama to be running.

```bash
# Requires Ollama running at http://localhost:11434
bartleby read --input docs/ --db ./db --embedding-provider ollama --embedding-model nomic-embed-text
```

**Models**:
- `nomic-embed-text` - General purpose embeddings
- `mxbai-embed-large` - Larger model for better quality
- Any other embedding model available in Ollama

#### Important Notes

1. **Database Compatibility**: Once a database is created with a specific embedding dimension, you must use the same embedding provider/model (or one with the same dimension) for all subsequent operations.

2. **Dimension Validation**: Bartleby automatically validates that the embedding model matches the database's expected dimension and will error if there's a mismatch.

3. **Performance**: Local models (sentence-transformers, Ollama) don't require API calls, while OpenAI requires network requests and has usage costs.

## What `write` does

- Generates a report on the given SQLite database, derived from your documents.

---

## A note on ~ vibe coding ~.

Yes, I vibe coded a lot of this codebase, though I've made some efforts to clean it up. Sorry.