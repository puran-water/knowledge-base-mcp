# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **MCP Tool Annotations**: All 31 tools now include standardized annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) for MCP client compatibility
- **Pydantic Input Models**: New `models.py` with Pydantic v2 input validation models for all tools, providing field constraints (min/max length, numeric bounds) and better error messages
- **Pagination Metadata**: All search tools now return consistent pagination fields (`count`, `retrieve_k`, `return_k`, `has_more`, `top_k` where applicable)
- **Comprehensive Docstrings**: Key tools now include detailed docstrings with Args/Returns sections following MCP best practices
- **httpx Async Client**: Migrated embedding and reranker HTTP calls from synchronous `requests` to async `httpx.AsyncClient` for better concurrency

### Changed (Breaking)
- **Docling-only extraction**: Removed per-page routing and triage logic
- **Removed `--extractor` CLI flag**: Docling is now the only extractor
- **Performance improvement**: ~60-65% faster ingestion (9.5h → 3.3-4.8h for 747-page corpus)
- **Default batch size**: Increased from 32 to 128 for better embedding throughput
- **Recommended chunk size**: Now 700 chars (was 1800) for reranker compatibility
- **kb.search signature**: Now accepts Pydantic `KBSearchInput` model for validated input

### Dependencies
- Added `httpx>=0.25.0` for async HTTP client support

### Documentation
- **CRITICAL: Mandatory neighbor expansion for all searches**: Documented that `kb.neighbors(n=10)` is MANDATORY after every search, not optional (CLAUDE.md, AGENTS.md, USAGE.md, FAQ.md)
- **Context distribution at chunk size 700**: Single chunks are insufficient - context (tables, procedures, definitions, evidence) distributed across 3-10 neighboring chunks
- **Neighbor search best practices**: Recommended `n=10` default captures distributed context while staying under 25,000 token MCP response limit
- **Multiple practical examples**: Table reconstruction, multi-step procedures, and conceptual queries all demonstrating kb_search → kb_neighbors(n=10) → comprehensive answer workflow
- **Token limit guidance**: n=10 returns ~19-20K tokens (safe), n=15 exceeds 25K limit and fails
- **Provenance enhancements**: Expanded documentation of rich metadata tracking (plan_hash, model_version, prompt_sha, client_decisions, table_headers, table_units, bboxes, page_numbers, section_path, element_ids)

### Added
- Initial public release
- Hybrid semantic search with three modes (semantic, rerank, hybrid)
- Multi-collection support via MCP scopes
- Incremental ingestion with change detection and deterministic chunk IDs
- Neighbor context expansion
- Time decay scoring for recency boost
- Comprehensive documentation (README, INSTALLATION, USAGE, ARCHITECTURE, FAQ)
- Example scripts for common use cases
- Docker Compose setup for Qdrant and TEI reranker
- Support for multiple document formats (PDF, DOCX, TXT, HTML, etc.)
- Production ingestion script with batch processing
- SQLite FTS5 lexical search integration
- RRF (Reciprocal Rank Fusion) for hybrid search
- Collection management script (`scripts/manage_collections.sh`)

### Technical Details
- FastMCP-based MCP server
- Qdrant vector database with HNSW indexing
- Ollama embeddings (snowflake-arctic-embed:xs default)
- Hugging Face TEI cross-encoder reranker
- Docling full-document PDF processing (semantic structure extraction)
- Character-based sliding window chunking
- Configurable search parameters (retrieve_k, return_k, top_k)
- Full metadata preservation (table headers, units, bboxes, provenance)

### Fixed
- Hardened ingestion MCP tools against path traversal by validating chunk and metadata artifact inputs with `_validate_artifact_path`.
- Updated agent prompts and documentation to reflect client-authored HyDE retries and hierarchical summary workflows.

### Performance
- Eliminates ~125 min PyMuPDF triage overhead per 747 pages
- Eliminates ~25-38 min temp PDF creation overhead
- All semantic structure preserved while achieving 60-65% speed improvement

## [1.0.0] - TBD

Initial public release.
