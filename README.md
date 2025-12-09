# Semantic Search MCP Server

A production-grade **Model Context Protocol (MCP)** server that puts a state-of-the-art MCP client (Claude Desktop/Code/Codex CLI) in the loop for both retrieval and ingestion. Agents plan with atomic MCP tools—deciding the extractor, chunker, reranker route, HyDE retries, graph hops, and metadata budget calls—while the server guarantees determinism, provenance, and thin-index security. Everything runs on local, open-source components, so embeddings and reranking stay **zero-cost** beyond your Claude subscription.

## 🌟 Features

- **Zero-Cost Embeddings & Reranking**: Ollama-powered embeddings and a local TEI cross-encoder keep every document and query free of per-token fees.
- **Structure-Aware Ingestion**: Docling processes entire PDFs in single calls for ~60-65% faster performance, extracting tables, figures, headings with bboxes, section breadcrumbs, and element IDs. Full-document processing eliminates per-page overhead while preserving all semantic structure.
- **Agent-Directed Hybrid Retrieval**: Auto mode chooses among dense, hybrid, sparse, and rerank routes; when scores are low it returns an abstain so the MCP client can decide whether to run HyDE or try alternate queries. Every primitive stays exposed for manual overrides.
- **Multi-Collection Support (manual)**: Organize documents into separate knowledge bases by adding entries to `NOMIC_KB_SCOPES`; the default configuration ships with a single collection (`daf_kb`).
- **Incremental & Deterministic Ingestion**: Smart update detection only reprocesses changed documents. Deterministic chunk IDs enable automatic upsert-based updates without manual cleanup.
- **Graph & Entities (lightweight)**: Ingestion extracts equipment/chemical/parameter entities and links them back to chunks so agents can pivot via `kb.entities`/`kb.linkouts`. (Full semantic relationship extraction is still on the roadmap.)
- **Operational Tooling**: `scripts/manage_cache.py` and `scripts/manage_collections.sh` help purge caches or manage Qdrant collections; GPU knobs (`DOCLING_DEVICE`, `DOCLING_BATCH_SIZE`) keep heavy PDFs flowing.
- **Canary QA Framework**: `ingest.assess_quality` can run user-supplied canary queries (`config/canaries/`) and report warnings before documents reach production.
- **Rich Provenance Tracking**: Every chunk carries `plan_hash`, `model_version`, `prompt_sha`, and `client_decisions` for full auditability. Search results include `page_numbers`, `section_path`, `element_ids`, `table_headers`, `table_units`, and `bboxes` for precise citations and source verification. All metadata preserved in both Qdrant vector payloads and FTS database.
- **MCP-Controlled Upserts**: `ingest.upsert`, `ingest.upsert_batch`, and `ingest.corpus_upsert` let agents push chunk artifacts straight into Qdrant + FTS without leaving the MCP workflow.
- **Client-Orchestrated Summaries & HyDE**: LLM clients contribute section summaries via `ingest.generate_summary` and generate context-aware HyDE hypotheses locally before re-querying `kb.dense`, with every decision recorded in plan provenance.
- **Observability & Guardrails**: Search logs include hashed subject IDs, stage-level timings, and top hits; `eval.py` runs gold sets with recall/nDCG/latency thresholds for CI gating.
- **MCP Integration**: Works seamlessly with Claude Desktop, Claude Code, Codex CLI, and any MCP-compliant client.
- **MCP Best Practices**: All 31 tools include standardized annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`), Pydantic input validation, pagination metadata, and comprehensive docstrings.
- **Agent Playbooks**: Ready-to-run "retrieve → assess → refine" workflows for Claude and other MCP clients are documented in [`CLAUDE.md`](CLAUDE.md) and [`.codex/AGENTS.md`](.codex/AGENTS.md).

> **Experimental / optional features** such as SPLADE sparse expansion, ColBERT late interaction, automatic summaries/outlines, HyDE query expansion, and enforced canary QA require additional services or configuration. See the status table below for details.

### Feature Status at a Glance

| Feature | Status | Notes |
| ------- | ------ | ----- |
| Core dense / hybrid / sparse retrieval | ✅ Working | `kb.search`, `kb.dense`, `kb.hybrid`, `kb.sparse`, `kb.batch` |
| Entity extraction + graph link-outs | ✅ Working | `kb.entities`, `kb.linkouts`, `kb.graph` (entity→chunk relationships) |
| Table retrieval | ✅ Working (data-dependent) | Requires tables extracted during ingestion |
| MCP tool annotations | ✅ Working | All 31 tools have `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint` |
| Pydantic input validation | ✅ Working | `models.py` provides validated input models with field constraints |
| Pagination metadata | ✅ Working | Search results include `count`, `retrieve_k`, `return_k`, `has_more` |
| Async HTTP client | ✅ Working | Embeddings and reranker use `httpx.AsyncClient` for better concurrency |
| Canary QA | ⚠️ Requires user config | Default `config/canaries/*.json` are placeholders; add queries to enforce gates |
| Document summaries / outlines | ⚠️ Client-provided | `ingest.generate_summary` stores semantic summaries with provenance; outlines still require building the heading index |
| HyDE retry | ⚠️ Client-generated | There is no server tool; draft the hypothesis in your MCP client, then re-query `kb.dense` / `kb.search` with it |
| MCP upsert pipeline | ✅ Working | `ingest.upsert`, `ingest.upsert_batch`, `ingest.corpus_upsert` |
| SPLADE sparse expansion | 💤 Planned | `--sparse-expander` hooks are present but no SPLADE model is bundled by default |
| ColBERT late interaction | 💤 Planned | Requires an external ColBERT service; disabled when `COLBERT_URL` is unset |

`✅` working today · `⚠️` requires extra configuration or build steps · `💤` planned / not yet implemented

## 🤖 MCP-First Architecture

Conventional RAG systems hide retrieval behind a monolithic API. This server embraces the MCP client as a planner:

- **Ingestion as a Toolchain** – `ingest.extract_with_strategy` processes entire PDFs with Docling in single calls (no per-page routing); `ingest.chunk_with_guidance` switches between enumerated chunkers (`heading_based`, `procedure_block`, `table_row`); `ingest.generate_metadata` enforces byte budgets and prompt hashes. Every step returns artifacts and plan hashes for replayable ingestion.
- **Retrieval as Composable Primitives** – `kb.sparse`, `kb.dense`, `kb.hybrid`, `kb.rerank`, `kb.hint`, `kb.table_lookup`, `kb.entities`, `kb.linkouts`, `kb.batch`, `kb.quality`, and `kb.promote/demote`. Use these with client-authored HyDE retries and planner heuristics to branch, retry, or fuse strategies mid-conversation.
- **Self-Critique with Insight** – Results surface full score vectors (`bm25`, `dense`, `rrf`, `rerank`, `prior`, `decay`) and `why` annotations (matched aliases, headers, table clues), letting the agent reason about confidence before presenting an answer.

Because Claude (or any MCP client) stays in the driver seat, you get agentic retrieval and deterministic ingestion without surrendering provenance or security.

## 🏗️ Architecture

```
┌──────────────────────────────┐
│  MCP Clients                 │
│  Claude Desktop / Code / CLI │
└────────────┬─────────────────┘
             │ MCP Protocol (stdio)
         ↓
┌────────────────────────────────────────┐
│         server.py (FastMCP)            │
│  ┌──────────────────────────────────┐  │
│  │  Search Modes:                   │  │
│  │  • Semantic (vector only)        │  │
│  │  • Rerank (vector + reranker)    │  │
│  │  • Hybrid (RRF + rerank)         │  │
│  └──────────────────────────────────┘  │
└─────┬──────────────┬────────────┬──────┘
      │              │            │
      ↓              ↓            ↓
┌──────────┐  ┌──────────┐  ┌────────────┐
│  Qdrant  │  │ SQLite   │  │  Ollama    │
│  Vector  │  │  FTS5    │  │ Embeddings │
│   DB     │  │  (BM25)  │  └────────────┘
└──────────┘  └──────────┘
      │
      ↓
┌──────────────┐
│ TEI Reranker │
│ (Hugging     │
│  Face)       │
└──────────────┘
```

**Processing Pipeline**:
```
Documents → Extract (Docling full-document) → Chunk → Graph & Summaries → Embed → [Qdrant + SQLite FTS] → Planner → Search → Rerank → Self-Critique → Results
```

## 📋 Use Cases

- **Engineering Documentation**: Search technical manuals, specifications, and handbooks (e.g., water treatment engineering, chemical engineering)
- **Legal Research**: Query case law, contracts, and regulatory documents
- **Medical Literature**: Search research papers, clinical guidelines, and medical textbooks
- **Academic Research**: Build searchable libraries of papers and books
- **Corporate Knowledge Bases**: Make internal documentation and reports searchable
- **Personal Research**: Organize and query your personal document collection

## 🚀 Quick Start

### Prerequisites

- Docker Desktop (for Qdrant + TEI reranker)
- Ollama (for embeddings)
- Python 3.9+
- Optional but recommended: set `HF_HOME` to a writable folder (e.g., `export HF_HOME="$PWD/.cache/hf"`) so Docling can cache layout models when triage routes a page to structured extraction.

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/knowledge-base-mcp.git
cd knowledge-base-mcp
```

2. **Install Ollama** (if not already installed):
   - Visit [ollama.com](https://ollama.com) and download for your platform
   - Pull the embedding model:
   ```bash
   ollama pull snowflake-arctic-embed:xs
   ```

3. **Start Docker services** (Qdrant + TEI Reranker):
```bash
docker-compose up -d
```

4. **Set up Python environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

5. **Configure your MCP client**:

Choose the configuration for your MCP client:

**For Claude Code**:
```bash
cp .mcp.json.example .mcp.json
# Edit .mcp.json and update the Python venv path
```

**For Claude Desktop**:
- Copy `claude_desktop_config.json.example` contents to:
  - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
  - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- Update the paths to match your system (use `wsl` command on Windows)

**For Codex CLI**:
- Use `.codex/config.toml.example` as a template

6. **Ingest your first documents**:
```bash
.venv/bin/python3 ingest.py \
  --root /path/to/your/documents \
  --qdrant-collection my_docs \
  --max-chars 700 \
  --batch-size 128 \
  --parallel 1 \
  --ollama-threads 4 \
  --fts-db data/my_docs_fts.db \
  --fts-rebuild \
  --max-file-mb 100
```

**Important CLI Parameters**:
- `--max-file-mb 100`: Maximum file size to process (default: 64MB). Increase for large handbooks/textbooks.
- `--fts-db`: **MUST** match collection name (e.g., `data/my_docs_fts.db` for `--qdrant-collection my_docs`)
- `--max-chars 700`: Recommended chunk size for reranker compatibility (old default: 1800)
- `--batch-size 128`: Embedding batch size (new default: 128 vs old default: 32)
- `--fts-rebuild`: Rebuild FTS database from scratch (omit for incremental updates)
- Document timeouts have been **removed** - all documents process to completion regardless of size

7. **Test the search**:
```bash
python validate_search.py \
  --query "your search query" \
  --collection my_docs \
  --mode hybrid
```

8. **Optionally run the evaluation harness** (fails CI if thresholds are missed):
```bash
python eval.py \
  --gold eval/gold_sets/my_docs.jsonl \
  --mode auto \
  --fts-db data/my_docs_fts.db \
  --min-ndcg 0.85 --min-recall 0.80 --max-latency 3000
```

9. **Optional:** add canary QA queries to `config/canaries/` so `ingest.assess_quality` can enforce pass/fail gates instead of only reporting warnings.

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.

## 🔍 Search Modes

### Semantic Search (`mode="semantic"`)
Pure vector similarity search using dense embeddings.

**Best for**: Conceptual queries, finding related content even without exact keyword matches

**Example**: "How do biological systems remove nitrogen?" will find relevant content even if it uses terms like "nitrification" or "denitrification"

### Rerank Search (`mode="rerank"`, default)
Vector retrieval followed by cross-encoder reranking.

**Best for**: Most use cases - good balance of speed and accuracy

**Example**: Standard searches where you want better precision than pure vector search

### Hybrid Search (`mode="hybrid"`)
Combines vector search + BM25 lexical search using RRF, then reranks.

**Best for**: Complex queries with both conceptual and specific keyword requirements

**Example**: "stainless steel 316L corrosion in chloride environments" benefits from both semantic understanding and exact term matching

### Sparse Search (`mode="sparse"`)
Runs alias-aware BM25 only, useful for short keyword queries or as a fallback when semantic routes miss exact terminology.

### Auto Planner (`mode="auto"`)
Default. Heuristics pick among semantic, hybrid, rerank, and sparse routes. When the top score falls below `ANSWERABILITY_THRESHOLD`, the server abstains and returns telemetry so the MCP client (Claude, etc.) can decide whether to generate a HyDE hypothesis and retry with `kb.dense` or `kb.sparse`.

### Additional MCP Tools

- `kb.collections` – list configured collection slugs and their backing indices.
- `kb.open(collection="slug", chunk_id=...)` – fetch a chunk by `chunk_id` or `element_id`, optionally slicing by char offsets.
- `kb.neighbors(collection="slug", chunk_id=...)` – pull FTS neighbors around a chunk for more context.
- `kb.summary(collection="slug", topic=...)` – retrieve lightweight section summaries (RAPTOR-style) built during ingest.
- `kb.graph(node_id=...)` – inspect the lightweight graph (doc → section → chunk → entity) generated from structured metadata.

## 📊 Search Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | — | Search query text |
| `mode` | `rerank` | `semantic`, `rerank`, `hybrid`, `sparse`, or `auto` |
| `top_k` | `8` | Final results returned (1–100) |
| `retrieve_k` | `24` | Initial candidate pool (1–256) |
| `return_k` | `8` | Post-rerank results (`≤ retrieve_k`) |
| `n` (for kb.neighbors) | `10` (recommended) | Neighbor radius for context expansion - MANDATORY for comprehensive answers |

### Parameter Tuning Guide

| Scenario | retrieve_k | return_k | top_k | mode |
|----------|-----------|----------|-------|------|
| **Quick search** | 12 | 8 | 5 | rerank |
| **Comprehensive** | 48 | 16 | 10 | hybrid |
| **High precision** | 24 | 12 | 5 | hybrid |
| **Exploratory** | 32 | 12 | 8 | semantic |

### Configuration knobs

Set via environment variables (or CLI flags when available):

| Environment variable | Purpose |
|----------------------|---------|
| `MIX_W_BM25`, `MIX_W_DENSE`, `MIX_W_RERANK` | Adjust blend between lexical, dense, and rerank signals. |
| `HF_HOME` | Hugging Face cache directory used by Docling models (default `.cache/hf`). |
| `GRAPH_DB_PATH`, `SUMMARY_DB_PATH` | Override lightweight graph and summary storage locations. |
| `ANSWERABILITY_THRESHOLD` | Minimum score required for auto mode to respond; lower scores return an abstain for the client to handle (e.g., run HyDE or rephrase). |
| **Context Retrieval** | **CRITICAL: Always use kb.neighbors(n=10) after search** - single chunks are insufficient at chunk size 700. |

**Note**: Document timeouts have been removed. All documents process to completion regardless of size.

## 📚 Usage

See [USAGE.md](USAGE.md) for comprehensive documentation including:
- Ingestion parameters and examples
- Multi-collection setup
- Advanced search features (neighbor expansion, time decay)
- Incremental ingestion patterns
- Performance tuning

For upcoming improvements, check [ROADMAP.md](ROADMAP.md).

## 📘 Further Reading

- [`CLAUDE.md`](CLAUDE.md) – MCP agent prompt for Claude Desktop/Code with ingestion and retrieval workflows.
- [`.codex/AGENTS.md`](.codex/AGENTS.md) – MCP agent prompt for Codex CLI.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) – deep dive into RRF, reranking, and chunking design choices.
- [`ROADMAP.md`](ROADMAP.md) – planned features including SPLADE sparse expansion and ColBERT late interaction.

## 🏛️ Architecture Details

See [ARCHITECTURE.md](ARCHITECTURE.md) for deep dive into:
- Reciprocal Rank Fusion (RRF) algorithm
- Cross-encoder reranking strategy
- Neighbor context expansion
- Embedding model selection rationale
- Chunking strategy

## 🛠️ Configuration

### Environment Variables

All configuration can be customized via environment variables. See [.env.example](.env.example) for full documentation.

Key variables:
- `OLLAMA_MODEL`: Embedding model (default: `snowflake-arctic-embed:xs`)
- `QDRANT_URL`: Qdrant server (default: `http://localhost:6333`)
- `TEI_RERANK_URL`: Reranker endpoint (default: `http://localhost:8087`)
- `HYBRID_RRF_K`: RRF parameter (default: 60)
- `NEIGHBOR_CHUNKS`: Context expansion (default: 1)

### Multi-Collection Setup

Configure multiple knowledge bases with independent search tools:

```json
{
  "NOMIC_KB_SCOPES": "{
    \"technical_docs\": {
      \"collection\": \"engineering_kb\",
      \"title\": \"Engineering Documentation\"
    },
    \"legal_docs\": {
      \"collection\": \"legal_kb\",
      \"title\": \"Legal Research\"
    }
  }"
}
```

This creates two MCP tools: `search_technical_docs` and `search_legal_docs`.

## 🔧 Troubleshooting

**Services not starting?**
```bash
# Check Docker services
docker-compose ps

# Check Ollama
curl http://localhost:11434/api/tags

# Check Qdrant
curl http://localhost:6333/collections
```

**Embeddings failing?**
- Ensure Ollama model is pulled: `ollama list`
- Check Ollama is running: `ollama serve` (usually auto-starts)
- Try reducing batch size: Add `--embed-batch-size 16` to ingest command

**Search returning no results?**
- Verify collection name matches ingestion
- Check Qdrant collection exists: `curl http://localhost:6333/collections/{collection_name}`
- Confirm FTS database path is correct

See [FAQ.md](FAQ.md) for more common issues.

## 📈 Performance

**Hardware recommendations**:
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Optimal**: 16GB RAM, 8+ CPU cores

**Benchmarks** (approximate, varies by hardware):
- Ingestion: 5-10 pages/second (with Ollama embeddings)
- Search latency: 100-300ms (hybrid mode with reranking)
- Storage: ~2-3KB per chunk (vector + payload)

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
- [Qdrant](https://qdrant.tech) - Vector database
- [Ollama](https://ollama.com) - Local LLM and embeddings
- [Hugging Face TEI](https://github.com/huggingface/text-embeddings-inference) - Reranking
- [MarkItDown](https://github.com/microsoft/markitdown) - Document extraction
- [Docling](https://github.com/DS4SD/docling) - High-fidelity PDF processing

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/knowledge-base-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/knowledge-base-mcp/discussions)
- **Documentation**: See docs in this repository

---

**Why local embeddings and reranking?**
- 💰 **Zero Additional Cost**: No per-document embedding fees, no per-query reranking charges - only Claude subscription
- 📈 **Unlimited Scale**: Ingest and search unlimited documents without incremental costs
- ⚡ **Fast**: Local search with <300ms latency - no API roundtrips for embeddings or reranking
- 🎯 **Control**: Full customization of embedding models, search parameters, and chunking strategy
