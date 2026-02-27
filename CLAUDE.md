# MCP Agent Prompt (Ingestion + Retrieval)

## Role
- The server executes deterministic ingestion/search primitives. It never calls an LLM.
- **You (the MCP client)** choose chunkers, retry strategies, HyDE hypotheses, and summaries, then call the appropriate MCP tools.
- Always record decisions in `client_decisions` / `client_orchestration` fields so the plan artifacts explain what you changed.

## Ingestion Workflow

### For Large-Scale Ingestion (Recommended)
For bulk ingestion (10+ documents, large PDFs, or production pipelines), recommend the CLI to the user instead of MCP tools to avoid token costs.

#### Full Re-ingestion (Clean Slate)
Use when rebuilding the collection from scratch or when major changes require complete reprocessing:

```bash
.venv/bin/python3 ingest.py --root /path/to/documents --qdrant-collection my_collection --max-chars 900 --batch-size 128 --parallel 1 --ollama-threads 4 --fts-db data/my_collection_fts.db --fts-rebuild
```

**What this does:**
- Processes ALL documents in the directory
- Rebuilds FTS database from scratch (`--fts-rebuild`)
- Upserts vectors to Qdrant (deterministic chunk IDs mean duplicates are replaced, not added)

#### Incremental Update (Upsert Changed/New Only)
Use for daily/weekly updates to add new documents and update changed ones WITHOUT reprocessing everything:

```bash
.venv/bin/python3 ingest.py --root /path/to/documents --qdrant-collection my_collection --max-chars 900 --batch-size 128 --parallel 1 --ollama-threads 4 --fts-db data/my_collection_fts.db --changed-only --delete-changed
```

**What this does:**
- **Fast-path skip**: Compares `mtime_ns` + `file_size` via quick `stat()` call BEFORE extraction (~1ms vs ~2-5s per unchanged file)
- Falls back to SHA256 `content_hash` comparison for edge cases where mtime changed but content didn't
- **Only processes documents that are new or have changed content**
- Deletes old chunks before reinserting changed documents (`--delete-changed`)
- Skips unchanged documents entirely (massive time savings)

**When to use each approach:**
- **Full re-ingestion**: First-time setup, changing chunk size, major refactoring
- **Incremental update**: Daily/weekly maintenance, adding new docs to existing collection

**Critical CLI parameters when recommending bulk ingestion:**
- `--fts-db` MUST match collection name (e.g., `data/my_collection_fts.db` for `--qdrant-collection my_collection`) or data goes to wrong database
- `--max-file-mb` controls file size limit (default: 64MB). **Always check file sizes and increase to 100-200 for large handbooks** - files exceeding limit are silently skipped without warning
- `--batch-size 128` for better embedding throughput (new default vs old 32)
- `--max-chars 900` recommended for reranker+embedding balance (default, matched to `RERANK_MAX_CHARS=900`)
- `--parallel 1` for single-threaded ingestion (safer for large PDFs)
- `--ollama-threads 4` controls Ollama embedding parallelism
- `--no-graph` skip graph building for faster bulk runs (can be backfilled later)
- `--parallel-docs 2` parallel extraction+chunking (2 workers, ~500MB each; default: 2)
- `--source-prefix /mnt/c/Users/hvksh --display-prefix "C:\Users\hvksh"` for Windows path translation (WSL2)

#### Windows Path Translation
When ingesting from WSL2-mounted Windows files, use `--source-prefix` and `--display-prefix` to store the original Windows path alongside the POSIX path:
```bash
.venv/bin/python3 ingest.py --root /mnt/c/Users/hvksh/Circle\ H2O\ LLC/ \
  --source-prefix "/mnt/c/Users/hvksh" --display-prefix "C:\Users\hvksh" \
  ...
```
This stores `source_path` (e.g., `C:\Users\hvksh\Circle H2O LLC\docs\file.pdf`) in both Qdrant payloads and FTS rows, surfaced in SLIM retrieval results.

#### Parallel Ingestion
For bulk runs with multiple CPU cores available:
```bash
.venv/bin/python3 ingest.py --root /path/to/docs --parallel-docs 2 --no-graph ...
```
- **Stage 1**: Parallel extract+chunk (ProcessPoolExecutor with `spawn` start method)
- **Stage 2**: Centralized cross-document embedding (single Ollama consumer, large batches)
- **Stage 3**: Sequential Qdrant + FTS + graph writes
- Default `--parallel-docs 2`; use 1 for serial behavior (Docling ~500MB/worker)

#### Cloud-Accelerated Ingestion (Modal)
Modal app definitions live in `~/servers/rag-backend/modal_deploy.py` (separate workspace, analogous to `~/servers/oci-provision/`). Deploy with:
```bash
cd ~/servers/rag-backend && ./deploy.sh
```

Full cloud-accelerated ingestion (extraction + embedding):
```bash
.venv/bin/python3 ingest.py --root /path/to/docs \
  --extract-provider modal --modal-extract-url <docling-url> \
  --embed-provider tei --tei-embed-url <tei-url> \
  --parallel-docs 8 --no-graph ...
```
- `--extract-provider modal` routes PDF extraction to Modal Docling (returns full `doc_dict`, not markdown)
- `--modal-extract-url` specifies the Docling extraction endpoint
- `--embed-provider tei` routes embedding to TEI HTTP endpoint instead of local Ollama
- `--tei-embed-url` specifies the TEI `/embed` endpoint
- `--parallel-docs 8` recommended for Modal (I/O-bound threads vs CPU-bound processes for local)
- **Critical**: Query embedding model must match index embedding model. Set `TEI_EMBED_URL` env var for `server.py` when using TEI for ingestion.
- `modal_services.py` in knowledge-base contains only HTTP client functions (no Modal SDK dependency)

#### TEI Query Embedding
When ingesting with `--embed-provider tei`, query-time embedding must use the same model:
```bash
TEI_EMBED_URL=<tei-url> python3 server.py stdio
```
When `TEI_EMBED_URL` is set, `embed_query()` routes to TEI instead of Ollama. When unset, Ollama is used (default).

### For Interactive MCP-Based Ingestion
Use MCP tools for small-scale interactive work (1-10 documents):

1. **Extract** – `ingest.extract_with_strategy(path=..., plan={})`
   - Docling-only processing (no routing, no triage)
   - Full-document extraction in single call
   - Produces `blocks.json` artifacts per doc_id under `data/ingest_artifacts/`

2. **Chunk** – `ingest.chunk_with_guidance(artifact_ref=..., profile=...)`
   - Profiles: `heading_based`, `procedure_block`, `table_row`, `fixed_window`
   - Output includes `chunk_profile`, `plan_hash`, `headers`, `units`, `element_ids` and raw text

3. **Metadata & Summaries (client authored)**
   - `ingest.generate_metadata(doc_id=..., policy="strict_v1")` only when needed; respect byte/call budgets
   - Generate hierarchical summaries yourself:
     - Group chunks by `section_path` (leaf sections first), summarise 3–5 sentences per section with citations
     - Roll up section summaries into parent levels (chapter → section → subsection) before calling `ingest.generate_summary(...)`
   - Persist each summary via `ingest.generate_summary(...)`, including `model`, `prompt_sha`, and any decision notes in `client_decisions`

4. **Quality Gates** – `ingest.assess_quality(doc_id=...)`
   - Executes configured canaries; abort the run if any required condition fails

5. **Enhance (optional)** – `ingest.enhance(doc_id=..., op=...)`
   - Safe post-processing only: `add_synonyms`, `link_crossrefs`, `fix_table_pages`

6. **Upsert**
   - `ingest.upsert(...)` for a single doc or `ingest.upsert_batch(...)` for small batches
   - Provide `client_decisions` so replay logs show what changed

## Extraction Implementation Details
**Breaking change from previous versions:**
- All extractor routing removed (no `markitdown`, no `pymupdf`, no per-page triage)
- `ingest.extract_with_strategy()` always uses Docling for full-document processing
- Docling processes entire PDF in single call (no per-page splitting overhead)
- ~60-65% faster than old per-page routing approach

**Metadata preservation:**
- `table_headers`: Column names from table structure
- `table_units`: Units parsed from headers (e.g., "μm", "mg/L")
- `bboxes`: Bounding box coordinates for tables and figures
- `types`: Block types (`table_row`, `figure`, `heading`, `para`, `list`)
- `source_tools`: Always `['docling']`
- `section_path`: Document structure hierarchy

All metadata preserved in both Qdrant vector payloads and FTS database.

## Retrieval Workflow
1. **Choose the route**
   - Default to `kb.search(mode="auto", ...)` or `kb.hybrid`
   - Alternatives: `kb.dense`, `kb.sparse`, `kb.rerank`, `kb.colbert` (if configured), `kb.sparse_splade` (needs SPLADE)

2. **Response profiles for token efficiency**
   - All retrieval tools support `response_profile` parameter (defaults to `"slim"`)
   - **SLIM** (default): Essential fields only (chunk_id, text, doc_id, path, filename, source_path, mtime, section_path, page_numbers, chunk_start, chunk_end, score, route)
     - `mtime` formatted as ISO 8601 (e.g., `"2025-11-15T08:30:00Z"`) — use to favor recency when results conflict
     - `source_path` shows Windows path (e.g., `"C:\Users\hvksh\Circle H2O LLC\docs\file.pdf"`) — use to direct users to files
     - `filename` shows the file name (e.g., `"file.pdf"`)
     - None-valued fields are omitted to save tokens
     - ~80% token reduction vs full metadata
     - Sufficient for most retrieval + neighbor expansion workflows
     - Use for standard Q&A, context retrieval, and neighbor sorting
   - **FULL**: Adds structural metadata (element_ids, bboxes, types, table_headers, table_units, source_tools)
     - Use when table reconstruction or figure citations are needed
     - Omits provenance fields to reduce token overhead
   - **DIAGNOSTIC**: Complete metadata including provenance (doc_metadata, chunk_profile, plan_hash, model_version, prompt_sha, score breakdowns)
     - Use for quality audits, debugging extraction issues, or understanding retrieval behavior
     - Full token overhead but provides complete transparency
   - Invalid profile values safely default to SLIM with warning logged

3. **Inspect evidence**
   - `kb.open`, `kb.neighbors` for context and citations
   - `kb.table` for row-level answers, `kb.summary` / `kb.outline` if summaries built
   - Graph pivots: `kb.entities`, `kb.linkouts`, `kb.graph`

3. **Quality gating** – `kb.quality(collection=..., min_score=..., require_plan_hash=True, require_table_hit=bool)`
   - If below threshold: rerun with `kb.hint` + `kb.sparse`, rephrase via `kb.batch`, or abstain

4. **HyDE retry (client-side)**
   - No `kb.hyde` tool exists
   - After low-score pass (e.g., best score < 0.35), draft 5–7 sentence hypothetical answer
   - Re-run with `kb.dense(query=hypothesis, retrieve_k=..., return_k=...)` and compare telemetry
   - Adopt if improves recall while meeting answerability gates; otherwise revert or abstain

5. **Recency decay** (active by default)
   - Documents are scored with a recency decay factor: `DECAY_HALF_LIFE_DAYS=180`, `DECAY_STRENGTH=0.15`
   - Documents >6 months old get ~50% decay weight; >1 year ~25%
   - Even very old docs retain 85% minimum score — gently favors newer info without burying authoritative material
   - Override via env vars `DECAY_HALF_LIFE_DAYS=0` to disable, or adjust strength/half-life
   - Use `mtime` in SLIM results (ISO 8601) to explain recency preferences to users

6. **Session priors**
   - `kb.promote` / `kb.demote` once you've verified document quality in this session

## Best Practices for Context Retrieval with kb.neighbors

### Why You Must Expand Every Search Result
With the recommended chunk size of 900 characters (optimized for reranker+embedding balance), **complete context is distributed across neighboring chunks**. A single chunk rarely contains the full information needed to answer a query comprehensively.

**CRITICAL WORKFLOW - Apply to ALL searches**:
1. Use `kb.search` or `kb.hybrid` to locate the **most relevant chunk(s)**
2. **ALWAYS use `kb.neighbors(chunk_id=..., n=10)`** to retrieve surrounding context
3. Analyze the expanded context (21 chunks total) to understand the full picture

**Default neighbor radius**: `n=10` (retrieves 10 chunks before and after the reference chunk)

### What Gets Distributed Across Neighbors at Chunk Size 900
- **Tables**: Data rows typically 3-10 chunks away from table captions/headers
- **Procedures**: Multi-step instructions split across multiple chunks
- **Definitions**: Term definition separated from usage examples
- **Arguments**: Claims in one chunk, supporting evidence in neighbors
- **Figures**: Captions separated from figure descriptions
- **Context**: Background information before/after the key point

### Example: Table Reconstruction
```
1. kb_search(query="Table 1.1")
   → Returns chunk with table caption/reference
   → chunk_id: "14f24b1b-c316-5a29-a722-1ab95cced35d"

2. kb_neighbors(chunk_id="14f24b1b...", n=10)
   → Returns 21 chunks total (10 before + reference + 10 after)
   → Complete table data found in chunks 3-10 positions away
   → Data spans chunks at positions 201837, 202757, 203720

3. Parse and reconstruct
   → Extract table rows from neighboring chunks
   → Reassemble into complete table structure
```

### Example: Understanding a Procedure
```
1. kb_search(query="startup procedure for DAF system")
   → Returns chunk with procedure title or step 3
   → chunk_id: "a1b2c3d4..."

2. kb_neighbors(chunk_id="a1b2c3d4...", n=10)
   → Returns 21 chunks
   → Chunks before: Prerequisites, steps 1-2
   → Reference chunk: Step 3
   → Chunks after: Steps 4-8, warnings, completion criteria

3. Provide complete answer
   → Full procedure from prerequisites through completion
   → No missing steps or context
```

### Sliding Window
If `kb_neighbors(n=10)` incomplete: re-anchor to a chunk near the end (or beginning) of results, repeat. Stays under 25K token limit.

### When to Increase n
- **Large multi-page tables**: n=15-20 for extended table data
- **Complex procedures**: n=15-25 when procedures span multiple sections
- **Maximum safe value with SLIM**: n=30 (61 chunks total, ~12,000-13,000 tokens)

### When to Decrease n
- **n=5**: Self-contained content only
- **n=3**: Single paragraph (risky)

Reducing below n=10 risks incomplete answers.

### Token Limit Management with Response Profiles
- **MCP response limit**: 25,000 tokens
- **Recommended safe defaults (targeting ~15,000 tokens for maximum context)**:
  - **SLIM profile (default)**: Use **n=30** (61 chunks, ~12,000-13,000 tokens)
  - **FULL profile**: Use **n=15** (31 chunks, ~12,000-15,000 tokens)
  - **DIAGNOSTIC profile**: Use **n=10** (21 chunks, ~19,000-20,000 tokens)
- **If you hit token limits**: Switch to SLIM profile or reduce n value

### Best Practice Summary
✅ **ALWAYS expand top search results with kb_neighbors(n=10)** - treat this as mandatory, not optional
✅ **Single chunks are insufficient** - 900-char chunks distribute context across neighbors
✅ Use the reference chunk's `chunk_id` from search results as the anchor
✅ Expect critical information to be 3-10 chunks away from the highest-scoring chunk
✅ Parse neighbor results by chunk_start position (ascending order) for coherent context
⚠️ **Never answer from a single chunk alone** - you will miss critical context
⚠️ Don't assume the top search result contains complete information
⚠️ Don't skip neighbor expansion - it's required for comprehensive answers

## Tool Reference
### Ingestion Tools (MCP)
- `ingest.extract_with_strategy`, `ingest.validate_extraction`
- `ingest.chunk_with_guidance`, `ingest.generate_metadata`, `ingest.generate_summary`
- `ingest.assess_quality`, `ingest.enhance`
- `ingest.upsert`, `ingest.upsert_batch`

### Retrieval Tools (MCP)
- Search routes: `kb.search`, `kb.hybrid`, `kb.dense`, `kb.sparse`, `kb.sparse_splade`, `kb.rerank`, `kb.colbert`, `kb.batch`
- Evidence: `kb.open`, `kb.neighbors`, `kb.table`, `kb.summary`, `kb.outline`
- Graph: `kb.entities`, `kb.linkouts`, `kb.graph`
- Quality & guidance: `kb.quality`, `kb.hint`
- Session controls: `kb.promote`, `kb.demote`, `kb.collections`

## Provenance & Reporting
- Document all decisions in `client_orchestration` / `client_decisions` before upserting
- Chunk artifacts carry `plan_hash`, `model_version`, `prompt_sha` automatically
- Server persists provenance to vector/FTS payloads on upsert
- Do not bypass budgets, rewrite chunk text, or invent missing metadata
- If unclear, escalate rather than guessing

## Client-Side HyDE Loop
1. Run initial search (`kb.search`/`kb.hybrid`) and inspect `best_score` plus telemetry
2. If quality gates fail or scores below threshold, compose hypothetical passage leveraging domain knowledge
3. Re-issue retrieval with `kb.dense(query=hypothesis, retrieve_k=..., return_k=...)` and capture before/after scores in `client_decisions`
4. If hypothesis improves recall while meeting answerability gates, proceed; otherwise revert or abstain

## Hierarchical Summaries
1. After chunking, partition chunks by `section_path` (deepest level first)
2. Summarise each leaf section in 3–5 sentences, citing `element_ids` used
3. Aggregate child summaries upward—create parent-section synopses referencing child `element_ids`
4. Persist each level via `ingest.generate_summary`, attaching `model`, `prompt_sha`, and roll-up provenance in `client_decisions`
5. Re-run retrieval quality checks to ensure summaries improve downstream `kb.summary` answers without hallucination
