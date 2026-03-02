# RLM + Knowledgebase Integration Plan

> Future improvement: Integrate RLM (Recursive Language Model) with the knowledgebase MCP server to enable recursive query decomposition with domain-specific retrieval.

## Overview

Integrate RLM (Recursive Language Model) from `rand/rlm-claude-code` with the existing RAG knowledgebase MCP server at `~/servers/knowledge-base`. This enables RLM's recursive decomposition to leverage domain-specific retrieval.

**Installation**: Clone locally to `~/rlm-claude-code`
**Scope**: Full integration (REPL helpers + hooks + memory pattern learning)

## Target Architecture

```
Claude Code
    |
    +-- RLM Plugin (hooks)
    |       |
    |       +-- REPL Environment
    |       |       +-- kb_search()      # NEW
    |       |       +-- kb_neighbors()   # NEW
    |       |       +-- kb_batch()       # NEW
    |       |       +-- llm(), peek()    # existing
    |       |
    |       +-- Memory Store (patterns)
    |
    +-- RAG KB MCP Server (existing, unchanged)
            +-- kb.search, kb.hybrid, kb.neighbors, etc.
```

---

## Phase 1: Install RLM

```bash
cd ~/
git clone https://github.com/rand/rlm-claude-code.git
cd rlm-claude-code
uv sync --all-extras
```

### Configure Environment
```bash
# Create .env in RLM directory
cat > ~/rlm-claude-code/.env << 'EOF'
ANTHROPIC_API_KEY=<your-key>
EOF
```

---

## Phase 2: Create MCP Client Wrapper

**Create**: `~/rlm-claude-code/src/kb_client.py`

Use the official MCP Python client (`mcp.client.stdio` or `fastmcp.client.Client`) - do NOT hand-roll JSON-RPC.

### Critical Configuration (from Codex review):
```python
# Load config from knowledgebase/.mcp.json
# MUST set cwd and env correctly or FTS paths will break

import json
from pathlib import Path

KB_ROOT = Path("/home/hvksh/servers/knowledge-base")
MCP_CONFIG = json.loads((KB_ROOT / ".mcp.json").read_text())
ENV = MCP_CONFIG["mcpServers"]["knowledge-base"]["env"]

# Server parameters
server_params = StdioServerParameters(
    command=str(KB_ROOT / ".venv/bin/python"),
    args=[str(KB_ROOT / "server.py"), "stdio"],
    env=ENV,  # MUST include NOMIC_KB_SCOPES
    cwd=str(KB_ROOT),  # CRITICAL: relative paths depend on this
)
```

### Return Shape Normalization:
| Tool | Returns |
|------|---------|
| `kb.search` | List of dicts |
| `kb.hybrid` | Dict with `rows` key |
| `kb.neighbors` | List of dicts |
| `kb.batch` | Dict with `results` key |

**Constraints**:
- `top_k` max is 100 (not 256) - clamp in client
- Use `kb.batch` for decomposed queries (single call, not N calls)
- Fallback to `mode="sparse"` when Ollama/TEI unavailable

---

## Phase 3: Extend REPL with KB Helpers

**Modify**: `~/rlm-claude-code/src/repl_environment.py`

Add KB helper functions to REPL globals (minimal set first):

| Function | Purpose |
|----------|---------|
| `kb_search(query, collection, mode="auto", top_k=8)` | Primary search (auto-routes) |
| `kb_neighbors(chunk_id, n=10)` | Context expansion (CRITICAL) |
| `kb_batch(queries, collection)` | Multi-query decomposition |

### Example REPL Usage
```python
# Simple search with auto-routing
results = kb_search("DAF startup procedure", collection="daf")

# ALWAYS expand context for top hit
if results:
    full_context = kb_neighbors(results[0]["chunk_id"], n=10)

# Decomposed multi-query (uses kb.batch - single call)
sub_queries = ["DAF performance high TSS", "clarifier performance high TSS"]
batch_results = kb_batch(sub_queries, collection="daf")
```

### Security Gate (per Codex):
Only inject KB helpers when RLM tool-access-level includes "read" or "full".

---

## Phase 4: Update RLM Configuration

**Modify**: `~/.claude/rlm-config.json`

Add kb_integration section:

```json
{
  "activation": {
    "mode": "intelligent",
    "complexity_score_threshold": 0.6
  },
  "kb_integration": {
    "enabled": true,
    "server_path": "/home/hvksh/servers/knowledge-base",
    "collections": {
      "daf": "daf_kb",
      "clarifier": "clarifier_kb",
      "biogas": "biogas_treatment_kb",
      "aerobic": "aerobic_treatment_kb",
      "ro": "ro_kb",
      "ix": "ix_kb",
      "dewatering": "dewatering_kb",
      "corrosion": "corrosion_kb",
      "evaporation": "evaporation_kb",
      "filtration": "media_filtration_kb",
      "sales": "sales_kb"
    },
    "auto_expand_neighbors": true,
    "neighbor_radius": 10,
    "min_score_threshold": 0.3
  }
}
```

---

## Phase 5: Hook Integration

**Modify**: `~/rlm-claude-code/scripts/check_complexity.py`

Add KB-aware complexity detection:
- Detect domain terms (daf, clarifier, wastewater, treatment, etc.)
- Auto-activate RLM for complex domain queries
- Pass collection hints to orchestrator

---

## Phase 6: Memory Pattern Learning

**Modify**: `~/rlm-claude-code/src/memory_store.py`

Log successful KB queries to memory:
- Query text, collection, mode used
- Result count and top score
- Decomposition strategy if used
- Enables learning which strategies work for which query types

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `~/rlm-claude-code/src/kb_client.py` | CREATE | MCP client wrapper |
| `~/rlm-claude-code/src/repl_environment.py` | MODIFY | Add kb_* helpers |
| `~/.claude/rlm-config.json` | MODIFY | Add kb_integration config |
| `~/rlm-claude-code/scripts/check_complexity.py` | MODIFY | Domain-aware activation |
| `~/rlm-claude-code/src/memory_store.py` | MODIFY | Pattern logging |

---

## Codex Review Findings

### Technical Gaps Addressed:
1. **Working directory**: Set `cwd=/home/hvksh/servers/knowledge-base` when spawning server
2. **Environment variables**: Load from `.mcp.json`, especially `NOMIC_KB_SCOPES`
3. **Use `kb.batch`**: For decomposed queries, call once instead of N times
4. **Sparse fallback**: When embedding services are down, use `mode="sparse"`
5. **Server lifecycle**: Spawn once per session, not per call

### Risk Mitigations:
- Startup self-check: Call `kb.collections` and fail fast if wrong config
- Expand neighbors only for top 1-2 hits (not all) to avoid latency explosions
- Use `response_profile="slim"` by default
- Gate KB helpers behind RLM's tool-access-level when appropriate

### Alternative Approach (Simpler):
If full integration is complex, start with **minimal integration**:
- Only add `kb_search(mode="auto")` + `kb_neighbors`
- Skip `kb_hybrid` wrapper (auto mode handles routing)
- Use `kb.batch` for multi-query decomposition

---

## Prerequisites Checklist

- [ ] Knowledgebase services running (Qdrant, Ollama, TEI reranker)
- [ ] Python 3.12+ installed
- [ ] uv package manager installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [ ] Anthropic API key available

---

## Verification Steps

1. **Test KB client standalone**:
   ```bash
   cd ~/rlm-claude-code
   uv run python -c "from src.kb_client import KnowledgeBaseClient; kb = KnowledgeBaseClient(); print(kb.search('DAF startup'))"
   ```

2. **Test REPL with KB helpers**:
   ```bash
   # In Claude Code session with RLM active
   # REPL should have kb_search, kb_neighbors available
   ```

3. **Test end-to-end flow**:
   - Ask: "What is the startup procedure for the DAF system?"
   - RLM should activate, use kb_search, expand with kb_neighbors
   - Response should cite specific chunks from daf_kb

---

## Implementation Order

1. Clone RLM repo and install dependencies
2. Create kb_client.py with basic search/neighbors
3. Add kb_* helpers to REPL environment
4. Update rlm-config.json with kb_integration
5. Test basic flow
6. Add hook modifications for domain detection
7. Add memory pattern logging
8. End-to-end testing with domain queries

---

## RLM vs RAG: Complementary Systems

**RLM (Recursive Language Model)**:
- Runtime reasoning/decomposition
- Dynamic chunking at query time
- Persistent cross-session memory
- "Navigate" large contexts

**RAG (this knowledgebase)**:
- Pre-indexed retrieval
- Static chunking at ingest time
- Document store only
- "Search" indexed contexts

**Together**: RLM decomposes complex queries into sub-tasks, RAG retrieves precise evidence for each, RLM synthesizes the final answer with citations.
