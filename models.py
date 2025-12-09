"""Pydantic input models for knowledge-base MCP server tools.

This module defines validated input models for all MCP tools, providing:
- Automatic schema generation for MCP clients
- Input validation with constraints (min/max, patterns)
- Type safety and documentation

Usage:
    @mcp.tool(name="kb.search", ...)
    async def kb_search(ctx: Context, params: KBSearchInput) -> Dict[str, Any]:
        ...
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --- Enums ---

class ResponseProfile(str, Enum):
    """Response payload size profiles for retrieval tools."""
    slim = "slim"
    full = "full"
    diagnostic = "diagnostic"


class SearchMode(str, Enum):
    """Search routing modes for kb.search."""
    auto = "auto"
    semantic = "semantic"
    rerank = "rerank"
    hybrid = "hybrid"
    sparse = "sparse"
    sparse_splade = "sparse_splade"
    colbert = "colbert"


class ChunkProfile(str, Enum):
    """Chunking strategy profiles."""
    auto = "auto"
    heading_based = "heading_based"
    procedure_block = "procedure_block"
    table_row = "table_row"
    fixed_window = "fixed_window"


class EnhanceOp(str, Enum):
    """Supported enhancement operations."""
    add_synonyms = "add_synonyms"
    link_crossrefs = "link_crossrefs"
    fix_table_pages = "fix_table_pages"


# --- Base Config ---

class StrictModel(BaseModel):
    """Base model with strict validation."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )


class FlexibleModel(BaseModel):
    """Base model allowing extra fields (for scope dicts, etc.)."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="allow",
    )


# --- Ingestion Tool Inputs ---

class IngestExtractInput(StrictModel):
    """Input for ingest.extract_with_strategy tool."""
    path: str = Field(
        ...,
        description="Absolute or relative path to the document to extract",
        min_length=1,
    )
    plan: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional plan dict with doc_id, triage settings, chunk_profile",
    )


class IngestValidateExtractionInput(StrictModel):
    """Input for ingest.validate_extraction tool."""
    artifact_ref: str = Field(
        ...,
        description="Path to the blocks.json artifact to validate",
        min_length=1,
    )
    rules: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Validation rules (min_blocks, min_text_chars, etc.)",
    )


class IngestChunkInput(StrictModel):
    """Input for ingest.chunk_with_guidance tool."""
    artifacts_ref: str = Field(
        ...,
        description="Path to the blocks.json artifact to chunk",
        min_length=1,
    )
    profile: str = Field(
        default="auto",
        description="Chunking profile: auto, heading_based, procedure_block, table_row, fixed_window",
    )
    max_chars: int = Field(
        default=1800,
        description="Maximum characters per chunk",
        ge=100,
        le=10000,
    )
    overlap_sentences: int = Field(
        default=1,
        description="Number of sentences to overlap between chunks",
        ge=0,
        le=10,
    )


class IngestGenerateMetadataInput(StrictModel):
    """Input for ingest.generate_metadata tool."""
    doc_id: str = Field(
        ...,
        description="Document ID to generate metadata for",
        min_length=1,
    )
    artifact_ref: Optional[str] = Field(
        default=None,
        description="Optional path to chunks artifact",
    )
    policy: str = Field(
        default="strict_v1",
        description="Metadata generation policy",
    )


class IngestAssessQualityInput(StrictModel):
    """Input for ingest.assess_quality tool."""
    doc_id: str = Field(
        ...,
        description="Document ID to assess",
        min_length=1,
    )
    artifact_ref: Optional[str] = Field(
        default=None,
        description="Optional path to chunks artifact",
    )


class IngestEnhanceInput(StrictModel):
    """Input for ingest.enhance tool."""
    doc_id: str = Field(
        ...,
        description="Document ID to enhance",
        min_length=1,
    )
    op: str = Field(
        ...,
        description="Enhancement operation: add_synonyms, link_crossrefs, fix_table_pages",
    )
    args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Operation-specific arguments",
    )

    @field_validator("op")
    @classmethod
    def validate_op(cls, v: str) -> str:
        allowed = {"add_synonyms", "link_crossrefs", "fix_table_pages"}
        if v not in allowed:
            raise ValueError(f"op must be one of: {', '.join(sorted(allowed))}")
        return v


class IngestUpsertInput(StrictModel):
    """Input for ingest.upsert tool."""
    doc_id: str = Field(
        ...,
        description="Document ID to upsert",
        min_length=1,
    )
    collection: Optional[str] = Field(
        default=None,
        description="Target collection name or slug",
    )
    chunks_artifact: Optional[str] = Field(
        default=None,
        description="Path to chunks.json artifact",
    )
    metadata_artifact: Optional[str] = Field(
        default=None,
        description="Path to metadata.json artifact",
    )
    thin_payload: Optional[bool] = Field(
        default=None,
        description="Omit text from Qdrant payload (use FTS for hydration)",
    )
    skip_vectors: bool = Field(
        default=False,
        description="Skip Qdrant vector upsert (FTS-only mode)",
    )
    update_graph: bool = Field(
        default=True,
        description="Update knowledge graph with entities",
    )
    update_summary: bool = Field(
        default=True,
        description="Update summary index",
    )
    fts_rebuild: bool = Field(
        default=False,
        description="Rebuild FTS index from scratch",
    )
    client_id: Optional[str] = Field(
        default=None,
        description="Client identifier for provenance tracking",
    )
    client_model: Optional[str] = Field(
        default=None,
        description="Client model name for provenance",
    )
    client_decisions: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of client decision records for audit",
    )


class IngestUpsertBatchInput(StrictModel):
    """Input for ingest.upsert_batch tool."""
    upserts: List[Dict[str, Any]] = Field(
        ...,
        description="List of upsert specs, each with doc_id and optional overrides",
        min_length=1,
    )
    collection: Optional[str] = Field(
        default=None,
        description="Default collection for all upserts",
    )
    parallel: int = Field(
        default=4,
        description="Maximum parallel upsert operations",
        ge=1,
        le=32,
    )
    thin_payload: Optional[bool] = Field(default=None)
    skip_vectors: bool = Field(default=False)
    update_graph: bool = Field(default=True)
    update_summary: bool = Field(default=True)
    fts_rebuild: bool = Field(default=False)
    client_id: Optional[str] = Field(default=None)
    client_model: Optional[str] = Field(default=None)


class IngestGenerateSummaryInput(StrictModel):
    """Input for ingest.generate_summary tool."""
    doc_id: str = Field(
        ...,
        description="Document ID for the summary",
        min_length=1,
    )
    summary_text: str = Field(
        ...,
        description="Summary text content (3-5 sentences recommended)",
        min_length=10,
        max_length=10000,
    )
    section_path: Optional[List[str]] = Field(
        default=None,
        description="Hierarchical path like ['Chapter 1', 'Overview']",
    )
    collection: Optional[str] = Field(default=None)
    element_ids: Optional[List[str]] = Field(
        default=None,
        description="Element IDs cited in the summary",
    )
    summary_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata including model, prompt_sha",
    )
    client_id: Optional[str] = Field(default=None)
    client_model: Optional[str] = Field(default=None)


class IngestCorpusUpsertInput(StrictModel):
    """Input for ingest.corpus_upsert tool."""
    root_path: str = Field(
        ...,
        description="Root directory containing documents to ingest",
        min_length=1,
    )
    collection: Optional[str] = Field(default=None)
    extractor: str = Field(
        default="auto",
        description="Extraction method (auto uses Docling for all)",
    )
    chunk_profile: str = Field(default="auto")
    max_chars: int = Field(default=1800, ge=100, le=10000)
    overlap_sentences: int = Field(default=1, ge=0, le=10)
    skip_vectors: bool = Field(default=False)
    update_graph: bool = Field(default=True)
    update_summary: bool = Field(default=True)
    fts_rebuild: bool = Field(default=False)
    dry_run: bool = Field(
        default=False,
        description="Simulate without writing",
    )
    thin_payload: Optional[bool] = Field(default=None)
    extensions: Optional[str] = Field(
        default=None,
        description="Comma-separated file extensions to process",
    )
    client_id: Optional[str] = Field(default=None)
    client_model: Optional[str] = Field(default=None)


# --- Retrieval Tool Inputs ---

class KBSearchInput(FlexibleModel):
    """Input for kb.search tool (unified search dispatcher)."""
    query: str = Field(
        ...,
        description="Search query text",
        min_length=1,
        max_length=10000,
    )
    collection: Optional[str] = Field(
        default=None,
        description="Collection slug or name to search",
    )
    mode: str = Field(
        default="rerank",
        description="Search mode: auto, semantic, rerank, hybrid, sparse, sparse_splade, colbert",
    )
    retrieve_k: int = Field(
        default=24,
        description="Number of candidates to retrieve",
        ge=1,
        le=256,
    )
    return_k: int = Field(
        default=8,
        description="Number of results to return",
        ge=1,
        le=256,
    )
    top_k: int = Field(
        default=8,
        description="Top-k for reranking stage",
        ge=1,
        le=256,
    )
    scope: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Scope filters and doc_id boosts",
    )
    response_profile: str = Field(
        default="slim",
        description="Response detail level: slim, full, diagnostic",
    )

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = {"auto", "semantic", "rerank", "hybrid", "sparse", "sparse_splade", "colbert"}
        if v.lower() not in allowed:
            raise ValueError(f"mode must be one of: {', '.join(sorted(allowed))}")
        return v.lower()


class KBSparseInput(FlexibleModel):
    """Input for kb.sparse tool (BM25 lexical search)."""
    query: str = Field(..., min_length=1, max_length=10000)
    collection: Optional[str] = Field(default=None)
    retrieve_k: int = Field(default=24, ge=1, le=256)
    return_k: int = Field(default=8, ge=1, le=256)
    scope: Optional[Dict[str, Any]] = Field(default=None)
    response_profile: str = Field(default="slim")


class KBSparseSpladeInput(FlexibleModel):
    """Input for kb.sparse_splade tool (SPLADE sparse vector search)."""
    query: str = Field(..., min_length=1, max_length=10000)
    collection: Optional[str] = Field(default=None)
    retrieve_k: int = Field(default=24, ge=1, le=256)
    return_k: int = Field(default=8, ge=1, le=256)
    scope: Optional[Dict[str, Any]] = Field(default=None)
    response_profile: str = Field(default="slim")


class KBDenseInput(FlexibleModel):
    """Input for kb.dense tool (semantic vector search)."""
    query: str = Field(..., min_length=1, max_length=10000)
    collection: Optional[str] = Field(default=None)
    retrieve_k: int = Field(default=24, ge=1, le=256)
    return_k: int = Field(default=8, ge=1, le=256)
    scope: Optional[Dict[str, Any]] = Field(default=None)
    response_profile: str = Field(default="slim")


class KBHybridInput(FlexibleModel):
    """Input for kb.hybrid tool (RRF fusion of BM25 + dense)."""
    query: str = Field(..., min_length=1, max_length=10000)
    collection: Optional[str] = Field(default=None)
    retrieve_k: int = Field(default=24, ge=1, le=256)
    return_k: int = Field(default=8, ge=1, le=256)
    top_k: int = Field(default=8, ge=1, le=256)
    scope: Optional[Dict[str, Any]] = Field(default=None)
    response_profile: str = Field(default="slim")


class KBRerankInput(FlexibleModel):
    """Input for kb.rerank tool (dense + TEI reranker)."""
    query: str = Field(..., min_length=1, max_length=10000)
    collection: Optional[str] = Field(default=None)
    retrieve_k: int = Field(default=24, ge=1, le=256)
    return_k: int = Field(default=8, ge=1, le=256)
    top_k: int = Field(default=8, ge=1, le=256)
    scope: Optional[Dict[str, Any]] = Field(default=None)
    response_profile: str = Field(default="slim")


class KBColbertInput(FlexibleModel):
    """Input for kb.colbert tool (ColBERT multi-vector search)."""
    query: str = Field(..., min_length=1, max_length=10000)
    collection: Optional[str] = Field(default=None)
    retrieve_k: int = Field(default=24, ge=1, le=256)
    return_k: int = Field(default=8, ge=1, le=256)
    scope: Optional[Dict[str, Any]] = Field(default=None)
    response_profile: str = Field(default="slim")


class KBBatchInput(FlexibleModel):
    """Input for kb.batch tool (multi-query batch search)."""
    queries: List[str] = Field(
        ...,
        description="List of queries to execute",
        min_length=1,
    )
    routes: Optional[List[str]] = Field(
        default=None,
        description="Per-query route overrides",
    )
    collection: Optional[str] = Field(default=None)
    collections: Optional[List[str]] = Field(
        default=None,
        description="Per-query collection overrides",
    )
    scopes: Optional[List[Dict[str, Any]]] = Field(default=None)
    scope: Optional[Dict[str, Any]] = Field(default=None)
    retrieve_k: int = Field(default=24, ge=1, le=256)
    return_k: int = Field(default=8, ge=1, le=256)
    response_profile: str = Field(default="slim")


class KBQualityInput(FlexibleModel):
    """Input for kb.quality tool (inspect search hit quality)."""
    hits: List[Dict[str, Any]] = Field(
        ...,
        description="Search hits to analyze",
    )
    collection: Optional[str] = Field(default=None)
    rules: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Quality gate rules",
    )
    query: Optional[str] = Field(
        default=None,
        description="Original query for coverage analysis",
    )


class KBHintInput(StrictModel):
    """Input for kb.hint tool (sparse query expansion hints)."""
    term: Optional[str] = Field(
        default=None,
        description="Single term to expand",
    )
    terms: Optional[List[str]] = Field(
        default=None,
        description="Multiple terms to expand",
    )

    @field_validator("terms")
    @classmethod
    def require_term_or_terms(cls, v: Optional[List[str]], info) -> Optional[List[str]]:
        if v is None and info.data.get("term") is None:
            raise ValueError("Either 'term' or 'terms' must be provided")
        return v


class KBTableInput(FlexibleModel):
    """Input for kb.table tool (table row lookup)."""
    query: str = Field(..., min_length=1, max_length=10000)
    collection: Optional[str] = Field(default=None)
    doc_id: Optional[str] = Field(
        default=None,
        description="Filter to specific document",
    )
    where: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional filter conditions",
    )
    limit: int = Field(
        default=10,
        description="Maximum rows to return",
        ge=1,
        le=100,
    )


class KBOpenInput(StrictModel):
    """Input for kb.open tool (open specific chunk by ID)."""
    chunk_id: Optional[str] = Field(
        default=None,
        description="Chunk ID to open",
    )
    element_id: Optional[str] = Field(
        default=None,
        description="Element ID to look up chunk for",
    )
    collection: Optional[str] = Field(default=None)
    start: Optional[int] = Field(
        default=None,
        description="Start character offset",
    )
    end: Optional[int] = Field(
        default=None,
        description="End character offset",
    )


class KBNeighborsInput(StrictModel):
    """Input for kb.neighbors tool (retrieve surrounding chunks)."""
    chunk_id: str = Field(
        ...,
        description="Reference chunk ID to find neighbors for",
        min_length=1,
    )
    collection: Optional[str] = Field(default=None)
    n: int = Field(
        default=1,
        description="Number of neighbors on each side (total = 2*n + 1)",
        ge=0,
        le=100,
    )
    response_profile: str = Field(default="slim")


class KBSummaryInput(StrictModel):
    """Input for kb.summary tool (query hierarchical summaries)."""
    topic: str = Field(
        ...,
        description="Topic to search summaries for",
        min_length=1,
        max_length=1000,
    )
    collection: Optional[str] = Field(default=None)
    limit: int = Field(
        default=3,
        description="Maximum summaries to return",
        ge=1,
        le=50,
    )


class KBOutlineInput(StrictModel):
    """Input for kb.outline tool (get document structure)."""
    doc_id: str = Field(
        ...,
        description="Document ID to get outline for",
        min_length=1,
    )
    collection: Optional[str] = Field(default=None)


class KBEntitiesInput(StrictModel):
    """Input for kb.entities tool (list knowledge graph entities)."""
    collection: Optional[str] = Field(default=None)
    types: Optional[List[str]] = Field(
        default=None,
        description="Filter by entity types",
    )
    match: Optional[str] = Field(
        default=None,
        description="Pattern to match entity names",
    )
    limit: int = Field(
        default=50,
        description="Maximum entities to return",
        ge=1,
        le=500,
    )


class KBLinkoutsInput(StrictModel):
    """Input for kb.linkouts tool (find entity mentions in documents)."""
    entity_id: str = Field(
        ...,
        description="Entity ID to find mentions for",
        min_length=1,
    )
    limit: int = Field(
        default=25,
        description="Maximum mentions to return",
        ge=1,
        le=100,
    )


class KBGraphInput(StrictModel):
    """Input for kb.graph tool (traverse knowledge graph)."""
    node_id: str = Field(
        ...,
        description="Starting node ID for traversal",
        min_length=1,
    )
    limit: int = Field(
        default=20,
        description="Maximum neighbors to return",
        ge=1,
        le=100,
    )


class KBPromoteInput(StrictModel):
    """Input for kb.promote tool (boost document in session)."""
    doc_id: str = Field(
        ...,
        description="Document ID to promote",
        min_length=1,
    )
    weight: float = Field(
        default=0.2,
        description="Promotion weight (positive)",
        ge=0.0,
        le=4.0,
    )


class KBDemoteInput(StrictModel):
    """Input for kb.demote tool (lower document in session)."""
    doc_id: str = Field(
        ...,
        description="Document ID to demote",
        min_length=1,
    )
    weight: float = Field(
        default=0.2,
        description="Demotion weight (will be negated)",
        ge=0.0,
        le=0.9,
    )


# Note: KBCollectionsInput not needed - tool takes no parameters
