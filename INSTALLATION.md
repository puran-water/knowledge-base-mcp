# Installation Guide

This guide will walk you through setting up the Semantic Search MCP Server from scratch.

## System Requirements

### Minimum
- **OS**: Linux, macOS, or Windows (with WSL2 recommended)
- **RAM**: 4GB available
- **CPU**: 2 cores
- **Disk**: 5GB for Docker images + space for your documents
- **Python**: 3.9 or newer

### Recommended
- **RAM**: 8GB available
- **CPU**: 4+ cores
- **Disk**: 10GB+ for Docker images and indexes

## Prerequisites Installation

### 1. Docker Desktop

Docker is required for running Qdrant (vector database) and TEI (reranker).

#### macOS
1. Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. Install the `.dmg` file
3. Launch Docker Desktop and wait for it to start
4. Verify installation:
```bash
docker --version
docker-compose --version
```

#### Windows
1. Install WSL2 first: [docs.microsoft.com/en-us/windows/wsl/install](https://docs.microsoft.com/en-us/windows/wsl/install)
2. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)
3. Enable WSL2 backend during installation
4. Launch Docker Desktop
5. Verify installation (in WSL2 terminal):
```bash
docker --version
docker-compose --version
```

#### Linux
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get install docker-compose-plugin

# Add your user to docker group (to avoid sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker-compose --version
```

### 2. Ollama

Ollama provides the embedding model for semantic search.

#### macOS
```bash
# Download and install from ollama.com
curl -fsSL https://ollama.com/install.sh | sh

# Or use Homebrew
brew install ollama
```

#### Windows
1. Download installer from [ollama.com/download/windows](https://ollama.com/download/windows)
2. Run the installer
3. Ollama will start automatically as a service

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Verify Ollama Installation
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Should return JSON with available models
```

If not running, start it:
```bash
# macOS/Linux
ollama serve

# Windows: Ollama runs as a service automatically
```

### 3. Python 3.9+

#### macOS
```bash
# Using Homebrew
brew install python@3.11
```

#### Windows (WSL2)
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# Fedora
sudo dnf install python3.11
```

Verify:
```bash
python3 --version
# Should show 3.9 or newer
```

## Application Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/knowledge-base-mcp.git
cd knowledge-base-mcp
```

Or if you downloaded as ZIP:
```bash
unzip knowledge-base-mcp.zip
cd knowledge-base-mcp
```

### Step 2: Pull the Embedding Model

```bash
ollama pull snowflake-arctic-embed:xs
```

This downloads the ~70MB embedding model. Wait for it to complete.

Verify:
```bash
ollama list
# Should show snowflake-arctic-embed:xs in the list
```

### Step 3: Start Docker Services

```bash
docker-compose up -d
```

This starts:
- **Qdrant** on port 6333 (vector database)
- **TEI Reranker** on port 8087 (cross-encoder for reranking)

Wait 30-60 seconds for services to initialize.

Verify services are healthy:
```bash
# Check status
docker-compose ps

# Should show both services as "healthy"

# Test Qdrant
curl http://localhost:6333/collections
# Should return: {"result":{"collections":[]}}

# Test Reranker
curl http://localhost:8087/health
# Should return: OK
```

If services aren't healthy, check logs:
```bash
docker-compose logs reranker
docker-compose logs qdrant
```

### Step 4: Set Up Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # macOS/Linux
# OR
# .venv\Scripts\activate   # Windows (legacy)
```

Your prompt should now show `(.venv)` prefix.

### Step 5: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `fastmcp` - MCP server framework
- `qdrant-client` - Vector database client
- `markitdown` - Document extraction
- `docling` - High-fidelity PDF processing
- Other dependencies

Installation may take 5-10 minutes due to large packages like `torch` and `transformers`.

> 💡 **Docling cache**: Set `HF_HOME` to a writable directory (for example `export HF_HOME="$PWD/.cache/hf"`) before your first ingest so Docling can cache its layout models once.

### Step 6: Create Configuration Files

```bash
# Optional: Copy environment variables template
cp .env.example .env
```

Configuration files will be created in the next step based on your MCP client.

### Step 7: Create Data Directory

```bash
mkdir -p data
```

This directory will store SQLite FTS databases (ignored by git).

## MCP Client Configuration

Choose the configuration option for your MCP client:

### Option A: Claude Code

1. **Copy the template**:
```bash
cp .mcp.json.example .mcp.json
```

2. **Get your Python venv path**:
```bash
which python  # macOS/Linux
# OR
where python  # Windows

# Example output: /home/user/knowledge-base-mcp/.venv/bin/python
```

3. **Edit `.mcp.json`** and update the Python path:
```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "/full/path/to/.venv/bin/python",
      "args": ["server.py", "stdio"],
      ...
    }
  }
}
```

4. **Verify connection**: Claude Code should detect the configuration automatically

### Option B: Claude Desktop

1. **Locate your Claude Desktop config**:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

2. **Copy the template contents**:
```bash
# View the example
cat claude_desktop_config.json.example
```

3. **Edit your Claude Desktop config** and add the MCP server configuration:

**For macOS/Linux**:
```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "/full/path/to/.venv/bin/python",
      "args": ["/full/path/to/knowledge-base-mcp/server.py", "stdio"],
      "env": {
        "OLLAMA_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "snowflake-arctic-embed:xs",
        "TEI_RERANK_URL": "http://localhost:8087",
        "QDRANT_URL": "http://localhost:6333",
        "FTS_DB_PATH": "/full/path/to/knowledge-base-mcp/data/fts.db",
        "NOMIC_KB_SCOPES": "{\"kb\":{\"collection\":\"main_kb\",\"title\":\"Main Knowledge Base\"}}"
      },
      "autoApprove": ["search_kb"]
    }
  }
}
```

**For Windows (using WSL)**:
```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "wsl",
      "args": [
        "-e",
        "bash",
        "-c",
        "cd /home/your-username/knowledge-base-mcp && OLLAMA_URL='http://localhost:11434' OLLAMA_MODEL='snowflake-arctic-embed:xs' TEI_RERANK_URL='http://localhost:8087' QDRANT_URL='http://localhost:6333' FTS_DB_PATH='/home/your-username/knowledge-base-mcp/data/fts.db' NOMIC_KB_SCOPES='{\"kb\":{\"collection\":\"main_kb\",\"title\":\"Main Knowledge Base\"}}' /home/your-username/knowledge-base-mcp/.venv/bin/python /home/your-username/knowledge-base-mcp/server.py stdio"
      ],
      "autoApprove": ["search_kb"]
    }
  }
}
```

4. **Restart Claude Desktop**

5. **Verify connection**: In Claude Desktop, you should see "Connected to MCP servers" with a checkmark next to "knowledge-base"

### Option C: Codex CLI

1. **Create Codex config directory** (if it doesn't exist):
```bash
mkdir -p .codex
```

2. **Copy the template**:
```bash
cp .codex/config.toml.example .codex/config.toml
```

3. **Edit `.codex/config.toml`** and update the Python path:
```toml
[mcp.knowledge-base]
command = "/full/path/to/.venv/bin/python"
args = ["server.py", "stdio"]
env = { OLLAMA_URL = "http://localhost:11434", OLLAMA_MODEL = "snowflake-arctic-embed:xs", TEI_RERANK_URL = "http://localhost:8087", QDRANT_URL = "http://localhost:6333", FTS_DB_PATH = "data/fts.db", NOMIC_KB_SCOPES = "{\"kb\":{\"collection\":\"main_kb\",\"title\":\"Main Knowledge Base\"}}" }

[approval]
allowed_mcp_tools = ["mcp__knowledge-base__search_kb"]
```

4. **Test with Codex**:
```bash
codex "test MCP connection"
```

## First Document Ingestion

Now let's ingest some documents to test the system.

### 1. Prepare Documents

Create a test directory with some documents:
```bash
mkdir -p test_docs
# Copy some PDFs, Word docs, or text files into test_docs/
```

### 2. Run Ingestion

```bash
python ingest.py \
  --root test_docs \
  --collection test_kb \
  --ext .pdf,.docx,.txt \
  --fts-db data/test_kb_fts.db
```

Parameters explained:
- `--root`: Directory to scan for documents
- `--collection`: Name for this collection in Qdrant
- `--ext`: File extensions to process (comma-separated)
- `--fts-db`: Path to SQLite FTS database for this collection

### 3. Monitor Progress

The ingestion script will show:
- Files discovered
- Extraction progress
- Embedding progress
- Upload progress

For a typical document collection:
- **Small** (10 PDFs, ~100 pages): 1-2 minutes
- **Medium** (100 PDFs, ~1000 pages): 10-20 minutes
- **Large** (1000 PDFs, ~10000 pages): 2-4 hours

### 4. Verify Ingestion

```bash
# Check Qdrant collection
curl http://localhost:6333/collections/test_kb

# Should show point count and vector configuration
```

### 5. Test Search

```bash
python validate_search.py \
  --query "your search query here" \
  --collection test_kb \
  --mode hybrid \
  --top-k 5
```

You should see search results with scores, paths, and text snippets.

## Updating Configuration for Your Collection

After successful ingestion, update your MCP configuration to include the new collection:

Edit `.mcp.json` (or Claude Desktop config):

```json
{
  "env": {
    ...
    "NOMIC_KB_SCOPES": "{\"test_kb\":{\"collection\":\"test_kb\",\"title\":\"Test Knowledge Base\"}}"
  },
  "autoApprove": ["search_test_kb"]
}
```

Now restart your MCP client (Claude Desktop or Codex) and you'll have a `search_test_kb` tool available.

## Verification Checklist

Before using in production, verify:

- [ ] Docker services are running: `docker-compose ps`
- [ ] Ollama is accessible: `curl http://localhost:11434/api/tags`
- [ ] Embedding model is available: `ollama list | grep snowflake`
- [ ] Qdrant is accessible: `curl http://localhost:6333/collections`
- [ ] Reranker is healthy: `curl http://localhost:8087/health`
- [ ] Python environment is activated: `which python` shows venv path
- [ ] Documents are ingested: `curl http://localhost:6333/collections/{your_collection}`
- [ ] Search works: `python validate_search.py --query "test" --collection {your_collection}`
- [ ] MCP client connects: Check for "Connected" status in Claude Desktop

## Troubleshooting

### Docker services won't start

```bash
# Check if ports are already in use
lsof -i :6333  # Qdrant
lsof -i :8087  # Reranker

# If ports are in use, either:
# 1. Stop the conflicting service
# 2. Change ports in docker-compose.yml
```

### Ollama connection refused

```bash
# Manually start Ollama
ollama serve

# Or check if it's running on a different port
curl http://localhost:11434/api/tags
```

### Ingestion fails with "model not found"

```bash
# Ensure model is pulled
ollama pull snowflake-arctic-embed:xs

# List available models
ollama list
```

### Out of memory during ingestion

```bash
# Reduce batch size
python ingest.py ... --embed-batch-size 16

# Or process fewer documents at once
python ingest.py ... --max-docs-per-run 100
```

### Search returns empty results

```bash
# Verify collection exists
curl http://localhost:6333/collections/{collection_name}

# Check point count (should be > 0)

# Verify FTS database exists
ls -lh data/{collection_name}_fts.db
```

## Next Steps

- Read [USAGE.md](USAGE.md) for advanced ingestion and search options
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design details
- Check [FAQ.md](FAQ.md) for common questions
- Explore [examples/](examples/) for sample scripts

## Uninstallation

To remove the system:

```bash
# Stop and remove Docker containers
docker-compose down -v

# Remove Python virtual environment
rm -rf .venv

# Remove data (optional - this deletes your indexes!)
rm -rf data/

# Uninstall Ollama model (optional)
ollama rm snowflake-arctic-embed:xs
```
