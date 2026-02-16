# KappaML MCP Server

Remote MCP server that exposes the [KappaML API](https://api.kappaml.com) as tools for AI assistants like Claude Desktop and Claude Code.

## Tools (21)

| Category | Tools |
|----------|-------|
| **Users** | `get_user_profile`, `update_user_profile` |
| **API Keys** | `create_api_key`, `list_api_keys`, `delete_api_key` |
| **Models** | `list_models`, `create_model`, `get_model`, `delete_model` |
| **Predict** | `predict`, `predict_batch` |
| **Learn** | `learn`, `learn_batch` |
| **Forecast** | `forecast` |
| **Metrics** | `get_metrics`, `get_metrics_history` |
| **Checkpoints** | `list_checkpoints`, `create_checkpoint`, `get_checkpoint`, `delete_checkpoint`, `restore_checkpoint` |

## Setup

### Install locally

```bash
cd kappaml-mcp
pip install -e .
```

### Run the server

```bash
# Option 1: Set API key via environment variable (single-tenant)
export KAPPAML_API_KEY=sk.xxx
python server.py

# Option 2: Clients send their own key via Authorization header (multi-tenant)
python server.py
```

The server starts on `http://0.0.0.0:8000`.

### Docker

```bash
docker build -t kappaml-mcp .
docker run -p 8000:8000 -e KAPPAML_API_KEY=sk.xxx kappaml-mcp
```

## Client Configuration

### Claude Code

```bash
claude mcp add --transport http kappaml http://localhost:8000/mcp \
  --header "Authorization: Bearer sk.xxx"
```

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kappaml": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer sk.xxx"
      }
    }
  }
}
```

## Authentication

The server extracts the API key from the `Authorization: Bearer <key>` header sent by the MCP client and forwards it as `X-API-Key` to the KappaML API. If no header is present, it falls back to the `KAPPAML_API_KEY` environment variable.
