"""KappaML MCP Server — exposes the KappaML API as MCP tools."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from mcp.server.fastmcp import Context, FastMCP

from kappaml_client import KappaMLClient

# ── Lifespan: shared httpx client ────────────────────────────────────────────


@asynccontextmanager
async def lifespan(server: FastMCP):
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        yield {"http_client": http_client}


mcp = FastMCP(
    "KappaML",
    stateless_http=True,
    lifespan=lifespan,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_client(ctx: Context) -> KappaMLClient:
    """Build a KappaMLClient from the request context.

    Auth priority:
      1. Authorization: Bearer <key> header sent by the MCP client
      2. KAPPAML_API_KEY environment variable (single-tenant fallback)
    """
    http_client: httpx.AsyncClient = ctx.request_context.lifespan_context[
        "http_client"
    ]

    api_key: str | None = None
    request = ctx.request_context.request
    if request:
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            api_key = auth_header[7:].strip()

    if not api_key:
        api_key = os.environ.get("KAPPAML_API_KEY")

    if not api_key:
        raise ValueError(
            "No API key found. Send an Authorization: Bearer <key> header "
            "or set the KAPPAML_API_KEY environment variable."
        )

    return KappaMLClient(http_client, api_key)


def _error(e: httpx.HTTPStatusError) -> dict:
    """Turn an HTTP error into a structured dict for the MCP response."""
    try:
        detail = e.response.json()
    except Exception:
        detail = e.response.text
    return {"error": True, "status_code": e.response.status_code, "detail": detail}


# ── User tools ───────────────────────────────────────────────────────────────


@mcp.tool()
async def get_user_profile(ctx: Context) -> dict:
    """Get the current user's profile."""
    try:
        return await _get_client(ctx).get_profile()
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def update_user_profile(
    ctx: Context,
    display_name: str | None = None,
    phone_number: str | None = None,
) -> dict:
    """Update the current user's profile.

    Args:
        display_name: New display name (1-100 chars).
        phone_number: New phone number in E.164 format.
    """
    try:
        return await _get_client(ctx).update_profile(
            display_name=display_name, phone_number=phone_number
        )
    except httpx.HTTPStatusError as e:
        return _error(e)


# ── API Key tools ────────────────────────────────────────────────────────────


@mcp.tool()
async def create_api_key(ctx: Context, name: str) -> dict:
    """Create a new API key.

    Args:
        name: A friendly name for the key.
    """
    try:
        return await _get_client(ctx).create_api_key(name)
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def list_api_keys(ctx: Context) -> list | dict:
    """List all API keys for the current user."""
    try:
        return await _get_client(ctx).list_api_keys()
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def delete_api_key(ctx: Context, key_id: str) -> dict:
    """Delete an API key.

    Args:
        key_id: The ID of the key to delete.
    """
    try:
        return await _get_client(ctx).delete_api_key(key_id)
    except httpx.HTTPStatusError as e:
        return _error(e)


# ── Model tools ──────────────────────────────────────────────────────────────


@mcp.tool()
async def list_models(ctx: Context) -> list | dict:
    """List all models for the current user."""
    try:
        return await _get_client(ctx).list_models()
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def create_model(ctx: Context, name: str, ml_type: str) -> dict:
    """Create a new model.

    Args:
        name: Name of the model.
        ml_type: ML task type — 'regression', 'classification', or 'forecasting'.
    """
    try:
        return await _get_client(ctx).create_model(name, ml_type)
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def get_model(ctx: Context, model_id: str) -> dict:
    """Get details of a specific model.

    Args:
        model_id: The model ID.
    """
    try:
        return await _get_client(ctx).get_model(model_id)
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def delete_model(ctx: Context, model_id: str) -> dict:
    """Delete a model.

    Args:
        model_id: The model ID to delete.
    """
    try:
        return await _get_client(ctx).delete_model(model_id)
    except httpx.HTTPStatusError as e:
        return _error(e)


# ── Predict tools ────────────────────────────────────────────────────────────


@mcp.tool()
async def predict(
    ctx: Context,
    model_id: str,
    features: dict[str, Any],
    instance: str | None = None,
) -> dict:
    """Make a single prediction.

    Args:
        model_id: The model ID.
        features: Feature name-value pairs, e.g. {"temperature": 25.5, "humidity": 65}.
        instance: Optional model instance ID (default "0").
    """
    try:
        return await _get_client(ctx).predict(
            model_id, features, instance=instance
        )
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def predict_batch(
    ctx: Context,
    model_id: str,
    samples: list[dict[str, Any]],
    instance: str | None = None,
) -> dict:
    """Make batch predictions.

    Args:
        model_id: The model ID.
        samples: List of feature dicts, e.g. [{"temp": 25}, {"temp": 30}].
        instance: Optional model instance ID (default "0").
    """
    try:
        return await _get_client(ctx).predict_batch(
            model_id, samples, instance=instance
        )
    except httpx.HTTPStatusError as e:
        return _error(e)


# ── Learn tools ──────────────────────────────────────────────────────────────


@mcp.tool()
async def learn(
    ctx: Context,
    model_id: str,
    features: dict[str, Any],
    target: float | int | str,
    instance: str | None = None,
) -> dict:
    """Learn from a single data point.

    Args:
        model_id: The model ID.
        features: Feature name-value pairs.
        target: The target value to learn from.
        instance: Optional model instance ID (default "0").
    """
    try:
        return await _get_client(ctx).learn(
            model_id, features, target, instance=instance
        )
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def learn_batch(
    ctx: Context,
    model_id: str,
    samples: list[dict[str, Any]],
    instance: str | None = None,
) -> dict:
    """Learn from a batch of data points.

    Args:
        model_id: The model ID.
        samples: List of dicts each with "features" and "target" keys,
                 e.g. [{"features": {"x": 1}, "target": 2.0}].
        instance: Optional model instance ID (default "0").
    """
    try:
        return await _get_client(ctx).learn_batch(
            model_id, samples, instance=instance
        )
    except httpx.HTTPStatusError as e:
        return _error(e)


# ── Forecast tool ────────────────────────────────────────────────────────────


@mcp.tool()
async def forecast(
    ctx: Context,
    model_id: str,
    horizon: int,
    features: list[dict[str, Any]] | None = None,
    instance: str | None = None,
) -> dict:
    """Forecast future values (time-series models).

    Args:
        model_id: The model ID.
        horizon: Number of time steps to forecast.
        features: Optional list of feature dicts, one per future time step.
        instance: Optional model instance ID (default "0").
    """
    try:
        return await _get_client(ctx).forecast(
            model_id, horizon, features=features, instance=instance
        )
    except httpx.HTTPStatusError as e:
        return _error(e)


# ── Metrics tools ────────────────────────────────────────────────────────────


@mcp.tool()
async def get_metrics(
    ctx: Context, model_id: str, instance: str | None = None
) -> dict:
    """Get current metrics for a model.

    Args:
        model_id: The model ID.
        instance: Optional model instance ID (default "0").
    """
    try:
        return await _get_client(ctx).get_metrics(model_id, instance=instance)
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def get_metrics_history(
    ctx: Context,
    model_id: str,
    timerange: str,
    instance: str | None = None,
) -> dict:
    """Get historical metrics for a model.

    Args:
        model_id: The model ID.
        timerange: Time range — e.g. '5m', '1h', '24h', '7d', '30d', '90d', '365d'.
        instance: Optional model instance ID (default "0").
    """
    try:
        return await _get_client(ctx).get_metrics_history(
            model_id, timerange, instance=instance
        )
    except httpx.HTTPStatusError as e:
        return _error(e)


# ── Checkpoint tools ─────────────────────────────────────────────────────────


@mcp.tool()
async def list_checkpoints(ctx: Context, model_id: str) -> dict:
    """List all checkpoints for a model.

    Args:
        model_id: The model ID.
    """
    try:
        return await _get_client(ctx).list_checkpoints(model_id)
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def create_checkpoint(ctx: Context, model_id: str) -> dict:
    """Create a new checkpoint for a model.

    Args:
        model_id: The model ID.
    """
    try:
        return await _get_client(ctx).create_checkpoint(model_id)
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def get_checkpoint(
    ctx: Context, model_id: str, checkpoint_id: str
) -> dict:
    """Get details of a specific checkpoint.

    Args:
        model_id: The model ID.
        checkpoint_id: The checkpoint ID.
    """
    try:
        return await _get_client(ctx).get_checkpoint(model_id, checkpoint_id)
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def delete_checkpoint(
    ctx: Context, model_id: str, checkpoint_id: str
) -> dict:
    """Delete a checkpoint.

    Args:
        model_id: The model ID.
        checkpoint_id: The checkpoint ID.
    """
    try:
        return await _get_client(ctx).delete_checkpoint(model_id, checkpoint_id)
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def restore_checkpoint(
    ctx: Context, model_id: str, checkpoint_id: str
) -> dict:
    """Restore a model from a checkpoint.

    Args:
        model_id: The model ID.
        checkpoint_id: The checkpoint ID to restore.
    """
    try:
        return await _get_client(ctx).restore_checkpoint(model_id, checkpoint_id)
    except httpx.HTTPStatusError as e:
        return _error(e)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
