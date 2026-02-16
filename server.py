"""KappaML MCP Server — exposes the KappaML API as MCP tools."""

from __future__ import annotations

import os
from typing import Any

import httpx
from kappaml import KappaML, KappaMLError
from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP(
    "KappaML",
    stateless_http=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_client(ctx: Context) -> KappaML:
    """Build a KappaML client from the request context.

    Auth priority:
      1. Authorization: Bearer <key> header sent by the MCP client
      2. KAPPAML_API_KEY environment variable (single-tenant fallback)
    """
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

    return KappaML(api_key=api_key)


async def _raw_request(
    client: KappaML,
    method: str,
    path: str,
    *,
    json: Any = None,
    params: dict[str, str] | None = None,
) -> Any:
    """Make a raw HTTP request using the SDK's underlying httpx client."""
    response = await client.client.request(
        method,
        f"{client.base_url}{path}",
        json=json,
        params=params,
    )
    response.raise_for_status()
    if response.status_code == 204 or not response.content:
        return {"status": "success"}
    return response.json()


def _error(e: httpx.HTTPStatusError) -> dict:
    """Turn an HTTP error into a structured dict for the MCP response."""
    try:
        detail = e.response.json()
    except Exception:
        detail = e.response.text
    return {"error": True, "status_code": e.response.status_code, "detail": detail}


def _sdk_error(e: KappaMLError) -> dict:
    """Turn a KappaML SDK error into a structured dict."""
    return {"error": True, "detail": str(e)}


# ── User tools ───────────────────────────────────────────────────────────────


@mcp.tool()
async def get_user_profile(ctx: Context) -> dict:
    """Get the current user's profile."""
    client = _get_client(ctx)
    try:
        return await _raw_request(client, "GET", "/users/me")
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
    client = _get_client(ctx)
    body: dict[str, str] = {}
    if display_name is not None:
        body["display_name"] = display_name
    if phone_number is not None:
        body["phone_number"] = phone_number
    try:
        return await _raw_request(client, "PATCH", "/users/me", json=body)
    except httpx.HTTPStatusError as e:
        return _error(e)


# ── API Key tools ────────────────────────────────────────────────────────────


@mcp.tool()
async def create_api_key(ctx: Context, name: str) -> dict:
    """Create a new API key.

    Args:
        name: A friendly name for the key.
    """
    client = _get_client(ctx)
    try:
        return await _raw_request(client, "POST", "/api-keys", json={"name": name})
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def list_api_keys(ctx: Context) -> list | dict:
    """List all API keys for the current user."""
    client = _get_client(ctx)
    try:
        return await _raw_request(client, "GET", "/api-keys")
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def delete_api_key(ctx: Context, key_id: str) -> dict:
    """Delete an API key.

    Args:
        key_id: The ID of the key to delete.
    """
    client = _get_client(ctx)
    try:
        return await _raw_request(client, "DELETE", f"/api-keys/{key_id}")
    except httpx.HTTPStatusError as e:
        return _error(e)


# ── Model tools ──────────────────────────────────────────────────────────────


@mcp.tool()
async def list_models(ctx: Context) -> list | dict:
    """List all models for the current user."""
    client = _get_client(ctx)
    try:
        return await _raw_request(client, "GET", "/models")
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def create_model(ctx: Context, name: str, ml_type: str) -> dict:
    """Create a new model.

    Args:
        name: Name of the model.
        ml_type: ML task type — 'regression', 'classification', or 'forecasting'.
    """
    client = _get_client(ctx)
    try:
        model_id = await client.create_model(name, ml_type, wait_for_deployment=False)
        return {"id": model_id}
    except KappaMLError as e:
        return _sdk_error(e)


@mcp.tool()
async def get_model(ctx: Context, model_id: str) -> dict:
    """Get details of a specific model.

    Args:
        model_id: The model ID.
    """
    client = _get_client(ctx)
    try:
        return await _raw_request(client, "GET", f"/models/{model_id}")
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def delete_model(ctx: Context, model_id: str) -> dict:
    """Delete a model.

    Args:
        model_id: The model ID to delete.
    """
    client = _get_client(ctx)
    try:
        await client.delete_model(model_id)
        return {"status": "success"}
    except KappaMLError as e:
        return _sdk_error(e)


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
    client = _get_client(ctx)
    try:
        if instance:
            params = {"instance": instance}
            return await _raw_request(
                client, "POST", f"/models/{model_id}/predict",
                json={"features": features}, params=params,
            )
        result = await client.predict(model_id, features)
        return {"prediction": result}
    except KappaMLError as e:
        return _sdk_error(e)
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
    client = _get_client(ctx)
    params = {"instance": instance} if instance else None
    try:
        return await _raw_request(
            client, "POST", f"/models/{model_id}/predict",
            json=[{"features": s} for s in samples], params=params,
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
    client = _get_client(ctx)
    try:
        if instance:
            params = {"instance": instance}
            return await _raw_request(
                client, "POST", f"/models/{model_id}/learn",
                json={"features": features, "target": target}, params=params,
            )
        return await client.learn(model_id, features, target)
    except KappaMLError as e:
        return _sdk_error(e)
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
    client = _get_client(ctx)
    params = {"instance": instance} if instance else None
    try:
        return await _raw_request(
            client, "POST", f"/models/{model_id}/learn",
            json=[
                {"features": s["features"], "target": s["target"]} for s in samples
            ],
            params=params,
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
    client = _get_client(ctx)
    body: dict[str, Any] = {"horizon": horizon}
    if features is not None:
        body["features"] = features
    params = {"instance": instance} if instance else None
    try:
        return await _raw_request(
            client, "POST", f"/models/{model_id}/forecast",
            json=body, params=params,
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
    client = _get_client(ctx)
    try:
        if instance:
            return await _raw_request(
                client, "GET", f"/models/{model_id}/metrics",
                params={"instance": instance},
            )
        return await client.get_metrics(model_id)
    except KappaMLError as e:
        return _sdk_error(e)
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
    client = _get_client(ctx)
    params: dict[str, str] = {"timerange": timerange}
    if instance:
        params["instance"] = instance
    try:
        return await _raw_request(
            client, "GET", f"/models/{model_id}/metrics/history", params=params,
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
    client = _get_client(ctx)
    try:
        return await _raw_request(client, "GET", f"/models/{model_id}/checkpoints")
    except httpx.HTTPStatusError as e:
        return _error(e)


@mcp.tool()
async def create_checkpoint(ctx: Context, model_id: str) -> dict:
    """Create a new checkpoint for a model.

    Args:
        model_id: The model ID.
    """
    client = _get_client(ctx)
    try:
        return await _raw_request(client, "POST", f"/models/{model_id}/checkpoints")
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
    client = _get_client(ctx)
    try:
        return await _raw_request(
            client, "GET", f"/models/{model_id}/checkpoints/{checkpoint_id}",
        )
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
    client = _get_client(ctx)
    try:
        return await _raw_request(
            client, "DELETE", f"/models/{model_id}/checkpoints/{checkpoint_id}",
        )
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
    client = _get_client(ctx)
    try:
        return await _raw_request(
            client, "POST",
            f"/models/{model_id}/checkpoints/{checkpoint_id}/restore",
        )
    except httpx.HTTPStatusError as e:
        return _error(e)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
