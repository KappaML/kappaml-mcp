"""Async HTTP client wrapper for the KappaML API."""

from typing import Any

import httpx

BASE_URL = "https://api.kappaml.com/v1"


class KappaMLClient:
    """Thin async wrapper around the KappaML REST API."""

    def __init__(self, http_client: httpx.AsyncClient, api_key: str) -> None:
        self._client = http_client
        self._headers = {"X-API-Key": api_key}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        response = await self._client.request(
            method,
            f"{BASE_URL}{path}",
            headers=self._headers,
            json=json,
            params=params,
        )
        response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return {"status": "success"}
        return response.json()

    # ── Users ────────────────────────────────────────────────────────────

    async def get_profile(self) -> dict:
        return await self._request("GET", "/users/me")

    async def update_profile(
        self,
        *,
        display_name: str | None = None,
        phone_number: str | None = None,
    ) -> dict:
        body: dict[str, str] = {}
        if display_name is not None:
            body["display_name"] = display_name
        if phone_number is not None:
            body["phone_number"] = phone_number
        return await self._request("PATCH", "/users/me", json=body)

    # ── API Keys ─────────────────────────────────────────────────────────

    async def create_api_key(self, name: str) -> dict:
        return await self._request("POST", "/api-keys", json={"name": name})

    async def list_api_keys(self) -> list[dict]:
        return await self._request("GET", "/api-keys")

    async def delete_api_key(self, key_id: str) -> dict:
        return await self._request("DELETE", f"/api-keys/{key_id}")

    # ── Models ───────────────────────────────────────────────────────────

    async def list_models(self) -> list[dict]:
        return await self._request("GET", "/models")

    async def create_model(self, name: str, ml_type: str) -> dict:
        return await self._request(
            "POST", "/models", json={"name": name, "ml_type": ml_type}
        )

    async def get_model(self, model_id: str) -> dict:
        return await self._request("GET", f"/models/{model_id}")

    async def delete_model(self, model_id: str) -> dict:
        return await self._request("DELETE", f"/models/{model_id}")

    # ── Predict ──────────────────────────────────────────────────────────

    async def predict(
        self, model_id: str, features: dict, *, instance: str | None = None
    ) -> dict:
        params = {"instance": instance} if instance else None
        return await self._request(
            "POST",
            f"/models/{model_id}/predict",
            json={"features": features},
            params=params,
        )

    async def predict_batch(
        self,
        model_id: str,
        samples: list[dict],
        *,
        instance: str | None = None,
    ) -> dict:
        params = {"instance": instance} if instance else None
        return await self._request(
            "POST",
            f"/models/{model_id}/predict",
            json=[{"features": s} for s in samples],
            params=params,
        )

    # ── Learn ────────────────────────────────────────────────────────────

    async def learn(
        self,
        model_id: str,
        features: dict,
        target: float | int | str,
        *,
        instance: str | None = None,
    ) -> dict:
        params = {"instance": instance} if instance else None
        return await self._request(
            "POST",
            f"/models/{model_id}/learn",
            json={"features": features, "target": target},
            params=params,
        )

    async def learn_batch(
        self,
        model_id: str,
        samples: list[dict],
        *,
        instance: str | None = None,
    ) -> dict:
        params = {"instance": instance} if instance else None
        return await self._request(
            "POST",
            f"/models/{model_id}/learn",
            json=[
                {"features": s["features"], "target": s["target"]} for s in samples
            ],
            params=params,
        )

    # ── Forecast ─────────────────────────────────────────────────────────

    async def forecast(
        self,
        model_id: str,
        horizon: int,
        *,
        features: list[dict] | None = None,
        instance: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"horizon": horizon}
        if features is not None:
            body["features"] = features
        params = {"instance": instance} if instance else None
        return await self._request(
            "POST",
            f"/models/{model_id}/forecast",
            json=body,
            params=params,
        )

    # ── Metrics ──────────────────────────────────────────────────────────

    async def get_metrics(
        self, model_id: str, *, instance: str | None = None
    ) -> dict:
        params = {"instance": instance} if instance else None
        return await self._request(
            "GET", f"/models/{model_id}/metrics", params=params
        )

    async def get_metrics_history(
        self,
        model_id: str,
        timerange: str,
        *,
        instance: str | None = None,
    ) -> dict:
        params: dict[str, str] = {"timerange": timerange}
        if instance:
            params["instance"] = instance
        return await self._request(
            "GET", f"/models/{model_id}/metrics/history", params=params
        )

    # ── Checkpoints ──────────────────────────────────────────────────────

    async def list_checkpoints(self, model_id: str) -> dict:
        return await self._request("GET", f"/models/{model_id}/checkpoints")

    async def create_checkpoint(self, model_id: str) -> dict:
        return await self._request("POST", f"/models/{model_id}/checkpoints")

    async def get_checkpoint(self, model_id: str, checkpoint_id: str) -> dict:
        return await self._request(
            "GET", f"/models/{model_id}/checkpoints/{checkpoint_id}"
        )

    async def delete_checkpoint(self, model_id: str, checkpoint_id: str) -> dict:
        return await self._request(
            "DELETE", f"/models/{model_id}/checkpoints/{checkpoint_id}"
        )

    async def restore_checkpoint(self, model_id: str, checkpoint_id: str) -> dict:
        return await self._request(
            "POST", f"/models/{model_id}/checkpoints/{checkpoint_id}/restore"
        )
