import httpx
from typing import Optional, Dict, Any

BASE_URL = "https://api.qtcq.xyz/"


class APIClient:
    """Simple async wrapper around the upstream QTC API.

    Methods raise httpx.HTTPStatusError for non-2xx responses so the FastAPI
    layer can convert them to proper HTTPExceptions if needed.
    """

    def __init__(self, base_url: str = BASE_URL, timeout: float = 10.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def fetch_data(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, str] = {"symbol": symbol}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        resp = await self._client.get("/data", params=params)
        resp.raise_for_status()
        return resp.json()

    async def create_order(self, order_payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = await self._client.post("/orders", json=order_payload)
        resp.raise_for_status()
        return resp.json()

    async def get_order(self, order_id: str) -> Dict[str, Any]:
        resp = await self._client.get(f"/orders/{order_id}")
        resp.raise_for_status()
        return resp.json()


# single shared client instance used by the FastAPI app
api_client = APIClient()
