"""Brave Search API client for web news/sentiment search."""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class BraveSearchClient:
    """Brave Web Search API for financial news and sentiment."""

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=10.0)

    async def search(
        self,
        query: str,
        count: int = 5,
        freshness: Optional[str] = None,
    ) -> dict:
        """Search the web. Returns dict with 'results' list.

        Args:
            query: Search query
            count: Number of results (max 20)
            freshness: Optional time filter ('pd'=past day, 'pw'=past week, 'pm'=past month)
        """
        params = {"q": query, "count": min(count, 20)}
        if freshness:
            params["freshness"] = freshness

        resp = await self.client.get(
            self.BASE_URL,
            params=params,
            headers={
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        web_results = data.get("web", {}).get("results", [])
        return {
            "query": query,
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "description": r.get("description", ""),
                    "age": r.get("age", ""),
                }
                for r in web_results
            ],
        }

    async def close(self):
        await self.client.aclose()
