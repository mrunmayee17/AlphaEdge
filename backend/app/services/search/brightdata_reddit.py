"""Reddit scraping via Bright Data Datasets API — replaces PRAW."""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class BrightDataClient:
    """Reddit scraping via Bright Data Datasets API.

    Flow: trigger scrape → get snapshot_id → poll until results ready.
    Typical latency: 10-30s for scrape to complete.
    """

    BASE_URL = "https://api.brightdata.com/datasets/v3"

    def __init__(self, api_key: str, dataset_id: str):
        self.api_key = api_key
        self.dataset_id = dataset_id
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search_reddit(
        self,
        keyword: str,
        num_posts: int = 10,
        sort_by: str = "Hot",
        date_range: str = "Past month",
    ) -> str:
        """Trigger Reddit scrape by keyword. Returns snapshot_id."""
        resp = await self.client.post(
            f"{self.BASE_URL}/trigger",
            params={
                "dataset_id": self.dataset_id,
                "notify": "false",
                "include_errors": "true",
                "type": "discover_new",
                "discover_by": "keyword",
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": [
                    {
                        "keyword": keyword,
                        "date": date_range,
                        "num_of_posts": num_posts,
                        "sort_by": sort_by,
                    }
                ]
            },
        )
        resp.raise_for_status()
        data = resp.json()
        snapshot_id = data.get("snapshot_id")
        if not snapshot_id:
            raise ValueError(f"No snapshot_id in response: {data}")
        logger.info(f"Reddit scrape triggered: keyword={keyword}, snapshot={snapshot_id}")
        return snapshot_id

    async def search_subreddit(
        self,
        subreddit_url: str,
        keyword: str = "",
        sort_by: str = "New",
        num_posts: int = 10,
    ) -> str:
        """Trigger Reddit scrape by subreddit URL. Returns snapshot_id."""
        resp = await self.client.post(
            f"{self.BASE_URL}/trigger",
            params={
                "dataset_id": self.dataset_id,
                "notify": "false",
                "include_errors": "true",
                "type": "discover_new",
                "discover_by": "subreddit_url",
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": [
                    {
                        "url": subreddit_url,
                        "sort_by": sort_by,
                        "keyword": keyword,
                        "start_date": "",
                    }
                ]
            },
        )
        resp.raise_for_status()
        data = resp.json()
        snapshot_id = data.get("snapshot_id")
        if not snapshot_id:
            raise ValueError(f"No snapshot_id in response: {data}")
        return snapshot_id

    async def get_snapshot(self, snapshot_id: str) -> Optional[list[dict]]:
        """Poll for completed snapshot data. Returns None if still processing."""
        resp = await self.client.get(
            f"{self.BASE_URL}/snapshot/{snapshot_id}",
            params={"format": "json"},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if resp.status_code == 202:
            return None  # still processing
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self.client.aclose()
