"""Optional Cloudflare Worker API client for remote job state sync."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


class CloudflareJobClient:
    """Syncs job state with the free-tier Cloudflare Worker + D1 API."""

    def __init__(
        self,
        worker_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.worker_url = (worker_url or os.environ.get("CF_WORKER_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("CF_API_KEY", "")

    @property
    def enabled(self) -> bool:
        return bool(self.worker_url)

    def update_job(self, job_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.patch(
            f"{self.worker_url}/jobs/{job_id}",
            json=payload,
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
