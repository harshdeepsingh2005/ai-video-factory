"""Google Drive OAuth2 resumable uploader (GCP Standard Tier, ₹0 / no billing).

Reads OAuth credentials from environment variables or Kaggle Secrets:
  - GDRIVE_CLIENT_ID
  - GDRIVE_CLIENT_SECRET
  - GDRIVE_REFRESH_TOKEN
Never stores credentials in source code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional


class GoogleDriveClient:
    """Resumable Google Drive API v3 client using OAuth2 refresh tokens."""

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
    ) -> None:
        self.client_id = client_id or os.environ.get("GDRIVE_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("GDRIVE_CLIENT_SECRET")
        self.refresh_token = refresh_token or os.environ.get("GDRIVE_REFRESH_TOKEN")
        self._service = None

    def _get_service(self):
        if self._service is not None:
            return self._service

        if not (self.client_id and self.client_secret and self.refresh_token):
            raise RuntimeError(
                "Missing Google Drive OAuth credentials. Set GDRIVE_CLIENT_ID, "
                "GDRIVE_CLIENT_SECRET, and GDRIVE_REFRESH_TOKEN (e.g. via Kaggle Secrets)."
            )

        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        creds = Credentials(
            token=None,
            refresh_token=self.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=self.client_id,
            client_secret=self.client_secret,
            scopes=self.SCOPES,
        )
        creds.refresh(Request())
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def ensure_folder(self, name: str, parent_id: Optional[str] = None) -> str:
        """Find or create a folder in Google Drive and return its file ID."""
        service = self._get_service()
        safe_name = name.replace("'", "\\'")
        query = (
            f"name = '{safe_name}' and "
            "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        )
        if parent_id:
            query += f" and '{parent_id}' in parents"

        resp = service.files().list(q=query, fields="files(id, name)", pageSize=5).execute()
        files = resp.get("files", [])
        if files:
            return files[0]["id"]

        metadata: Dict[str, object] = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            metadata["parents"] = [parent_id]

        created = service.files().create(body=metadata, fields="id").execute()
        return created["id"]

    def ensure_production_tree(
        self, date_str: str, subfolders: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Create AI-Video-Factory/Production/YYYY-MM-DD/<subfolders> hierarchy."""
        subfolders = subfolders or [
            "Long-Form",
            "Shorts",
            "Thumbnails",
            "Narration",
            "Subtitles",
            "Scripts",
            "Assets",
            "Manifest",
        ]
        root_id = self.ensure_folder("AI-Video-Factory")
        prod_id = self.ensure_folder("Production", parent_id=root_id)
        date_id = self.ensure_folder(date_str, parent_id=prod_id)
        folder_ids = {"_root_date": date_id}
        for sub in subfolders:
            folder_ids[sub] = self.ensure_folder(sub, parent_id=date_id)
        return folder_ids

    def upload_file(
        self,
        local_path: Path,
        parent_folder_id: str,
        chunk_size_bytes: int = 8 * 1024 * 1024,
    ) -> Dict[str, object]:
        """Resumable upload with size verification."""
        from googleapiclient.http import MediaFileUpload

        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Cannot upload missing file: {local_path}")

        service = self._get_service()
        media = MediaFileUpload(
            str(local_path),
            resumable=True,
            chunksize=chunk_size_bytes,
        )
        metadata = {"name": local_path.name, "parents": [parent_folder_id]}
        request = service.files().create(
            body=metadata,
            media_body=media,
            fields="id, name, size",
        )
        response = None
        while response is None:
            _, response = request.next_chunk()

        return {
            "id": response["id"],
            "name": response["name"],
            "size": int(response.get("size", local_path.stat().st_size)),
        }
