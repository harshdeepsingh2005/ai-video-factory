"""Google Drive client unit tests (mocked, no credentials required)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from storage.google_drive import GoogleDriveClient


def test_drive_client_requires_credentials(monkeypatch):
    monkeypatch.delenv("GDRIVE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GDRIVE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("GDRIVE_REFRESH_TOKEN", raising=False)
    client = GoogleDriveClient()
    with pytest.raises(RuntimeError, match="Missing Google Drive OAuth credentials"):
        client._get_service()


def test_drive_ensure_production_tree_mocked():
    client = GoogleDriveClient("id", "secret", "token")
    mock_svc = MagicMock()
    mock_svc.files.return_value.list.return_value.execute.return_value = {"files": []}
    counter = iter(range(1, 20))
    mock_svc.files.return_value.create.return_value.execute.side_effect = (
        lambda: {"id": f"folder_{next(counter)}"}
    )
    client._service = mock_svc

    tree = client.ensure_production_tree("2026-11-27")
    assert "Long-Form" in tree
    assert "Shorts" in tree
    assert "Manifest" in tree
