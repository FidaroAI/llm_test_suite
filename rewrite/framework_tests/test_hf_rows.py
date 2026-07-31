import json

import pytest

from llmeval.generation.hf_rows import DownloadFailed, cached_rows, fetch_rows


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    """Serves ``rows`` 2 at a time, recording the offsets it was asked for."""

    def __init__(self, rows, status_code=200):
        self.rows = rows
        self.status_code = status_code
        self.offsets = []

    def get(self, url, params=None, headers=None, timeout=None):
        offset = params["offset"]
        self.offsets.append(offset)
        page = self.rows[offset:offset + 2]
        return FakeResponse(
            self.status_code,
            {"num_rows_total": len(self.rows), "rows": [{"row": r} for r in page]},
        )


def test_fetch_rows_pages_until_it_has_them_all():
    session = FakeSession([{"i": 0}, {"i": 1}, {"i": 2}])
    rows = fetch_rows("d", "c", "s", session=session, page_size=2)
    assert rows == [{"i": 0}, {"i": 1}, {"i": 2}]
    assert session.offsets == [0, 2]


def test_a_gated_dataset_raises_with_the_hint():
    session = FakeSession([{"i": 0}], status_code=403)
    with pytest.raises(DownloadFailed, match="accept the terms"):
        fetch_rows("d", "c", "s", session=session, gated_hint="accept the terms")


def test_a_non_ok_status_raises():
    session = FakeSession([{"i": 0}], status_code=500)
    with pytest.raises(DownloadFailed, match="500"):
        fetch_rows("d", "c", "s", session=session)


def test_cached_rows_writes_once_and_reuses_the_file(tmp_path):
    path = tmp_path / "dataset.json"
    session = FakeSession([{"i": 0}])
    assert cached_rows(path, "d", "c", "s", session=session, page_size=2) == [{"i": 0}]
    assert json.loads(path.read_text()) == [{"i": 0}]

    exploding = FakeSession([])
    exploding.get = lambda *a, **k: pytest.fail("should not re-download")
    assert cached_rows(path, "d", "c", "s", session=exploding) == [{"i": 0}]
