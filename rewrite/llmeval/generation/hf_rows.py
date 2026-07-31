"""Download a dataset from the Hugging Face datasets-server rows API.

Three plugins (``agentharm_refusal``, ``multifaceted``, ``research_rubrics``) pull from the
same endpoint in the same way, so the paging loop lives here once. Ported from the legacy
``scripts_repo/download_*.mjs``, which the promptfoo suite still uses.

Raw rows are stored untransformed: shaping into test cases is each plugin's job, and keeping
the download dumb means a transform change never forces a re-download.

``requests`` is imported lazily, so the core package stays network-free and the tests inject a
fake session instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROWS_URL = "https://datasets-server.huggingface.co/rows"
# The datasets-server caps a single request at 100 rows.
PAGE_SIZE = 100
_TIMEOUT = 30


class DownloadFailed(RuntimeError):
    """The dataset could not be downloaded."""


def _session(session):
    if session is not None:
        return session
    import requests  # lazy: keep the core package network-free

    return requests.Session()


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    *,
    session=None,
    token: str | None = None,
    gated_hint: str | None = None,
    page_size: int = PAGE_SIZE,
) -> list[dict[str, Any]]:
    """Every row of ``dataset``/``config``/``split``, untransformed.

    :param gated_hint: appended to the error for 401/403, which for a gated dataset means
        "accept the terms" rather than "something is broken".
    """
    http = _session(session)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    rows: list[dict[str, Any]] = []
    offset, total = 0, None
    while total is None or offset < total:
        params = {
            "dataset": dataset, "config": config, "split": split,
            "offset": offset, "length": page_size,
        }
        resp = http.get(ROWS_URL, params=params, headers=headers, timeout=_TIMEOUT)
        if resp.status_code in (401, 403):
            raise DownloadFailed(
                f"HF rows API {resp.status_code} for {dataset}"
                + (f": {gated_hint}" if gated_hint else "")
            )
        if resp.status_code != 200:
            raise DownloadFailed(
                f"HF rows API {resp.status_code} for {dataset} at offset {offset}"
            )
        page = resp.json()
        total = page.get("num_rows_total")
        entries = page.get("rows") or []
        if not entries:
            break
        rows.extend(entry["row"] for entry in entries)
        offset += len(entries)
        if total is None:
            total = len(rows)
    logger.info("downloaded %d row(s) from %s", len(rows), dataset)
    return rows


def cached_rows(
    path: Path | str, dataset: str, config: str, split: str, **kwargs
) -> list[dict[str, Any]]:
    """``fetch_rows``, but written to ``path`` and reused if it is already there.

    The reuse is the whole point of doing downloads in ``generate_testcases``: the first call
    pays for the network, every later one is local.
    """
    path = Path(path)
    if path.is_file():
        logger.info("reusing cached dataset %s", path)
        return json.loads(path.read_text(encoding="utf-8"))
    rows = fetch_rows(dataset, config, split, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows
