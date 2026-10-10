"""AlphaFoldDB structure downloads."""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import DownloadConfig

logger = logging.getLogger(__name__)

STATUS_OK, STATUS_SKIP, STATUS_FAIL = "ok", "skip", "fail"


def _fetch_bytes(url: str, cfg: DownloadConfig, accept_json: bool = False) -> bytes:
    headers = {"User-Agent": cfg.user_agent}
    if accept_json:
        headers["Accept"] = "application/json"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
        return resp.read()


def _write(path: str, data: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(data)


def download_one(acc: str, out_dir: str, cfg: DownloadConfig) -> tuple[str, str]:
    """Fetch one AlphaFoldDB structure. Returns (accession, status).

    Tries the direct v4 URL first (retrying on network errors, giving up on 404),
    then falls back to the prediction API, which resolves the current file URL.
    """
    out_path = os.path.join(out_dir, f"{acc}.pdb")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return acc, STATUS_SKIP

    acc_u = acc.strip().upper()
    direct_url = cfg.file_template.format(acc=acc_u)

    for attempt in range(cfg.retries + 1):
        try:
            _write(out_path, _fetch_bytes(direct_url, cfg))
            return acc, STATUS_OK
        except urllib.error.HTTPError as err:
            if err.code == 404:
                break
        except Exception:
            pass
        if attempt < cfg.retries:
            time.sleep(cfg.backoff)

    try:
        meta = json.loads(
            _fetch_bytes(cfg.api_template.format(acc=acc_u), cfg, accept_json=True).decode("utf-8")
        )
        pdb_url = meta[0].get("pdbUrl") if meta else None
        if not pdb_url:
            return acc, STATUS_FAIL        # accession has no AlphaFoldDB entry
        data = _fetch_bytes(pdb_url, cfg)
        if not data:
            return acc, STATUS_FAIL
        _write(out_path, data)
        return acc, STATUS_OK
    except Exception:
        return acc, STATUS_FAIL


def _fmt_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def download_split_structures(split: str, accessions, out_dir: str, cfg: DownloadConfig) -> list[str]:
    """Download structures for one split. Returns accessions that failed."""
    os.makedirs(out_dir, exist_ok=True)
    accessions = [a for a in accessions if isinstance(a, str)]
    total = len(accessions)
    counts = {STATUS_OK: 0, STATUS_SKIP: 0, STATUS_FAIL: 0}
    failed: list[str] = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = [pool.submit(download_one, acc, out_dir, cfg) for acc in accessions]
        for i, fut in enumerate(as_completed(futures), 1):
            acc, status = fut.result()
            counts[status] += 1
            if status == STATUS_FAIL:
                failed.append(acc)
            if i % cfg.progress_every == 0 or i == total:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (total - i) / rate if rate > 0 else 0.0
                logger.info(
                    "[%s] %7d/%d (%5.1f%%) ok=%d skip=%d fail=%d %.1f/s eta %s",
                    split, i, total, 100 * i / total, counts[STATUS_OK],
                    counts[STATUS_SKIP], counts[STATUS_FAIL], rate, _fmt_hms(eta),
                )

    logger.info(
        "[%s] done in %s: %d downloaded, %d already present, %d failed (%.1f%%)",
        split, _fmt_hms(time.time() - start), counts[STATUS_OK], counts[STATUS_SKIP],
        counts[STATUS_FAIL], 100 * counts[STATUS_FAIL] / max(total, 1),
    )
    return failed


def download_all(raw: dict, splits, pdb_root: str, cfg: DownloadConfig, failed_path: str | None = None):
    """Download every split. Returns ({split: pdb_dir}, {split: failed_accessions}).

    Proteins that fail get no retrieval hits downstream.
    """
    pdb_dirs, failed = {}, {}
    for split in splits:
        pdb_dirs[split] = os.path.join(pdb_root, split)
        failed[split] = download_split_structures(split, raw[split]["AlphaFoldDB"], pdb_dirs[split], cfg)
    if failed_path:
        os.makedirs(os.path.dirname(failed_path), exist_ok=True)
        with open(failed_path, "w") as fh:
            json.dump(failed, fh, indent=2)
        logger.info("failed-accession lists saved to %s", failed_path)
    return pdb_dirs, failed
