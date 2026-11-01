"""Fetch the two public UCI datasets into ``data/raw``.

Raw data is never committed to git. Both datasets are CC BY 4.0, so you are free to
download and use them as long as you cite the original authors (see README).
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

from fairpath.config import load_dataset_spec, load_settings

log = logging.getLogger(__name__)


def _find_member(zf: zipfile.ZipFile, member: str) -> bytes | None:
    for info in zf.infolist():
        name = Path(info.filename).name
        if name == member:
            return zf.read(info)
        if name.endswith(".zip"):  # UCI nests student.zip inside the outer archive
            inner = zipfile.ZipFile(io.BytesIO(zf.read(info)))
            found = _find_member(inner, member)
            if found is not None:
                return found
    return None


def download_dataset(name: str, force: bool = False, timeout: int = 60) -> Path:
    settings = load_settings()
    spec = load_dataset_spec(name)
    target = settings.path("data_raw") / spec.file
    if target.exists() and not force:
        log.info("%s already present at %s", name, target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s from %s", name, spec.download_url)
    req = urllib.request.Request(spec.download_url, headers={"User-Agent": "fairpath/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(payload))
    data = _find_member(zf, spec.zip_member or spec.file)
    if data is None:
        raise RuntimeError(f"Could not find {spec.zip_member} inside the archive from {spec.download_url}")
    target.write_bytes(data)
    log.info("Saved %s (%d bytes)", target, len(data))
    return target
