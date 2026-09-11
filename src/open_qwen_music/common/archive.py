
from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


def _parse_archive_uri(uri: str) -> tuple[str, Path, str]:
    text = str(uri)
    for scheme in ("tar", "zip"):
        prefix = f"{scheme}://"
        if not text.startswith(prefix):
            continue
        payload = text.removeprefix(prefix)
        try:
            archive, member = payload.split("::", 1)
        except ValueError as exc:
            raise ValueError(
                f"{scheme} URI must be {prefix}<archive>::<member>"
            ) from exc
        if not archive or not member:
            raise ValueError(
                f"{scheme} URI of archive/member must not be empty:{text!r}"
            )
        return scheme, Path(archive), member
    raise ValueError(f"only supports tar:// or zip:// URI,received {text!r}")


def read_archive_bytes(uri: str) -> bytes:

    scheme, archive_path, member = _parse_archive_uri(uri)
    if scheme == "tar":
        with tarfile.open(archive_path, "r:*") as archive:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"tar member does not exist:{member}")
            return extracted.read()
    with zipfile.ZipFile(archive_path) as archive:
        return archive.read(member)
