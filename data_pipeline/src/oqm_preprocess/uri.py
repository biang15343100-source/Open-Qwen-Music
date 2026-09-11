
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

SCHEME_LOOSE = "file"
SCHEME_ZIP = "zip"
SCHEME_TAR = "tar"
SCHEME_TARGZ = "targz"
SCHEME_PARQUET = "parquet"
SCHEME_SQLITE = "sqlite"

MEMBER_SCHEMES = frozenset({SCHEME_ZIP, SCHEME_TAR, SCHEME_TARGZ})
PARAM_SCHEMES = frozenset({SCHEME_PARQUET, SCHEME_SQLITE})
ALL_SCHEMES = frozenset({SCHEME_LOOSE, *MEMBER_SCHEMES, *PARAM_SCHEMES})


_STORAGE_TO_SCHEME = {
    "loose": SCHEME_LOOSE,
    "zip": SCHEME_ZIP,
    "tar": SCHEME_TAR,
    "targz": SCHEME_TARGZ,
    "parquet": SCHEME_PARQUET,
    "sqlite": SCHEME_SQLITE,
}
_SCHEME_TO_STORAGE = {v: k for k, v in _STORAGE_TO_SCHEME.items()}

_CLIP_RE = re.compile(r"#t=(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)$")
_SEP = "::"


class UriError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AudioRef:

    storage_class: str
    container: Path
    member: str = ""
    params: dict[str, str] | None = None
    clip_start_sec: float | None = None
    clip_end_sec: float | None = None

    @property
    def scheme(self) -> str:
        return _STORAGE_TO_SCHEME[self.storage_class]

    @property
    def is_clip(self) -> bool:
        return self.clip_start_sec is not None or self.clip_end_sec is not None

    @property
    def sequential_only(self) -> bool:
        return self.storage_class == "targz"

    def with_clip(self, start: float | None, end: float | None) -> AudioRef:
        return replace(self, clip_start_sec=start, clip_end_sec=end)

    def to_uri(self) -> str:
        return build(
            self.storage_class,
            self.container,
            self.member,
            params=self.params,
            clip_start_sec=self.clip_start_sec,
            clip_end_sec=self.clip_end_sec,
        )

    def __str__(self) -> str:
        return self.to_uri()


def build(
    storage_class: str,
    container: str | Path,
    member: str = "",
    *,
    params: dict[str, str] | None = None,
    clip_start_sec: float | None = None,
    clip_end_sec: float | None = None,
) -> str:
    scheme = _STORAGE_TO_SCHEME.get(storage_class)
    if scheme is None:
        raise UriError(f"Unknown storage_class {storage_class!r}")

    path = str(container)
    if not path.startswith("/"):
        raise UriError(f"The container path must be an absolute path: {path!r}")
    if _SEP in path:
        raise UriError(f"The container path must not contain {_SEP!r}: {path!r}")

    uri = f"{scheme}://{path}"

    if scheme in MEMBER_SCHEMES:
        if not member:
            raise UriError(f"{scheme} must provide member")
        uri += _SEP + member
    elif scheme in PARAM_SCHEMES:
        if not params:
            raise UriError(f"{scheme} must provide params")
        uri += _SEP + "&".join(f"{k}={v}" for k, v in params.items())
    elif member:
        raise UriError(f"{scheme} does not accept member(should be written directly into the container path)")

    if clip_start_sec is not None or clip_end_sec is not None:
        start = 0.0 if clip_start_sec is None else float(clip_start_sec)
        end = -1.0 if clip_end_sec is None else float(clip_end_sec)
        uri += f"#t={start:g},{end:g}"
    return uri


def parse(uri: str) -> AudioRef:
    if not isinstance(uri, str) or "://" not in uri:
        raise UriError(f"is not legal URI: {uri!r}")

    body = uri
    clip_start: float | None = None
    clip_end: float | None = None
    m = _CLIP_RE.search(body)
    if m:
        body = body[: m.start()]
        clip_start = float(m.group(1))
        clip_end = float(m.group(2))
        if clip_start < 0:
            clip_start = None
        if clip_end < 0:
            clip_end = None

    scheme, _, rest = body.partition("://")
    if scheme not in ALL_SCHEMES:
        raise UriError(f"Unknown scheme {scheme!r},optional {sorted(ALL_SCHEMES)}")
    if not rest.startswith("/"):
        raise UriError(f"The container path must be an absolute path: {uri!r}")

    container_str, sep, tail = rest.partition(_SEP)
    container = Path(container_str)
    storage_class = _SCHEME_TO_STORAGE[scheme]

    member = ""
    params: dict[str, str] | None = None
    if scheme in MEMBER_SCHEMES:
        if not sep or not tail:
            raise UriError(f"{scheme} is missing {_SEP}member part: {uri!r}")
        member = tail
    elif scheme in PARAM_SCHEMES:
        if not sep or not tail:
            raise UriError(f"{scheme} is missing {_SEP}parameter part: {uri!r}")
        params = _parse_params(tail, uri)
    elif sep:
        raise UriError(f"{scheme} should not appear {_SEP}: {uri!r}")

    return AudioRef(
        storage_class=storage_class,
        container=container,
        member=member,
        params=params,
        clip_start_sec=clip_start,
        clip_end_sec=clip_end,
    )


def _parse_params(tail: str, uri: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in tail.split("&"):
        if not chunk:
            continue
        key, eq, value = chunk.partition("=")
        if not eq:
            raise UriError(f"The parameter section is missing '=': {chunk!r} in {uri!r}")
        out[key] = value
    if not out:
        raise UriError(f"The parameter section is empty: {uri!r}")
    return out
