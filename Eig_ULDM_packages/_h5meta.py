"""HDF5 root-attribute helpers for provenance metadata.

Provenance is stored **only as attributes** (never as new groups or
datasets), so readers that iterate over ``f.keys()`` see exactly the
historical layout.  HDF5 limits a compact attribute to about 64 KiB, so long
1-D numeric arrays are split into numbered chunks
(``<name>__chunk000``, ... plus ``<name>__nchunks``); :func:`read_array_attr`
reassembles them transparently.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

import numpy as np

_CHUNK_ELEMS = 4096  # elements per attribute chunk (<= 32 KiB for 8-byte types, well below the 64 KiB limit)


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def write_attrs(attrs, mapping: Mapping[str, Any]) -> None:
    """Write ``mapping`` into an h5py ``AttributeManager``.

    * str / int / float / bool / numpy scalars: stored as is (same as
      ``attrs[k] = v``).
    * 1-D numeric arrays or homogeneous numeric lists: stored as arrays,
      split into chunks when longer than 4096 elements.
    * dicts, mixed lists, None: stored as JSON strings.
    """
    for k, v in mapping.items():
        k = str(k)
        if v is None or isinstance(v, dict):
            attrs[k] = json.dumps(_jsonable(v))
            continue
        if isinstance(v, (list, tuple)):
            try:
                arr = np.asarray(v)
            except Exception:
                arr = None
            if arr is None or arr.dtype == object or arr.dtype.kind in "USO" or arr.ndim != 1:
                attrs[k] = json.dumps(_jsonable(v))
                continue
            v = arr
        if isinstance(v, np.ndarray):
            if v.dtype.kind in "biuf" and v.ndim == 1 and v.size > _CHUNK_ELEMS:
                n = -(-v.size // _CHUNK_ELEMS)
                for i in range(n):
                    attrs[f"{k}__chunk{i:03d}"] = v[i * _CHUNK_ELEMS:(i + 1) * _CHUNK_ELEMS]
                attrs[f"{k}__nchunks"] = n
                continue
            if v.dtype.kind in "biufc":
                attrs[k] = v
            else:
                attrs[k] = json.dumps(_jsonable(v))
            continue
        attrs[k] = v


def read_array_attr(attrs, name: str) -> Optional[np.ndarray]:
    """Return attribute ``name`` as an array (reassembling chunks), or None."""
    if name in attrs:
        return np.asarray(attrs[name])
    key_n = f"{name}__nchunks"
    if key_n in attrs:
        n = int(attrs[key_n])
        parts = [np.asarray(attrs[f"{name}__chunk{i:03d}"]) for i in range(n)]
        return np.concatenate(parts)
    return None


def attrs_to_dict(attrs) -> Dict[str, Any]:
    """Plain dict of attributes (bytes decoded, chunked arrays reassembled)."""
    out: Dict[str, Any] = {}
    chunked = set()
    for k in attrs.keys():
        if k.endswith("__nchunks"):
            chunked.add(k[: -len("__nchunks")])
    for k, v in attrs.items():
        if "__chunk" in k or k.endswith("__nchunks"):
            continue
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        out[k] = v
    for base in chunked:
        out[base] = read_array_attr(attrs, base)
    return out


def software_versions() -> Dict[str, str]:
    """Versions of the packages that determine the numbers in the outputs."""
    vers: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for mod in ("scipy", "h5py", "pyshtools"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            vers[mod] = "not installed"
    try:
        from . import __version__ as tk_version
    except Exception:  # pragma: no cover
        tk_version = "unknown"
    vers["eig_uldm_packages"] = tk_version
    return vers


def provenance(producer: str, **extra: Any) -> Dict[str, Any]:
    """Standard provenance attributes: producer, UTC time, software versions."""
    meta: Dict[str, Any] = {
        "producer": producer,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for k, v in software_versions().items():
        meta[f"version_{k}"] = v
    meta.update(extra)
    return meta
