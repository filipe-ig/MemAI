"""Local embedding for memai, via model2vec static embeddings.

The model (default: minishlab/potion-base-8M, ~30MB, numpy-only
inference) ships bundled in this package (models/potion-base-8M/), so
no network access or Hugging Face download is needed out of the box --
this matters on corporate networks that block huggingface.co. Set
MEMAI_EMBED_MODEL to a Hugging Face repo id or local path to use a
different model instead; it is loaded lazily on first use and cached
for the process lifetime. If it cannot be loaded (e.g. an override
points somewhere unreachable), every function here degrades to
returning None and memai falls back to FTS-only retrieval -- rows
written in that state get their vectors backfilled automatically once
the model is available.

Vectors are L2-normalized here so distance ordering is stable
regardless of the model's own normalization config.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

# Import everything with native extensions eagerly, at module import time
# (server startup, main thread). Importing a C-extension DLL lazily from
# inside an MCP tool call deadlocks on Windows (numpy's multiarray DLL
# load blocks forever once the stdio server's reader threads are up), so
# only the model *weights* may load lazily -- never the libraries.
try:
    import numpy as np
    from model2vec import StaticModel
except Exception:  # pragma: no cover - broken install
    np = None
    StaticModel = None

_BUNDLED_MODEL_DIR = Path(__file__).parent / "models" / "potion-base-8M"

MODEL_NAME = os.environ.get("MEMAI_EMBED_MODEL", str(_BUNDLED_MODEL_DIR))

_model = None
_dim: int | None = None
_load_failed = False


def _get_model():
    global _model, _dim, _load_failed
    if _model is None and not _load_failed:
        if StaticModel is None:
            _load_failed = True
            return None
        try:
            _model = StaticModel.from_pretrained(MODEL_NAME)
            _dim = int(_model.encode(["probe"]).shape[1])
        except Exception:
            _load_failed = True
            _model = None
    return _model


def model_name() -> str:
    return MODEL_NAME


def embedding_dim() -> int | None:
    """Vector dimension of the active model, or None if unavailable."""
    _get_model()
    return _dim


def embed_texts(texts: list[str]) -> list[bytes] | None:
    """Embed texts to float32-packed blobs (sqlite-vec's input format).

    Returns None when the model is unavailable, so callers can skip
    vector writes rather than fail the whole transaction.
    """
    model = _get_model()
    if model is None or not texts:
        return None if model is None else []
    vecs = np.asarray(model.encode(texts), dtype="float32")
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    dim = vecs.shape[1]
    return [struct.pack(f"{dim}f", *row) for row in vecs]
