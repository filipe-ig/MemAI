"""Keep tests hermetic: never load the real model2vec model (network
download on first use). Default = embedder unavailable, so existing
tests exercise the FTS-only degradation path. Vector tests opt into a
deterministic fake embedder via the fake_embedder fixture.
"""

from __future__ import annotations

import math
import re
import struct

import pytest

from memai import embed

# Controlled vocabulary -> vector index. No hashing, no collisions:
# each known word gets its own dimension, unknown words are ignored.
VOCAB = ["car", "maintenance", "schedule", "database", "tuning", "note", "alpha", "beta"]
SYNONYMS = {"automobile": "car", "vehicle": "car"}
FAKE_DIM = len(VOCAB)


def make_fake_embed(dim: int = FAKE_DIM):
    def fake_embed_texts(texts: list[str]) -> list[bytes]:
        out = []
        for t in texts:
            v = [0.0] * dim
            for w in re.findall(r"[a-z]+", t.lower()):
                w = SYNONYMS.get(w, w)
                if w in VOCAB and VOCAB.index(w) < dim:
                    v[VOCAB.index(w)] += 1.0
            n = math.sqrt(sum(x * x for x in v))
            if n == 0:
                v[dim - 1] = 1.0
                n = 1.0
            out.append(struct.pack(f"{dim}f", *[x / n for x in v]))
        return out

    return fake_embed_texts


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch):
    monkeypatch.setattr(embed, "_model", None)
    monkeypatch.setattr(embed, "_dim", None)
    monkeypatch.setattr(embed, "_load_failed", True)


@pytest.fixture
def fake_embedder(monkeypatch):
    monkeypatch.setattr(embed, "embedding_dim", lambda: FAKE_DIM)
    monkeypatch.setattr(embed, "embed_texts", make_fake_embed(FAKE_DIM))
    monkeypatch.setattr(embed, "model_name", lambda: f"fake-model-{FAKE_DIM}d")
    return FAKE_DIM
