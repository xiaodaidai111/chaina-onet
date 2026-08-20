# -*- coding: utf-8 -*-
"""Pre-download BAAI/bge-m3 into the local model cache."""
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.as_posix())
from config import EMBEDDING_MODEL, EMBEDDING_MODEL_CACHE  # noqa: E402

from huggingface_hub import snapshot_download

print("downloading", EMBEDDING_MODEL, "to", EMBEDDING_MODEL_CACHE, flush=True)
snapshot_download(repo_id=EMBEDDING_MODEL, local_dir=EMBEDDING_MODEL_CACHE)
print("model ready", flush=True)